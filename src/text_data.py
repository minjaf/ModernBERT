# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 OLMo authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import logging
import math
import os
import random
import sys
import json
from itertools import islice
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataset, StreamingDataLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner
from composer.utils import dist
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Mapping
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding

from filelock import FileLock
from contextlib import nullcontext
import h5py
from Bio.Seq import Seq
from fractions import Fraction

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sequence_packer import BufferedIterable, GreedyBestFitSequencePacker

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


# Subclass DistributedSampler to use PCG64DXSM for shuffling
class DistributedSamplerPCG64DXSM(DistributedSampler):
    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            # use numpy's RNG PCG64DXSM instead of torch.randperm
            rng = np.random.Generator(np.random.PCG64DXSM(self.seed + self.epoch))
            indices = rng.permutation(len(self.dataset)).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_tokenizer(
    om_tokenizer_config: DictConfig,
) -> Tokenizer:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    resolved_om_tokenizer_config = om.to_container(om_tokenizer_config, resolve=True)
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        "kwargs", {}
    )
    tokenizer_name = resolved_om_tokenizer_config["name"]  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get(
        "model_max_length",
        int(1e30),
    )

    return tokenizer


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_len: int,
        pad_sequences: bool,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[int] = None,
        predownload: int = 100_000,
        partition_algo: str = "orig",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1s",
        shuffle_seed: int = 9176,
        cache_limit: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        group_method = kwargs.pop("group_method", None)
        if group_method is not None:
            raise NotImplementedError(
                "group_method is deprecated and has been removed.\nTo "
                + "concatenate, use the --concat_tokens "
                + "argument when creating your MDS dataset with concat_c4.py"
            )

        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(f"StreamingTextDataset() got an unexpected keyword argument: {kwargs}")

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(f"local directory {local} does not contain split {split}")

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            cache_limit=cache_limit,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_sequences = pad_sequences

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

        return self.tokenizer(text_sample["text"],
                              truncation=True,
                              padding="max_length" if self.pad_sequences else False,
                              max_length=self.max_seq_len)

    def _read_binary_tokenized_sample(self, sample: BatchEncoding):
        if not self.pad_sequences:
            raise RuntimeError(
                "packing with pre-tokenized data is not implemented, make sure that there is no paddings.")

        seq_len = sample["len"] if "len" in sample else len(sample["input_ids"])

        input_ids = np.frombuffer(sample["input_ids"], dtype=np.int64).copy()
        if "attention_mask" in sample:
            attention_mask = np.frombuffer(sample["attention_mask"], dtype=np.int64).copy()
        else:
            attention_mask = np.ones_like(input_ids)

        # calculate padding
        pad_len = self.max_seq_len - seq_len

        # pad or truncate input_ids and attention_mask
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id)
            attention_mask = np.pad(attention_mask, (0, pad_len), constant_values=0)
        elif pad_len < 0:
            input_ids = input_ids[: self.max_seq_len]
            attention_mask = attention_mask[: self.max_seq_len]

        token_type_ids = np.zeros(self.max_seq_len, dtype=np.int64)

        return BatchEncoding(
            data={
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "token_type_ids": token_type_ids.tolist(),
            },
            n_sequences=1,
        )

    # How to process a sample
    def __getitem__(self, idx: int) -> Union[Dict[str, Any], torch.Tensor]:
        sample = super().__getitem__(idx)
        if "text" in sample:
            token_sample = self._tokenize(sample)
        elif "input_ids" in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError("StreamingTextDataset needs samples to have a `text` or `input_ids` column")
        return token_sample


# We need a custom collator for MLM probs propagation
class DataCollatorForLanguageModelingWithMLMProbs(transformers.DataCollatorForLanguageModeling):
    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError("TF is not supported for MLM with MLM probs")
    
    # TODO: implement masking probabilities
    def torch_mask_tokens(self, inputs: Any, 
                          special_tokens_mask: Optional[Any] = None,
                          mask_probs_array: Optional[Any] = None,
                          ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        Based on the original implementation of transformers' `DataCollatorForLanguageModeling.torch_mask_tokens`
        
        It performs masking in a way that produces on expectation the following masked inputs:
         - (1-self.mlm_probability) of the original positions will be untouched.
         - self.mlm_probability * 80%  of the original positions get replaced with a mask token
         - self.mlm_probability * 10%  of the original positions get replaced with a random token
         - self.mlm_probability * 10%  of the original positions also remain untouched.
        This generates the masked_inputs.

        It also generates a labels array, which has ignore tokens in the (1-mask_prob) positions

        These proportions are expectation values since the random transformation is performed
        """
        # TODO: check if any corrections are needed to tenzors that are not 2D
        assert len(inputs.shape) == 2, f"inputs must be a 2D tensor, bsize * input_len, but got {inputs.shape}"
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        Num_non_special_tokens = (~special_tokens_mask).sum(dim=1).float()
       
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if mask_probs_array is not None:
            # Validate that mask_probs_array has the same shape as seq
            if mask_probs_array.shape != labels.shape:
                raise ValueError(f"mask_probs_array shape {mask_probs_array.shape} must match labels shape {labels.shape}")
            
            # Normalize mask_probs_array so that the average probability equals mask_prob
            # This ensures that approximately mask_prob * 100% of tokens are masked overall
            mask_probs_array.masked_fill_(special_tokens_mask, value=0.0)
            # average across non-special-tokens
            # if denominator is 0, set it to 1 and convert to float
            denominator = torch.where(Num_non_special_tokens == 0, 
                                    torch.ones_like(Num_non_special_tokens), 
                                    Num_non_special_tokens).float()
            avg_prob = torch.sum(mask_probs_array, dim=1) / denominator
            assert torch.all(avg_prob <= 1.0), f"avg_prob {avg_prob} must be less than 1.0"
            assert torch.all(avg_prob > 0.0), f"avg_prob {avg_prob} must be greater than 0.0"
            # Scale the probabilities to maintain the target average
            mask_probs = mask_probs_array * (self.mlm_probability / avg_prob.unsqueeze(1))
            # Clip to ensure probabilities stay in [0, 1] range
            mask_probs = torch.clip(mask_probs, 0.0, 1.0)
            probability_matrix = mask_probs
        else:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # compute fraction of masked tokens
        # print ("--------------------------------")
        # print (f"mlm_probability: {self.mlm_probability}")
        # print (f"Num_non_special_tokens: {Num_non_special_tokens}")
        # fraction_masked = masked_indices.sum(dim=1).float() / Num_non_special_tokens
        # print (f"Average fraction of masked tokens: {fraction_masked.mean()}")
        # # print (f"Fraction of masked tokens: {fraction_masked}")
        # print ("--------------------------------")

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError("Numpy is not supported for MLM with MLM probs")

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        raise NotImplementedError("Numpy is not supported for MLM with MLM probs")

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            # we don't need padding here, because we do pad ourselves in the dataset
            # batch = pad_without_fast_tokenizer_warning(
            #     self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            # )
            # handle mlm_efficiency_path
            if "mlm_efficiency_path" in examples[0]:
                mlm_efficiency_paths = [example.pop("mlm_efficiency_path") for example in examples]
            else:
                mlm_efficiency_paths = None
            
            # now handle all tensors
            batch = default_data_collator(examples, self.return_tensors)
            
            # and keep the mlm_efficiency_path as single value in the batch
            if mlm_efficiency_paths is not None: 
                batch["mlm_efficiency_path"] = mlm_efficiency_paths
        else:
            raise NotImplementedError("Only dicts are supported for MLM with MLM probs")
            # batch = {
            #     "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            # }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            if "MLM_probs" in batch:
                batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
					batch["input_ids"], special_tokens_mask=special_tokens_mask, 
                    mask_probs_array=batch["MLM_probs"]
				)
            else:
                batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
					batch["input_ids"], special_tokens_mask=special_tokens_mask
				)
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(self, base_collator: Callable, eos_token_id: Optional[int] = None, bos_token_id: Optional[int] = None):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError("Must supply a value for either eos_token_id or bos_token_id, but got None for both.")
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                "Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. "
                + "Please supply `eos_token_id` if sequences end with an EOS token, or use "
                + "`bos_token_id` if sequences start with a BOS token."
            )
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True
        else:
            self.split_token_id = eos_token_id
            self.bos_mode = False

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.split_token_id is not None
        is_separator = torch.eq(batch["input_ids"], self.split_token_id)
        cumulative_sep = torch.cumsum(is_separator, dim=1).to(batch["input_ids"].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


def build_streaming_dataset(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    pad_sequences: bool,
    device_batch_size: int,
):
    # build streams
    streams_dict = cfg.dataset.get("streams", None)
    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            streams.append(
                Stream(
                    remote=stream.get("remote", None) or cfg.dataset.get("remote", None),
                    local=stream.get("local", None) or cfg.dataset.get("local", None),
                    split=stream.get("split", None) or cfg.dataset.get("split", None),
                    proportion=stream.get("proportion", None),
                    repeat=stream.get("repeat", None),
                    choose=stream.get("choose", None),
                    download_retry=stream.get("download_retry", None) or cfg.dataset.get("download_retry", 2),
                    download_timeout=stream.get("download_timeout", None) or cfg.dataset.get("download_timeout", 60),
                    validate_hash=stream.get("validate_hash", None) or cfg.dataset.get("validate_hash", None),
                    keep_zip=stream.get("keep_zip", None) or cfg.dataset.get("keep_zip", False),
                )
            )

    # build dataset potentially with streams
    # todo: make StreamingGenomeDataset similar to NoStreamingGenomeDataset
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        max_seq_len=cfg.dataset.max_seq_len,
        pad_sequences=pad_sequences,
        streams=streams,
        remote=cfg.dataset.get("remote", None),
        local=cfg.dataset.get("local", None),
        split=cfg.dataset.get("split", None),
        download_retry=cfg.dataset.get("download_retry", 2),
        download_timeout=cfg.dataset.get("download_timeout", 60),
        validate_hash=cfg.dataset.get("validate_hash", None),
        keep_zip=cfg.dataset.get("keep_zip", False),
        epoch_size=cfg.dataset.get("epoch_size", None),
        predownload=cfg.dataset.get("predownload", 100_000),
        partition_algo=cfg.dataset.get("partition_algo", "orig"),
        num_canonical_nodes=cfg.dataset.get("num_canonical_nodes", 128),
        batch_size=device_batch_size,
        shuffle=cfg.dataset.get("shuffle", False),
        shuffle_algo=cfg.dataset.get("shuffle_algo", "py1s"),
        shuffle_seed=cfg.dataset.get("shuffle_seed", 9176),
        cache_limit=cfg.dataset.get("cache_limit", None),
    )
    return dataset


def build_no_streaming_dataset(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    pad_sequences: bool = True,
):
    if cfg.dataset.get('data_type', None) == 'genome':
        return NoStreamingGenomeDataset(
            tokenizer=tokenizer,
            local=cfg.dataset.get("local", None),
            split=cfg.dataset.get("split", None),
            max_seq_len=cfg.dataset.max_seq_len,
            pad_sequences=pad_sequences,
            augment_rc=cfg.dataset.get("augment_rc", False),
            sample_chunk=cfg.dataset.get("sample_chunk", False),
            min_seq_len=cfg.dataset.get("min_seq_len", 10),
            mlm_efficiency_path=cfg.dataset.get("mlm_efficiency_path", None),
            mlm_efficiency_in_shards=cfg.dataset.get("mlm_efficiency_in_shards", False),            
            append_mlm_efficiency=cfg.dataset.get("append_mlm_efficiency", False),
            fully_precomputed_mlm_efficiency=cfg.dataset.get("fully_precomputed_mlm_efficiency", False),
            use_mlm_efficiency_frequency=cfg.dataset.get("use_mlm_efficiency_frequency", 1),
            mask_probabilities_inverted=cfg.dataset.get("mask_probabilities_inverted", False),
            split_as_subsample_from_train_split=cfg.dataset.get("split_as_subsample_from_train_split", False),
        )
    else:
        return NoStreamingDataset(
            tokenizer=tokenizer,
            local=cfg.dataset.get("local", None),
            split=cfg.dataset.get("split", None),
            max_seq_len=cfg.dataset.max_seq_len,
            pad_sequences=pad_sequences,
        )


def build_text_dataloader(
    cfg: DictConfig,
    tokenizer: Tokenizer,
    device_batch_size: int,
    device_microbatch_size: int,
):
    assert cfg.name == "text", f"Tried to build text dataloader with cfg.name={cfg.name}"
    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            + "concatenate, use the --concat_tokens "
            + "argument when creating your MDS dataset with convert_dataset.py"
        )
    use_sequence_packing = cfg.get("sequence_packing", False)
    if cfg.dataset.get("streaming", True):
        dataset = build_streaming_dataset(cfg, tokenizer, pad_sequences=not use_sequence_packing,
                                          device_batch_size=device_batch_size)
        sampler = None
    else:
        assert cfg.dataset.get("local", None) is not None, "Local path must be provided when not using streaming"
        # sequence packing should never use padded sequences, regular dataloaders may if tokenizing on the fly
        dataset = build_no_streaming_dataset(cfg, tokenizer=tokenizer, pad_sequences=not use_sequence_packing) # seems to be a bug in the original code
        sampler = DistributedSamplerPCG64DXSM(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_global_rank(),
            shuffle=cfg.dataset.get("shuffle", False),
            seed=cfg.dataset.get("shuffle_seed", 9176),
            drop_last=cfg.drop_last,
        )

    mlm_probability = cfg.dataset.get("mlm_probability", None)
    use_adaptive_mlm_probability = (cfg.dataset.get("mlm_efficiency_path", None) is not None) or \
                                    cfg.dataset.get("mlm_efficiency_in_shards", False)

    # only use sequence packing if using the no_streaming_dataset
    # if not cfg.dataset.get("streaming", True) and cfg.get("sequence_packing", False):
    if use_sequence_packing:
        # if use_adaptive_mlm_probability: # TODO: implement adaptive MLM probability
        #     raise NotImplementedError("Adaptive MLM probability is not supported for sequence packing")
            
        if cfg.dataset.get("streaming", True):
            # streaming dataset already handles splitting and shuffling data for each rank
            dataloader = DataLoader(
                dataset,
                collate_fn=lambda x: x,
                batch_size=device_batch_size,
                drop_last=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.get("pin_memory", True),
                prefetch_factor=cfg.get("prefetch_factor", 2),
                persistent_workers=cfg.get("persistent_workers", True),
                timeout=cfg.get("timeout", 0),
                sampler=sampler,
            )
        else:
            dataloader = DataLoader(
                dataset,
                collate_fn=lambda x: x,
                batch_size=device_batch_size,
                drop_last=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.get("pin_memory", True),
                prefetch_factor=cfg.get("prefetch_factor", 2),
                persistent_workers=cfg.get("persistent_workers", True),
                timeout=cfg.get("timeout", 0),
                sampler=sampler,
            )
        # todo: make GreedyBestFitSequencePacker and BufferedIterable inherit from dataloaders
        # to save state of StreamDataset to ckpt['state']['dataset_state']
        # check how dataset is extracted from dataloader in composer.core.state:_dataset_of
        # another possible way is to manually set state of dataset when resuming training
        sequence_packer = GreedyBestFitSequencePacker.from_composer(
            dataloader,
            batch_size=device_batch_size,
            micro_batch_size=device_microbatch_size,
            max_seq_len=cfg.dataset.max_seq_len,
            buffer_size=cfg.get("packing_buffer_size", 5 * device_batch_size),
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            mask_prob=mlm_probability,
            seed=cfg.dataset.get("shuffle_seed", 42),
            batch_size_warmup_min_size=cfg.get("batch_size_warmup_min_size", None),
            batch_size_warmup_tokens=cfg.get("batch_size_warmup_tokens", None),
            world_size=dist.get_world_size(),
        )
        buffered_iterable = BufferedIterable(sequence_packer, buffer_size=cfg.get("packing_prefetch_factor", 5))
        return buffered_iterable
    else:
        collate_fn = DataCollatorForLanguageModelingWithMLMProbs(
            tokenizer=dataset.tokenizer, mlm=mlm_probability is not None, mlm_probability=mlm_probability
        )

        # collate_fn = transformers.DataCollatorForLanguageModeling(
        #     tokenizer=dataset.tokenizer, mlm=mlm_probability is not None, mlm_probability=mlm_probability
        # )

        eos_token_id = cfg.dataset.get("eos_token_id")
        bos_token_id = cfg.dataset.get("bos_token_id")
        if (eos_token_id is not None) or (bos_token_id is not None):
            # Note: Will raise an error if both are non-None
            collate_fn = ConcatenatedSequenceCollatorWrapper(
                base_collator=collate_fn, eos_token_id=eos_token_id, bos_token_id=bos_token_id
            )

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=device_batch_size,
            drop_last=cfg.drop_last,
            num_workers=cfg.num_workers,
            pin_memory=cfg.get("pin_memory", True),
            prefetch_factor=cfg.get("prefetch_factor", 2),
            persistent_workers=cfg.get("persistent_workers", True),
            timeout=cfg.get("timeout", 0),
            sampler=sampler,
        )


class NoStreamingDataset(Dataset):
    """
    A dataset class that can read data with raw mds-format (mosaic streaming-format without compression)
    from local. In comparison with `StreamingTextDataset` that also can read data with mds-format from local,
    this class is slimmer, more efficient, and does not contain redundant code required for streaming.
    """

    def __init__(
        self,
        local: str,
        split: Optional[str],
        max_seq_len: int,
        tokenizer: Optional[Tokenizer] = None,
        pad_sequences: bool = True,
    ) -> None:
        super().__init__()
        if split is not None:
            split_path = os.path.join(local, split)
        else:
            split_path = local
        index_file_path = os.path.join(split_path, "index.json")
        obj = json.load(open(index_file_path))
        self.shards = []
        for info in obj["shards"]:
            shard = reader_from_json(local, split, info)
            raw_filename = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
            assert os.path.isfile(raw_filename), f"Raw file {raw_filename} does not exist"
            shard.validate(True)
            self.shards.append(shard)
        samples_per_shard = np.array([shard.samples for shard in self.shards], np.int64)
        self.len = samples_per_shard.sum()
        self.spanner = Spanner(samples_per_shard)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_sequences = pad_sequences

    def _tokenize(self, text_sample):
        assert self.tokenizer is not None, "Tokenizer required if data is not pretokenized"
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

        return self.tokenizer(
            text_sample["text"],
            truncation=True,
            padding="max_length" if self.pad_sequences else False,
            max_length=self.max_seq_len,
        )

    def __getitem__(self, index: int):
        shard_id, shard_sample_id = self.spanner[index]
        shard = self.shards[shard_id]
        sample = shard[shard_sample_id]
        if "input_ids" in sample:
            for k in list(sample.keys()):
                if isinstance(sample[k], np.ndarray):
                    sample[k] = sample[k][: self.max_seq_len]
                else:
                    del sample[k]
            if "attention_mask" not in sample:
                sample["attention_mask"] = np.ones_like(sample["input_ids"])
            return sample
        elif "text" in sample:
            return self._tokenize(sample)
        else:
            RuntimeError("Data sample must contain a field with `input_ids` or `text`")

    def __len__(self):
        return self.len


class NoStreamingGenomeDataset(NoStreamingDataset):
    MLM_PROB_DTYPE = np.float16

    def __init__(
        self,
        local: str,
        split: Optional[str],
        max_seq_len: int,
        split_as_subsample_from_train_split: Union[bool, float] = False, # if float, use the fraction of the train split to create other splits (default: False)
        tokenizer: Optional[Tokenizer] = None,
        pad_sequences: bool = True,
        augment_rc: bool = False,
        sample_chunk: bool = False,
        min_seq_len: int = 10,
        mlm_efficiency_path: str | None = None,
        mlm_efficiency_in_shards: bool = False,
        append_mlm_efficiency: bool = False,
        fully_precomputed_mlm_efficiency: bool = False,
        use_mlm_efficiency_frequency: float = 1.0,
        mask_probabilities_inverted: bool = False,
    ) -> None:
        """
        Args:
            local: path to the local directory
            split: split to use
            max_seq_len: maximum sequence length
            split_as_subsample_from_train_split: if True, use the train split to create the valid split; if float, use the fraction of the train split to create the valid split (default: False)
            tokenizer: tokenizer to use
            pad_sequences: if True, pad sequences to the maximum sequence length
            augment_rc: if True, augment the sequences with reverse complement
            sample_chunk: if True, sample chunks of sequences
            min_seq_len: minimum sequence length
            mlm_efficiency_path: path to the mlm efficiency file (default: None)
            mlm_efficiency_in_shards: if True, use the mlm efficiency in shards (default: False)
            append_mlm_efficiency: if True, append the mlm efficiency to the file (default: False)
            fully_precomputed_mlm_efficiency: if True, use the fully precomputed mlm efficiency (default: False)
            use_mlm_efficiency_frequency: frequency of using mlm efficiency (default: 1.0)
            mask_probabilities_inverted: if True, invert the mask probabilities (default: False)
        """

        if isinstance(split_as_subsample_from_train_split, float):
            assert 0 < split_as_subsample_from_train_split < 1, "split_as_subsample_from_train_split must be a float between 0 and 1"
            self.split_as_subsample_from_train_split = split_as_subsample_from_train_split
            self.split = split
            # always init as train to get all samples
            super().__init__(local, "train", max_seq_len, tokenizer, pad_sequences)  
            frac = Fraction(split_as_subsample_from_train_split).limit_denominator(max_denominator=super().__len__()-1)           
            self.frac_p = frac.numerator
            self.frac_q = frac.denominator
            logger.info(f"Creating valid split as a subsample from train split. Split: {split}, fraction: {self.frac_p}/{self.frac_q}")
        else:
            assert not split_as_subsample_from_train_split, "split_as_subsample_from_train_split must be a float between 0 and 1 or False"
            super().__init__(local, split, max_seq_len, tokenizer, pad_sequences)

        # todo: add seed and check that its ok for multiple workers
        self.augment_rc = augment_rc
        self.sample_chunk = sample_chunk
        self.min_seq_len = min_seq_len
        self.mlm_efficiency_path = mlm_efficiency_path
        self.mlm_efficiency_in_shards = mlm_efficiency_in_shards
        self.mask_probabilities_inverted = mask_probabilities_inverted
        self.fully_precomputed_mlm_efficiency = fully_precomputed_mlm_efficiency

        # get mean token length for tokenizer
        tokens = self.tokenizer.get_vocab()

        try:
            all_special_tokens = self.tokenizer.special_tokens_map["additional_special_tokens"]
        except KeyError:
            all_special_tokens = []

        for k,v in self.tokenizer.special_tokens_map.items():
            if k != "additional_special_tokens":
                all_special_tokens.append(v)
                assert isinstance(v, str), f"Special token {v} is not a string"
        
        # exclude special tokens
        tokens = {k: v for k, v in tokens.items() if k not in all_special_tokens}

        token_lengths = [len(token) for token in tokens.keys()]
        self.mean_token_length = sum(token_lengths) / len(token_lengths)
        # print (f"Mean token length: {self.mean_token_length}") # debug
    
        if self.mlm_efficiency_path is not None or self.mlm_efficiency_in_shards:
            if self.mlm_efficiency_path and self.mlm_efficiency_in_shards:
                raise ValueError("Specify either mlm_efficiency_in_shards or mlm_efficiency_in_shards, not both")
            self.use_mlm_efficiency_frequency = use_mlm_efficiency_frequency
            if split is None:
                raise ValueError("Split is required to save mlm efficiency")
            if self.mlm_efficiency_path is not None:
                self.mlm_efficiency_path = self._get_full_mlm_efficiency_path(split)
                if not fully_precomputed_mlm_efficiency:
                    os.makedirs(self.mlm_efficiency_path, exist_ok=append_mlm_efficiency)
            else:
                assert not append_mlm_efficiency, "Can not append mlm efficiency when it is in mds files"

    def _get_full_mlm_efficiency_path(self, split: str) -> str:
        if self.mlm_efficiency_path is None:
            return None
        return os.path.join(self.mlm_efficiency_path, split + "/")

    def __len__(self):
        if self.split_as_subsample_from_train_split:
            n = super().__len__()
            train_len = (n // self.frac_q) * self.frac_p + min(n % self.frac_q, self.frac_p) 
            if self.split == "train":
                return train_len   # number of kept ids
            else:
                return n - train_len  # number of kept ids
        else:
            return super().__len__()

    def _tokenize(self, text_sample):
        assert self.tokenizer is not None, "Tokenizer required if data is not pretokenized"
        if not hasattr(self.tokenizer, "_pad_token"):
            self.tokenizer._pad_token = self.tokenizer.pad_token_id
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

        text = text_sample['text']

        st_index = 0
        max_seq_len = self.max_seq_len
        if self.sample_chunk:
            # todo: do we really want uniform length sampling here?
            max_seq_len = random.randint(self.min_seq_len, self.max_seq_len)
            # choose start index somewhere in the first half
            st_index = random.randint(0, max(1, len(text) - int(self.mean_token_length*max_seq_len*1.1))) # 1.1 is a safety factor

        text = text[st_index:]
        text = text[:max_seq_len * 10]  # cut to make tokenization faster if text is too long

        if self.augment_rc and random.random() > 0.5:
            if self.mlm_efficiency_path or self.mlm_efficiency_in_shards:
                raise NotImplementedError("Reverse complement is not supported for adaptive masking")
            text = str(Seq(text).reverse_complement())

        if self.mlm_efficiency_path or self.mlm_efficiency_in_shards:
            # we need to 
            # 1) load the mlm efficiency data
            # 2) propagate offeset mapping through the model to save predicts later
            result = self.tokenizer(
                text,
                truncation=True,
                padding="max_length" if self.pad_sequences else False,
                max_length=self.max_seq_len if self.pad_sequences else max_seq_len, # if we have sequence packing, we do not padd and truncate to max_seq_len; else we should have all sequences  having strictly the same length
                return_offsets_mapping=True,
            )
            result["offsets_mapping_starts"] = [st_index + offset[0] for offset in result["offset_mapping"]]
            result["offsets_mapping_ends"] = [st_index + offset[1] for offset in result["offset_mapping"]]
            del result["offset_mapping"]
            assert len(result["input_ids"]) == len(result["offsets_mapping_starts"]) == len(result["offsets_mapping_ends"]), "Offsets mapping and input_ids have different lengths"
        else:
            result = self.tokenizer(
                text,
                truncation=True,
                padding="max_length" if self.pad_sequences else False,
                max_length=self.max_seq_len if self.pad_sequences else max_seq_len, # if we have sequence packing, we do not padd and truncate to max_seq_len; else we should have all sequences  having strictly the same length
            )
        
        return result

    def __getitem__(self, index: int):
        # import datetime
        # tstamp = datetime.datetime.now()
        # print (f"fetch {index}, {self.mlm_efficiency_path}")

        if self.split_as_subsample_from_train_split:
            if self.split == "train":
                index = (index // self.frac_p) * self.frac_q + index % self.frac_p
            else:
                r = self.frac_q - self.frac_p
                index = (index // r) * self.frac_q + self.frac_p + (index % r)
            assert index < super().__len__(), f"Index {index} is out of bounds for split {self.split} with length {super().__len__()}, frac_p: {self.frac_p}, frac_q: {self.frac_q}"

        shard_id, shard_sample_id = self.spanner[index]
        shard = self.shards[shard_id]
        sample = shard[shard_sample_id]
        if "text" in sample:
            result = self._tokenize(sample)
            
            if self.mlm_efficiency_path or self.mlm_efficiency_in_shards: # going to return precomputed mlm probs
                default_mlm_prob = 1.0 if not self.mask_probabilities_inverted else 0.0
                use_mlm_efficiency = random.random() < self.use_mlm_efficiency_frequency
                start_index = min(result["offsets_mapping_starts"])
                end_index = max(result["offsets_mapping_ends"])
                assert end_index - start_index <= len(sample['text'])
                assert end_index - start_index >= 0

                if not use_mlm_efficiency: # skip precomputed mlms and return defaults
                    mlm_probs = np.zeros(end_index - start_index, dtype=self.MLM_PROB_DTYPE)
                    mlm_probs += default_mlm_prob
                    result["mlm_efficiency_path"] = self.mlm_efficiency_path if self.mlm_efficiency_path else "none"
                elif self.mlm_efficiency_path: # read from hdf files
                    # check if we have hdf_file
                    hdf5_file = os.path.join(self.mlm_efficiency_path, f"shard_{shard_id}.hdf5")
                    result["mlm_efficiency_path"] = self.mlm_efficiency_path

                    if not os.path.exists(hdf5_file):
                        if self.fully_precomputed_mlm_efficiency:
                            raise ValueError("Fully precomputed mlm efficiency is indicated but hdf5 file not found: "+hdf5_file)
                        # print (f"Shard shard_{shard_id}.hdf5 not found") # debug
                        with FileLock(hdf5_file+".lock"):
                            # print (f"Creating shard shard_{shard_id}.hdf5") # debug
                            with h5py.File(hdf5_file, "w") as f:
                                f.create_dataset(str(shard_sample_id), data=[np.nan]*len(sample['text']), dtype=self.MLM_PROB_DTYPE)

                    # check if we have sample in hdf_file
                    # print ("Acquiring mlm probs for ", hdf5_file) # debug
                    mode = "r" if self.fully_precomputed_mlm_efficiency else "a"
                    lock_cm = FileLock(hdf5_file + ".lock") if not self.fully_precomputed_mlm_efficiency else nullcontext()
                    with lock_cm:
                        with h5py.File(hdf5_file, mode) as f:
                            if str(shard_sample_id) not in f:
                                if self.fully_precomputed_mlm_efficiency:
                                    raise ValueError(f"Fully precomputed mlm efficiency is indicated but sample not found in hdf5 file: {shard_sample_id}")
                                # print (f"Sample {shard_sample_id} not found in shard_{shard_id}.hdf5") # debug
                                f.create_dataset(str(shard_sample_id), data=[np.nan]*len(sample['text']), dtype=self.MLM_PROB_DTYPE)

                            # print (f"------>>>>>use_mlm_efficiency = {use_mlm_efficiency}") # debug
                            mlm_probs = f[str(shard_sample_id)][start_index:end_index]
                else: # read from shard
                    mlm_probs = sample["MLM"]
                    result["mlm_efficiency_path"] = "none"

                mlm_probs = np.nan_to_num(mlm_probs, nan=default_mlm_prob, copy=False) # probability=1.0 for unseen positions

                if self.mask_probabilities_inverted:
                    mlm_probs = 1.0 - mlm_probs

                # mlm_probs are stored in base-pair resolution, so we need to convert them to bpe-token resolution according to offsets_mapping
                mlm_probs_list = []
                non_pad_MLM_probs = []
                for tok_start,tok_end in zip(result["offsets_mapping_starts"], result["offsets_mapping_ends"]):
                    if tok_end - tok_start == 0: # service tokens and padding
                        mlm_probs_list.append(0.)
                    elif tok_end - tok_start > 0:
                        mlm_probs_list.append(np.mean(mlm_probs[tok_start-start_index:tok_end-start_index]))
                        non_pad_MLM_probs.append(np.mean(mlm_probs[tok_start-start_index:tok_end-start_index]))
                    else:
                        raise ValueError(f"tok_end - tok_start = {tok_end - tok_start} is negative")
                assert len(mlm_probs_list)==len(result["offsets_mapping_starts"])==len(result["offsets_mapping_ends"])
                if len(non_pad_MLM_probs) > 0:
                    all_zeros = all([m==0 for m in non_pad_MLM_probs])
                    if all_zeros: # add some constant to avoid all 0.0s
                        for index, (m,st,en) in enumerate(zip(mlm_probs_list,
                                                                result["offsets_mapping_starts"], 
                                                                result["offsets_mapping_ends"])
                                                        ):
                            if en != st: # this is not service token
                                mlm_probs_list[index] += 0.5
                    result["mean_non_pad_MLM_probs"] = np.mean(non_pad_MLM_probs)
                else:
                    raise ValueError("No mlm probs found")
                assert len(mlm_probs_list) == len(result["input_ids"]), "MLM probs and input ids have different lengths"
                result["MLM_probs"] = mlm_probs_list
                # propagate shard id and shard sample id to save MLM probs after we get it
                result["shard_id"] = [shard_id]
                result["shard_sample_id"] = [shard_sample_id]
                return result
            else:
                # raise ValueError("Why we are here? Split is {self.split}")
                return result
        else:
            raise RuntimeError("Data sample must contain a field with `text`")

# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="the name of the tokenizer to use")
    parser.add_argument("--local_path", type=str, required=True, help="the path to the local copy of the dataset")
    parser.add_argument(
        "--remote_path", type=str, default=None, help="the path to the remote copy to stream from (optional)"
    )
    parser.add_argument("--split", type=str, default="val", help="which split of the dataset to use")
    parser.add_argument("--max_seq_len", type=int, default=32, help="max sequence length to test")

    args = parser.parse_args()

    if args.remote_path is not None:
        print(f"Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}")
    else:
        print(f"Reading {args.split} split from {args.local_path}")

    cfg = {
        "name": "text",
        "dataset": {
            "local": args.local_path,
            "remote": args.remote_path,
            "split": args.split,
            "shuffle": False,
            "max_seq_len": args.max_seq_len,
            "keep_zip": True,  # in case we need compressed files after testing
        },
        "drop_last": False,
        "num_workers": 4,
        "pin_memory": True,
    }
    cfg = om.create(cfg)
    device_batch_size = 2
    device_microbatch_size = 2

    tokenizer_cfg = {"name": args.tokenizer, "kwargs": {}}
    tokenizer_cfg["kwargs"] = {"model_max_length": args.max_seq_len}
    tokenizer_cfg = om.create(tokenizer_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)

    loader = build_text_dataloader(cfg, tokenizer, device_batch_size, device_microbatch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print("\n")
        print("#" * 20, f"Batch {batch_ix}", "#" * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch["input_ids"]):
            print("-" * 20, f" Sample {sample_ix} ", "-" * 20)
            print(tokenizer.decode(token_sample))
