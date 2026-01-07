# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a FlexBERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import torch
from torch import Tensor

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from omegaconf import DictConfig, OmegaConf

import src.bert_layers as bert_layers_module
import src.bert_layers.configuration_bert as configuration_bert_module
import transformers
from composer.metrics.nlp import BinaryF1Score, LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.devices import DeviceCPU

from torchmetrics import MeanSquaredError, Metric
from torchmetrics.classification.accuracy import MulticlassAccuracy, MultilabelAccuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef
from torchmetrics.utilities import dim_zero_cat

from filelock import FileLock
import h5py
import numpy as np
import datetime

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None

all = ["create_flex_bert_mlm", "create_flex_bert_classification"]


# we want the efficent versions to have the same name as the TorchMetrics' name
def rename_class(new_name):
    def class_renamer(cls):
        cls.__name__ = new_name
        return cls

    return class_renamer


@rename_class("LanguageCrossEntropy")
class FALanguageCrossEntropy(LanguageCrossEntropy):
    """Torchmetric that computes cross entropy on language modeling outputs using flash_attn's Cross Entropy.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_items (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False, ignore_index: int = -100):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if CrossEntropyLoss is None:
            raise ImportError("flash_attn is not installed. Please install flash_attn to use FALanguageCrossEntropy.")

        self.ignore_index = ignore_index
        self.loss_fn = CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")


@rename_class("LanguageCrossEntropy")
class EfficientCrossEntropy(Metric):
    """Torchmetric that grabs the precomputed ce_loss value from the model outputs"""

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            loss (~torch.Tensor): A Tensor of loss values to compare against.
        """
        self.sum_loss += loss
        self.total_items += 1

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items  # type: ignore (third-party)


@rename_class("ZLoss")
class EfficientZLoss(Metric):
    """Torchmetric that grabs the precomputed z_loss value from the model outputs"""

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            loss (~torch.Tensor): A Tensor of loss values to compare against.
        """
        self.sum_loss += loss
        self.total_items += 1

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items  # type: ignore (third-party)

def _write_sample_to_file(sample_data, f, metric_to_save, write2file_threshold):
    """Write a single sample to its corresponding file using thread-safe operations."""
    shard_sample_id, num_repeat, start_idx, end_idx, true_probs, offset_starts, offset_ends, shard_id = sample_data
    if num_repeat == 0:
        return
        
    max_probability = true_probs[start_idx:end_idx].max()
    
    # skip if max probability of all tokens in the sample is less than write2file_threshold
    if max_probability < write2file_threshold or write2file_threshold == 0.0: # in case true probs are bool
        return

    assert str(shard_sample_id) in f, f"shard_sample_id {shard_sample_id} not found in {f.filename}. Available keys: {list(f.keys())}"
    data = f[str(shard_sample_id)][:]  # read all data
    for i in range(num_repeat):
        token_idx = start_idx + i
        # update only if true prob is > write2file_threshold to save time
        if true_probs[token_idx] > write2file_threshold:
            st = offset_starts[token_idx]
            en = offset_ends[token_idx]
            if metric_to_save == "is_correct":
                data[st:en] = true_probs[token_idx]
            elif metric_to_save == "probability":
                data[st:en] = 1. - true_probs[token_idx]  # update in memory
            else:
                raise ValueError(f"Invalid metric_to_save: {metric_to_save}")
    f[str(shard_sample_id)][:] = data  # single write operation

def _write_samples_from_same_h5_to_file_single_thread(samples, metric_to_save, write2file_threshold, filepath):
    shard_id = [sample[-1] for sample in samples]
    assert len(set(shard_id)) == 1, "shard_sample_ids must be the same for all samples"
    shard_id = shard_id[0]
    save_path = os.path.join(filepath, f"shard_{shard_id}.hdf5")
    timestamp = datetime.datetime.now()
    with FileLock(save_path + ".lock"):
        with h5py.File(save_path, "a") as f:
            timedelta_opening_file = datetime.datetime.now() - timestamp
            timestamp = datetime.datetime.now()
            for sample in samples:
                _write_sample_to_file(sample, f, metric_to_save, write2file_threshold)
            timedelta_writing_to_file = datetime.datetime.now() - timestamp
    return timedelta_opening_file, timedelta_writing_to_file


class bpLoss(Metric):
    """Torchmetric that grabs the per-sample per-token loss"""

    # Make torchmetrics call update only once
    full_state_update = False


    def __init__(self, dist_sync_on_step: bool = False, filepath: str | None = None,
            metric_to_save : str = "probability", # or "is_correct",
            metric_to_return : str = "mean_non_pad_MLM_probs", # or "mean_is_correct",
            write2file_threshold: float = 0.0, n_threads: int = 1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # self.add_state("true_probs", default=[], dist_reduce_fx="cat")
        # self.add_state("offset_starts", default=[], dist_reduce_fx="cat")
        # self.add_state("offset_ends", default=[], dist_reduce_fx="cat")        
        # self.add_state("shard_sample_ids", default=[], dist_reduce_fx="cat")
        # self.add_state("shard_ids", default=[], dist_reduce_fx="cat")
        # self.add_state("num_repeats", default=[], dist_reduce_fx="cat")
        self.add_state("N_toks_processed", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("N_correct_tokens", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("return_metric", default=[], dist_reduce_fx="cat")
        self.filepath = filepath
        self.metric_to_save = metric_to_save
        self.metric_to_return = metric_to_return
        self.write2file_threshold = write2file_threshold
        self.n_threads = n_threads

    def update(self, logits: Tensor, labels_reduced: Tensor, labels_full: Tensor, batch: dict) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            entropy (~torch.Tensor): A Tensor of entropy values to compare against.
            offsets (~torch.Tensor): A Tensor of offsets to compare against.
        """

        assert (labels_reduced.size()[0] == (labels_full != -100).sum()).item(), "labels_reduced and labels_full have different number of non-masked tokens"
        assert len(labels_full.size()) == len(batch["shard_id"].size()) == len(batch["shard_sample_id"].size()) == len(batch["offsets_mapping_starts"].size()) == len(batch["offsets_mapping_ends"].size()) == 2, "labels_full, shard_id, shard_sample_id, offsets_mapping_starts and offsets_mapping_ends must be a 2D tensor"
        assert logits.size()[0] == labels_reduced.size()[0], "logits and labels_reduced have different batch sizes"
        mask = labels_full != -100
        # print ("labels_reduced: ", labels_reduced.size()) # size: N_masked
        # print ("logits: ", logits.size()) # size: N_masked x V
        # print ("labels_full: ", labels_full.size()) # size: B x S
        # print ("input_ids: ", batch["input_ids"].size()) # size: B x S

        if self.metric_to_save == "is_correct":
            highest_probability_index = logits.argmax(dim=1)
            assert highest_probability_index.size() == labels_reduced.size(), "highest_probability_index and labels_reduced have different sizes"
            is_correct = highest_probability_index == labels_reduced
            true_probs = is_correct # just for naming convention, actually these are not probabilities
        elif self.metric_to_save == "probability":            
            logits = torch.softmax(logits, dim=1)
            # logits is N_masked x V
            # labels is N_masked
            # Get the probability for each true token
            true_probs = torch.gather(logits, dim=1, index=labels_reduced.unsqueeze(1)).squeeze(1)
            # print ("------>Mean of true_probs: ", true_probs.mean().item())
            # print ("------>Max of true_probs: ", true_probs.max().item())
            # print ("------>N of true_probs > self.write2file_threshold: ", (true_probs > self.write2file_threshold).sum().item(), " out of ", true_probs.size()[0])
        else:
            raise ValueError(f"Invalid metric_to_save: {self.metric_to_save}")

        offset_starts = batch["offsets_mapping_starts"][mask]
        offset_ends = batch["offsets_mapping_ends"][mask]
        assert torch.all(offset_ends - offset_starts > 0), "offset_ends must be greater than offset_starts"
        assert torch.allclose(batch["shard_sample_id"].min(dim=1)[0], batch["shard_sample_id"].max(dim=1)[0]), "shard_sample_id must be unique value per sample in batch"
        num_repeats = mask.sum(dim=1)
        # if num_repeats.min() == 0:
        #     print("Example of batch where num_repeats == 0:")
        #     print(f"num_repeats: {num_repeats}")
        #     print(f"labels_full shape: {labels_full.shape}")
        #     print(f"mask shape: {mask.shape}")
        #     print(f"mask sum per sample: {mask.sum(dim=1)}")
        #     print(f"labels_full sample: {labels_full[0] if labels_full.size(0) > 0 else 'empty'}")
        #     print(f"mask sample: {mask[0] if mask.size(0) > 0 else 'empty'}")
        #     print("This indicates some samples have no masked tokens to predict")
        shard_sample_ids = batch["shard_sample_id"].flatten()
        shard_ids = batch["shard_id"].flatten()
        save_paths = batch["mlm_efficiency_path"]
        
        # samples_ids = torch.repeat_interleave(batch["shard_sample_id"].flatten(), mask.sum(dim=1))
        
        assert len(true_probs) == len(offset_starts) == len(offset_ends), "offset_starts, offset_ends, samples_ids and shard_ids must have the same length"
        assert len(num_repeats) == len(shard_sample_ids) == len(shard_ids) == len(save_paths), "num_repeats, shard_sample_ids, shard_ids and save_paths must have the same length"
        assert sum(num_repeats) == true_probs.size()[0], "num_repeats does not match the number of true probabilities"

        # self.true_probs.append(true_probs)
        # self.offset_starts.append(offset_starts)
        # self.offset_ends.append(offset_ends)
        # self.shard_sample_ids.append(shard_sample_ids)
        # self.shard_ids.append(shard_ids)
        assert len(set(save_paths)) == 1, "save_paths must be the same for all samples in the batch"
        save_path = save_paths[0]
        if self.filepath is None:
            self.filepath = save_path
        assert self.filepath == save_path, f"save_paths must be the same for all metric calls, {self.filepath} != {save_path}"
        # self.num_repeats.append(num_repeats)
        self.N_toks_processed += true_probs.size()[0]
        if self.metric_to_return == "mean_is_correct":
            self.N_correct_tokens += true_probs.sum().item()
        elif self.metric_to_return == "mean_non_pad_MLM_probs":
            self.return_metric.append(batch["mean_non_pad_MLM_probs"])
        else:
            raise ValueError(f"Invalid metric_to_return: {self.metric_to_return}")
        self._to_file(true_probs, offset_starts, offset_ends, shard_ids, shard_sample_ids, num_repeats)
        del true_probs, offset_starts, offset_ends, shard_ids, shard_sample_ids, num_repeats

    def _to_file(self, true_probs, offset_starts, offset_ends, shard_ids, shard_sample_ids, num_repeats):        
        # convert to numpy
        true_probs = true_probs.cpu().numpy()
        offset_starts = offset_starts.cpu().numpy()
        offset_ends = offset_ends.cpu().numpy()
        shard_sample_ids = shard_sample_ids.cpu().numpy()
        shard_ids = shard_ids.cpu().numpy()
        num_repeats = num_repeats.cpu().numpy()

        # Prepare sample data for processing
        sample_data_dict = {}
        idx_of_token_in_concatenated_batch = 0
        
        for idx_of_sample_in_batch in range(len(shard_sample_ids)):
            shard_sample_id = shard_sample_ids[idx_of_sample_in_batch]
            num_repeat = num_repeats[idx_of_sample_in_batch]
            shard_id = shard_ids[idx_of_sample_in_batch]
            
            if num_repeat == 0:
                continue
                
            start_idx = idx_of_token_in_concatenated_batch
            end_idx = idx_of_token_in_concatenated_batch + num_repeat
            
            sample_data = (
                shard_sample_id, num_repeat, start_idx, end_idx,
                true_probs, offset_starts, offset_ends, shard_id
            )
            if shard_id not in sample_data_dict:
                sample_data_dict[shard_id] = []
            sample_data_dict[shard_id].append(sample_data)
            
            idx_of_token_in_concatenated_batch += num_repeat
            
        assert idx_of_token_in_concatenated_batch == true_probs.shape[0], "idx_of_token_in_concatenated_batch does not match the number of true probabilities"

        total_timedelta_opening_file = datetime.timedelta(0)
        total_timedelta_writing_to_file = datetime.timedelta(0)
        wall_time = datetime.datetime.now()
        
        # print(f"Total number of samples: {len(shard_ids)}, number of shards: {len(sample_data_dict)}")
        
        if self.n_threads <= 1:
            # Use optimized single-threaded approach for I/O bottleneck scenarios
            for shard_id, samples in sample_data_dict.items():
                result = _write_samples_from_same_h5_to_file_single_thread(samples, 
                                                                            self.metric_to_save,
                                                                            self.write2file_threshold, 
                                                                            self.filepath,
                                                                        )
                if result is not None:
                    timedelta_opening_file, timedelta_writing_to_file = result
                    total_timedelta_opening_file += timedelta_opening_file
                    total_timedelta_writing_to_file += timedelta_writing_to_file
        else:
            # Use multithreading for CPU-bound scenarios
            max_workers = min(self.n_threads, len(sample_data_dict))
            print(f"Using {max_workers} threads for multithreaded approach")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(_write_samples_from_same_h5_to_file_single_thread, 
                    sample_data_dict[shard_id], self.metric_to_save, self.write2file_threshold, self.filepath): shard_id
                    for shard_id in sample_data_dict.keys()
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_sample):
                    result = future.result()
                    if result is not None:
                        timedelta_opening_file, timedelta_writing_to_file = result
                        total_timedelta_opening_file += timedelta_opening_file
                        total_timedelta_writing_to_file += timedelta_writing_to_file
        
        del true_probs, offset_starts, offset_ends, shard_ids, shard_sample_ids, num_repeats
        wall_time = datetime.datetime.now() - wall_time
        # print("---->timedelta_opening_file = ", total_timedelta_opening_file.total_seconds() * 1000, "ms")
        # print("---->timedelta_writing_to_file = ", total_timedelta_writing_to_file.total_seconds() * 1000, "ms")
        # print("---->wall_time = ", wall_time.total_seconds() * 1000, "ms")

    def compute(self):
        """Aggregate the state over all processes to compute the metric.
        """
        # convert to tensors and concatenate
        # true_probs = dim_zero_cat(self.true_probs)
        # offset_starts = dim_zero_cat(self.offset_starts)
        # offset_ends = dim_zero_cat(self.offset_ends)
        # shard_sample_ids = dim_zero_cat(self.shard_sample_ids)
        # shard_ids = dim_zero_cat(self.shard_ids)
        # num_repeats = dim_zero_cat(self.num_repeats)

        # self._to_file(true_probs, offset_starts, offset_ends, shard_ids, shard_sample_ids, num_repeats)
        # return self.N_toks_processed.item()
        if self.metric_to_return == "mean_is_correct":
            return self.N_correct_tokens / self.N_toks_processed
        elif self.metric_to_return == "mean_non_pad_MLM_probs":
            return_value = dim_zero_cat(self.return_metric)
            return self.return_metric.mean().item()
        else:
            raise ValueError(f"Invalid metric_to_return: {self.metric_to_return}")

        # TODO. Should we reset the state? Let's belive trainer will do it for us

class eval_bpLoss(bpLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.metric_to_return == "mean_non_pad_MLM_probs":
            raise ValueError("""eval_bpLoss does not support mean_non_pad_MLM_probs.
            During evaluation, we only compute the mean of the correct tokens.
            Because we don't reset metric state, and mean_non_pad_MLM_probs requires accumulation
            this will cause out of memory error""")

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.reset()

    def _to_file(self, true_probs, offset_starts, offset_ends, shard_ids, shard_sample_ids, num_repeats): 
        super()._to_file(true_probs, offset_starts, offset_ends, shard_ids, shard_sample_ids, num_repeats)
        save_path = os.path.join(self.filepath, f"processed_samples.hdf5")
        # print (f"Saving processed samples to {save_path}")
        with FileLock(save_path + ".lock"):
            with h5py.File(save_path, "a") as f:
                # Get unique pairs of (shard_id, sample_id)
                unique_pairs = set(zip(shard_ids.cpu().numpy(), shard_sample_ids.cpu().numpy()))
                
                # Write each unique pair to the h5 file
                for shard_id, sample_id in unique_pairs:
                    shard_id_str = str(shard_id)
                       
                    # Set the sample_id to True
                    f[shard_id_str][sample_id] = True

class EfficientHuggingFaceModel(HuggingFaceModel):
    def eval_forward(self, batch, outputs: Optional[Any] = None):
        outputs = self.forward(batch) if outputs is None else outputs
        self.labels = batch.pop("labels")
        return outputs

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> Dict:
        if metric.device.type == "cpu":
            self.labels = DeviceCPU().batch_to_device(self.labels)

        if getattr(metric, "needs_batch", False):
            raise ValueError(f"Unsupported metric {metric=}")

        if getattr(outputs, "ce_loss", False) and isinstance(metric, EfficientCrossEntropy):
            metric_result = metric.update(outputs["ce_loss"])
        elif getattr(outputs, "z_loss", False) and isinstance(metric, EfficientZLoss):
            metric_result = metric.update(outputs["z_loss"])
        elif isinstance(metric, EfficientCrossEntropy):
            metric_result = metric.update(outputs["loss"])
        elif isinstance(metric, bpLoss):
            metric_result = metric.update(outputs["logits"], outputs.get("labels", None), self.labels, batch)
        else:
            metric_result = metric.update(outputs["logits"], outputs.get("labels", self.labels))

        if metric_result is not None:
            # Add the metric name once for each datapoint in the batch
            metric_result["metric_name"] = [metric.__class__.__name__ for _ in range(0, batch["input_ids"].shape[0])]
        else:
            metric_result = {}
        return metric_result


def create_flex_bert_mlm(
    pretrained_model_name: str = "bert-base-uncased",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    recompute_metric_loss: Optional[bool] = False,
    disable_train_metrics: Optional[bool] = False,
):
    """FlexBERT masked language model based on |:hugging_face:| Transformers.

    For more information, see
    `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a FlexBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided, the state dictionary
            stored at `pretrained_checkpoint` will be loaded into the model
            after initialization. Default: ``None``.
        disable_train_metrics (bool, optional): Only calculate metrics for
            validation set when True.
            Default: ``False``.

    .. code-block::

        {
        "_name_or_path": "bert-base-uncased",
        "alibi_starting_size": 512,
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": null,
        "gradient_checkpointing": false,
        "hidden_act": "silu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.16.0",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": 30522
        }

    To create a FlexBERT model for Masked Language Model pretraining:

     .. testcode::

         from src.mosaic import create_flex_bert_mlm
         model = create_flex_bert_mlm()
    """
    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    if isinstance(model_config, DictConfig):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    config = configuration_bert_module.FlexBertConfig.from_pretrained(pretrained_model_name, **model_config)

    if "prenorm" in config.bert_layer:
        assert config.final_norm, "Final norm must be used with prenorm attention"
    else:
        assert "postnorm" in config.bert_layer, "config.bert_layer str must contain either prenorm or postnorm"
        assert not config.final_norm, "Final norm should not be used with postnorm attention"

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = bert_layers_module.FlexBertForMaskedLM.from_composer(
            pretrained_checkpoint=pretrained_checkpoint, config=config
        )
    else:
        model = bert_layers_module.FlexBertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    metrics = [MaskedAccuracy(ignore_index=-100)]

    if recompute_metric_loss or model_config["loss_function"] not in ["fa_cross_entropy", "cross_entropy"]:
        if CrossEntropyLoss is not None:
            metrics = [FALanguageCrossEntropy(ignore_index=-100)] + metrics
        else:
            metrics = [LanguageCrossEntropy(ignore_index=-100)] + metrics
    else:
        metrics = [EfficientCrossEntropy()] + metrics
    if model_config.get("loss_kwargs", {}).get("return_z_loss", False):
        metrics += [EfficientZLoss()]
    if model_config.get("save_mlm_probs", None) is not None:
        metrics += [bpLoss(**model_config.get("save_mlm_probs", {}))]
    if model_config.get("eval_mlm_probs", None) is not None:
        metrics += [eval_bpLoss(**model_config.get("eval_mlm_probs", {}))]

    eval_metrics = copy.deepcopy(metrics)
    eval_metrics = [m for m in eval_metrics if m.__class__.__name__ != "bpLoss"] # we do not need to compute bpLoss on eval set
    if disable_train_metrics:
        metrics = [m for m in metrics if m.__class__.__name__ == "bpLoss"] # we always keep bpLoss on the training set
        if len(metrics) == 0:
            metrics = None

    hf_model = EfficientHuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        eval_metrics=eval_metrics,
        allow_embedding_resizing=model.config.allow_embedding_resizing,
    )

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model


def create_flex_bert_classification(
    num_labels: int,
    pretrained_model_name: str = "bert-base-uncased",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    custom_eval_metrics: Optional[list] = [],
    multiple_choice: Optional[bool] = False,
):
    """FlexBERT classification model based on |:hugging_face:| Transformers.

    For more information, see `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a FlexBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        num_labels (int): The number of classes in the classification task.
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided,
            the state dictionary stored at `pretrained_checkpoint` will be
            loaded into the model after initialization. Default: ``None``.
        custom_eval_metrics (list, optional): Classes of custom metrics to
            evaluate the model. Default: ``[]``.
        multiple_choice (bool, optional): Whether the model is used for
            multiple choice tasks. Default: ``False``.

    .. code-block::
        {
            "_name_or_path": "bert-base-uncased",
            "alibi_starting_size": 512,
            "architectures": ["BertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout": null,
            "gradient_checkpointing": false,
            "hidden_act": "silu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1",
                "2": "LABEL_2"
            },
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2
            },
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.16.0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }

    To create a FlexBERT model for classification:

     .. testcode::
        from flex_bert import create_flex_bert_classification
        model = create_flex_bert_classification(num_labels=3) # if the task has three classes.

    Note:
        This function can be used to construct a BERT model for regression by
        setting ``num_labels == 1``. This will have two noteworthy effects.
        First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics
        will be :class:`~torchmetrics.MeanSquaredError` and
        :class:`~torchmetrics.SpearmanCorrCoef`. For the classifcation case
        (when ``num_labels > 1``), the training loss is
        :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.MulticlassAccuracy` and
        :class:`~torchmetrics.MatthewsCorrCoef`, as well as
        :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    if not model_config:
        model_config = {}

    # By default, turn off attention dropout in FlexBERT
    # Flash Attention 2 supports dropout in the attention module
    # while our previous Triton Flash Attention layer only works with
    # attention_probs_dropout_prob = 0.
    if "attention_probs_dropout_prob" not in model_config:
        model_config["attention_probs_dropout_prob"] = 0.0

    model_config["num_labels"] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    model_cls = bert_layers_module.FlexBertForSequenceClassification

    if multiple_choice:
        model_cls = bert_layers_module.FlexBertForMultipleChoice

    if isinstance(model_config, DictConfig):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    config = configuration_bert_module.FlexBertConfig.from_pretrained(pretrained_model_name, **model_config)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = model_cls.from_composer(pretrained_checkpoint=pretrained_checkpoint, config=config)
    else:
        model = model_cls(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [
            MulticlassAccuracy(num_classes=num_labels, average="micro"),
            MatthewsCorrCoef(task="multiclass", num_classes=model.config.num_labels),
        ]
        if num_labels == 2:
            metrics.append(BinaryF1Score())

    if model_config.get("problem_type", "") == "multi_label_classification":
        metrics = [
            MultilabelAccuracy(num_labels=num_labels, average="micro"),
        ]

    hf_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        eval_metrics=[
            *metrics,
            *[metric_cls() for metric_cls in custom_eval_metrics],
        ],
        allow_embedding_resizing=model.config.allow_embedding_resizing,
    )

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model
