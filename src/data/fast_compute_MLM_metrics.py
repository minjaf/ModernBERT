# Run like this:

import os
import argparse
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import get_worker_info
from torch.distributed import dist
import transformers
from transformers import DataCollatorForLanguageModeling, default_data_collator
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf as om
import json
from tqdm import tqdm
import numpy as np
import random
random.seed(111)
from collections.abc import Mapping
from typing import Optional, Tuple, Union, Any, Dict, List
import logging

# Add ModernBERT to path
import sys
sys.path.append(os.path.abspath("../.."))
from src.text_data import NoStreamingGenomeDataset
from src import flex_bert as flex_bert_module
from src import hf_bert as hf_bert_module
from src import mosaic_bert as mosaic_bert_module
from src.bert_layers.model import init_mlm_model_from_pretrained
from src.bert_layers.configuration_bert import FlexBertConfig
from composer.utils.checkpoint import _ensure_valid_checkpoint

# Initialize global logger
logger = logging.getLogger(__name__)

Tokenizer = Union[AutoTokenizer, Any]

class _GenomeDatasetForMasking(NoStreamingGenomeDataset):
	def _tokenize(self, text_sample):
		assert self.tokenizer is not None, "Tokenizer required if data is not pretokenized"
		if not hasattr(self.tokenizer, "_pad_token"):
			self.tokenizer._pad_token = self.tokenizer.pad_token_id
		if self.tokenizer._pad_token is None:
			# Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
			raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

		text = text_sample['text']

		if self.sample_chunk:
			raise NotImplementedError("Chunk sampling is not implemented for sequence metrics")

		if self.augment_rc:
			raise NotImplementedError("Reverse complement is not implemented for sequence metrics")

		# Get tokenization with offset mapping
		tokenized = self.tokenizer(
			text,
			truncation=False,
			padding=False,  # We'll handle padding ourselves
			return_offsets_mapping=True,
			add_special_tokens=False
		)

		meaningful_seq_len = self.max_seq_len - 2 # -2 for CLS and SEP tokens
		assert meaningful_seq_len > 0, "Meaningful sequence length must be greater than 0"

		# Split long sequences into chunks
		input_ids = tokenized['input_ids']
		attention_mask = tokenized['attention_mask']
		offset_mapping = tokenized['offset_mapping']

		offsets_mapping_starts = [offset[0] for offset in tokenized["offset_mapping"]]
		offsets_mapping_ends = [offset[1] for offset in tokenized["offset_mapping"]]
		
		i = 0
		chunks = []
		while i < len(input_ids):
			chank_start_index = i
			chank_end_index = min(i + meaningful_seq_len, len(input_ids))
			n_padding_tokens = meaningful_seq_len - (chank_end_index - chank_start_index)
			assert n_padding_tokens >= 0, f"Number of padding tokens must >=0, but got {n_padding_tokens}"

			chunk_input_ids = input_ids[chank_start_index:chank_end_index]
			chunk_attention_mask = attention_mask[chank_start_index:chank_end_index]
			chunk_offset_mapping_starts = offsets_mapping_starts[chank_start_index:chank_end_index]
			chunk_offset_mapping_ends = offsets_mapping_ends[chank_start_index:chank_end_index]	
			
			# Add special tokens
			chunk_input_ids = [self.tokenizer.cls_token_id] + chunk_input_ids + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id]*n_padding_tokens
			chunk_attention_mask = [1] + chunk_attention_mask + [1] + [0]*n_padding_tokens
			chunk_offset_mapping_starts = [0] + chunk_offset_mapping_starts + [0] + [0]*n_padding_tokens # set (0,0) for CLS, SEP, PAD
			chunk_offset_mapping_ends = [0] + chunk_offset_mapping_ends + [0] + [0]*n_padding_tokens # set (0,0) for CLS, SEP, PAD
			requered_outputs = {
				'input_ids': torch.tensor(chunk_input_ids),
				'attention_mask': torch.tensor(chunk_attention_mask),
				'offset_mapping_starts': chunk_offset_mapping_starts,
				'offset_mapping_ends': chunk_offset_mapping_ends,
			}
			optional_outputs = {
				'text': text,
				'file_id': text_sample['file_id'],
				'line_id': text_sample['line_id'],
				'line_chunk_offset': text_sample['chunk_offset']
			}
			chunks.append(requered_outputs | optional_outputs) # merge required and optional outputs
			i += meaningful_seq_len

		return chunks


	def __getitem__(self, index: int):
		shard_id, shard_sample_id = self.spanner[index]
		shard = self.shards[shard_id]
		sample = shard[shard_sample_id]
		if "text" in sample:
			# propagate shard id and shard sample id to save MLM probs after we get it
			results = self._tokenize(sample)
			for result in results:
				result["shard_id"] = [shard_id]
				result["shard_sample_id"] = [shard_sample_id]
				result["mlm_efficiency_path"] = self.mlm_efficiency_path
				yield result
		else:
			raise RuntimeError("Data sample must contain a field with `text`")

	def __len__(self):
		return self.len

class GenomeDatasetForMasking(IterableDataset):
	def __init__(self, dataset: _GenomeDatasetForMasking):
		self.dataset = dataset
		super().__init__()
	
	def __iter__(self):
		wi = get_worker_info()
		worker_id = 0 if wi is None else wi.id
		num_workers = 1 if wi is None else wi.num_workers

		rank = dist.get_rank() if dist.is_initialized() else 0
		world = dist.get_world_size() if dist.is_initialized() else 1

		global_workers = world * num_workers
		my_id = rank * num_workers + worker_id

		print ("------>Worker id: ", worker_id)
		print ("------>Number of workers: ", num_workers)
		print ("------>Rank: ", rank)
		print ("------>World size: ", world)
		print ("------>Global workers: ", global_workers)
		print ("------>My id: ", my_id)

		for i in range(len(self.dataset)):   # deterministic order
			if i % global_workers == my_id:
				samples = self.dataset.__getitem__(i)
				for sample in samples:
					yield sample
		raise StopIteration


class DataCollatorForMaskingAllPositions(transformers.DataCollatorForLanguageModeling):
	def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
		raise NotImplementedError("TF is not supported for MLM with MLM probs")
						
	def torch_mask_tokens(self, inputs: Any, 
						  special_tokens_mask: Optional[Any] = None,
						  ) -> Tuple[Any, Any, Any]:
		"""
		Prepare masked tokens inputs/labels for masked language modeling.
		Based on the original implementation of transformers' `DataCollatorForLanguageModeling.torch_mask_tokens`

		given the batched-input and masking fraction, create super-batch where we mask all positions in the input, 
		mask_fraction of the tokens in each split are masked.
		return the super-batch and the labels array.
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
		assert (Num_non_special_tokens > 0).all(), f"Num_non_special_tokens must be greater than 0, but got {Num_non_special_tokens}"

		masking_fraction = self.mlm_probability
		assert masking_fraction < 1.0, f"Masking fraction must be less than or equal to 1.0, but got {masking_fraction}"
		assert masking_fraction > 0.0, f"Masking fraction must be greater than 0.0, but got {masking_fraction}"

		num_splits = int(1/masking_fraction)
		assert num_splits > 0, "Number of splits must be greater than 0"
		masked_token_ids = []
		n_repeats = []
		for i in range(inputs.shape[0]):
			nonspecial_tokens_in_sequence = np.nonzero(~special_tokens_mask[i].numpy())[0]
			_num_splits = num_splits
			if _num_splits > len(nonspecial_tokens_in_sequence):
				_num_splits = len(nonspecial_tokens_in_sequence)
			splits = np.array_split(nonspecial_tokens_in_sequence, _num_splits)
			assert len(splits) >= 1, f"Number of splits must be greater than or equal to 1, but got {len(splits)}"
			# evenly distribute tokens across splits
			splits = [nonspecial_tokens_in_sequence[np.arange(len(split))*_num_splits + i] for i, split in enumerate(splits)]
			n_repeats.append(len(splits))
			masked_token_ids.extend(splits)

		n_repeats = torch.tensor(n_repeats, dtype=torch.long)
		masked_indices = torch.zeros((n_repeats.sum().item(), inputs.shape[1]), dtype=bool)
		for sample_id in range(len(masked_token_ids)):
			sample_masked_token_ids = masked_token_ids[sample_id]
			masked_indices[sample_id, sample_masked_token_ids] = True
	
		labels = torch.repeat_interleave(labels, n_repeats, dim=0)
		labels[~masked_indices] = -100  # We only compute loss on masked tokens

		inputs = torch.repeat_interleave(inputs, n_repeats, dim=0)

		# we replace masked input tokens with tokenizer.mask_token ([MASK])
		inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

		return inputs, labels, n_repeats
	
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

			keep_for_model = ["input_ids", "attention_mask", "offset_mapping_starts", "offset_mapping_ends", 
				"labels", "shard_id", "shard_sample_id"]
			batch = [{k: v for k, v in example.items() if k in keep_for_model} for example in examples]
			
			# now handle all tensors
			batch = default_data_collator(batch, self.return_tensors)			
		else:
			raise NotImplementedError("Only dicts are supported for MLM with MLM probs")
			# batch = {
			#     "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
			# }

		# If special token mask has been preprocessed, pop it from the dict.
		special_tokens_mask = batch.pop("special_tokens_mask", None)
		batch["input_ids"], batch["labels"], n_repeats = self.torch_mask_tokens(
					batch["input_ids"], special_tokens_mask=special_tokens_mask
		)

		for k, v in batch.items():
			if not k in ["input_ids", "labels"]:
				batch[k] = torch.repeat_interleave(v, n_repeats, dim=0) # repeat the tensor along the first dimension

		# and keep the mlm_efficiency_path as single value in the batch
		if mlm_efficiency_paths is not None: 
			batch["mlm_efficiency_path"] = mlm_efficiency_paths

		return batch


def get_logger(logLevel=logging.INFO):
	global logger
	logger.setLevel(logLevel)
	handler = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger

def build_model(cfg: DictConfig):
	if cfg.name == "hf_bert":
		return hf_bert_module.create_hf_bert_mlm(
			pretrained_model_name=cfg.pretrained_model_name,
			use_pretrained=cfg.get("use_pretrained", None),
			model_config=cfg.get("model_config", None),
			tokenizer_name=cfg.get("tokenizer_name", None),
			gradient_checkpointing=cfg.get("gradient_checkpointing", None),
		)
	elif cfg.name == "mosaic_bert":
		return mosaic_bert_module.create_mosaic_bert_mlm(
			pretrained_model_name=cfg.pretrained_model_name,
			pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
			model_config=cfg.get("model_config", None),
			tokenizer_name=cfg.get("tokenizer_name", None),
			gradient_checkpointing=cfg.get("gradient_checkpointing", None),
		)
	elif cfg.name == "flex_bert":
		return flex_bert_module.create_flex_bert_mlm(
			pretrained_model_name=cfg.pretrained_model_name,
			pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
			model_config=cfg.get("model_config", None),
			tokenizer_name=cfg.get("tokenizer_name", None),
			gradient_checkpointing=cfg.get("gradient_checkpointing", None),
			recompute_metric_loss=cfg.get("recompute_metric_loss", False),
			disable_train_metrics=cfg.get("disable_train_metrics", False),
		)
	else:
		raise ValueError(f"Not sure how to build model with name={cfg.name}")

def load_model_and_tokenizer(checkpoint_filepath):
	# Load config
	model_path = os.path.dirname(checkpoint_filepath)
	cfg_path = os.path.join(model_path, "cfg.yaml")
	yaml_cfg = om.load(cfg_path)

	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")

	model = build_model(yaml_cfg.model)
	
	# Load checkpoint
	logger.info(f"Loading checkpoint from {checkpoint_filepath}")
	state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu")
	state_dict = state.get("state", {})
	model_state = state_dict.get("model", {})
	assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"
	model.load_state_dict(model_state)
	
	if torch.cuda.is_available():
		model = model.cuda()
	model.eval()
	
	return model, tokenizer

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
	parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
	parser.add_argument("--mlm_efficiency_path", type=str, required=True, help="Where to save the mlm efficiency")
	parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
	parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for processing")
	parser.add_argument("--mlm_probability", type=float, default=0.18, help="Probability of masking tokens")
	parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
	args = parser.parse_args()

	# Set logging level
	log_level = getattr(logging, args.log_level.upper())
	logger = get_logger(log_level)

	# Load model and tokenizer
	logger.info(f"Loading model and tokenizer from {args.model_path}")
	model, tokenizer = load_model_and_tokenizer(args.model_path)
	
	logger.info("Creating dataset")	
	# Create dataset
	_dataset = _GenomeDatasetForMasking(
		local=args.data_dir,
		split=args.split,
		max_seq_len=args.max_seq_len,
		tokenizer=tokenizer,
		mlm_efficiency_path=args.mlm_efficiency_path
	)

	dataset = GenomeDatasetForMasking(_dataset)

	collate_fn = DataCollatorForMaskingAllPositions(
		tokenizer=tokenizer, mlm_probability=args.mlm_probability
	)
	dataloader = DataLoader(
		dataset,
		collate_fn=collate_fn,
		batch_size=args.batch_size,
		drop_last=False,
		num_workers=args.num_workers,
		pin_memory=False,
		prefetch_factor=2,
		persistent_workers=False,
		timeout=0,
	)

	for batch in dataloader:
		with torch.no_grad():
			outputs = model(batch["input_ids"])
		raise ValueError("Stop here")

if __name__ == "__main__":
	main() 