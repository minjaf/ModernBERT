# Run like this:
# conda activate bert24
# python3 compute_MLM_metrics.py --model_path ~/DNALM/ModernBERT/runs/moderngena-base-pretrain-promoters_multi/ --data_dir /mnt/nfs_dna/minja/DNALM/promoter_pretrain/ --split valid --max_seq_len 1024 --output_file human.100samples.json # --log_level DEBUG

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf as om
import json
from tqdm import tqdm
import numpy as np
import random
random.seed(111)
from typing import Optional, Union, Dict, Any
from Bio.Seq import Seq
import datetime
import logging

# Add ModernBERT to path
import sys
sys.path.append(os.path.abspath("../.."))
from src.text_data import NoStreamingDataset
from src import flex_bert as flex_bert_module
from src import hf_bert as hf_bert_module
from src import mosaic_bert as mosaic_bert_module
from src.bert_layers.model import init_mlm_model_from_pretrained
from src.bert_layers.configuration_bert import FlexBertConfig
from composer.utils.checkpoint import _ensure_valid_checkpoint

# Initialize global logger
logger = logging.getLogger(__name__)

Tokenizer = Union[AutoTokenizer, Any]

# This class is used to create a dataset and sample batches from it
# Not really torch-logic class since it both generates samples and group them into batches
# Moreover, batches are variable length and we don't have control over the batch size
class NoStreamingGenomeDatasetAndDataloader(NoStreamingDataset):
	def __init__(
		self,
		local: str,
		split: Optional[str],
		max_seq_len: int,
		tokenizer: Optional[Tokenizer] = None,
		pad_sequences: bool = True,
		augment_rc: bool = False,
		sample_chunk: bool = False,
		min_seq_len: int = 10,
	) -> None:
		super().__init__(local, split, max_seq_len, tokenizer, pad_sequences)
		# todo: add seed and check that its ok for multiple workers
		self.augment_rc = augment_rc
		self.sample_chunk = sample_chunk
		self.min_seq_len = min_seq_len
		self.meaningful_seq_len = max_seq_len - 2  # -2 for CLS and SEP tokens

	def _tokenize(self, text_sample):
		assert self.tokenizer is not None, "Tokenizer required if data is not pretokenized"
		if self.tokenizer._pad_token is None:
			# Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
			raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

		# print (text_sample['file_id'], text_sample['line_id']) # GCF_003957565 0
		# raise Exception("Stop here")

		text = text_sample['text']
		logger.debug(f"text:\n {text[:250]}")
		# raise Exception("Stop here")

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
		
		# Split long sequences into chunks
		input_ids = tokenized['input_ids']

		logger.debug(f"tokenized:\n {self.tokenizer.decode(input_ids[:30])}")
		attention_mask = tokenized['attention_mask']
		offset_mapping = tokenized['offset_mapping']
		
		# If sequence is shorter than max length, just add special tokens
		if len(input_ids) <= self.meaningful_seq_len:
			return [{
				'input_ids': torch.tensor([self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]),
				'attention_mask': torch.tensor([1] + attention_mask + [1]),
				'offset_mapping': offset_mapping,
				'text': text,
				'file_id': text_sample['file_id'],
				'line_id': text_sample['line_id'],
			}]
		
		# Split into chunks of meaningful_seq_len
		chunks = []
		for i in range(0, len(input_ids) - self.meaningful_seq_len + 1, self.meaningful_seq_len):
			chunk_input_ids = input_ids[i:i + self.meaningful_seq_len]
			chunk_attention_mask = attention_mask[i:i + self.meaningful_seq_len]
			chunk_offset_mapping = offset_mapping[i:i + self.meaningful_seq_len]
			
			# Add special tokens
			chunk_input_ids = [self.tokenizer.cls_token_id] + chunk_input_ids + [self.tokenizer.sep_token_id]
			chunk_attention_mask = [1] + chunk_attention_mask + [1]
			
			chunks.append({
				'input_ids': torch.tensor(chunk_input_ids),
				'attention_mask': torch.tensor(chunk_attention_mask),
				'offset_mapping': chunk_offset_mapping,
				'text': text,
				'file_id': text_sample['file_id'],
				'line_id': text_sample['line_id'],
			})
		
		return chunks

	def __getitem__(self, index: int):
		shard_id, shard_sample_id = self.spanner[index]
		shard = self.shards[shard_id]
		sample = shard[shard_sample_id]
		
		if "input_ids" in sample:
			raise NotImplementedError("Pre-tokenized data is not supported")
		elif "text" in sample:
			# Handle raw text data
			result = self._tokenize(sample)
			return result
		else:
			raise RuntimeError("Data sample must contain a field with `input_ids` or `text`")

	def __len__(self):
		return self.len
		

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

def load_model_and_tokenizer(model_path):
	# Load config
	cfg_path = os.path.join(model_path, "cfg.yaml")
	yaml_cfg = om.load(cfg_path)

	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")

	model = build_model(yaml_cfg.model)
	
	# Load checkpoint
	# checkpoint_filepath = os.path.join(model_path, "ep11-ba68300-rank0.pt")
	checkpoint_filepath = os.path.join(model_path, "latest-rank0.pt")	
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

def get_mask_positions(seq_length, n_tokens_to_mask, offset):
	"""Generate mask positions with given offset.
	For example, if seq_length=10, n_tokens_to_mask=3:
	offset=0: [0,3,6]
	offset=1: [1,4,7]
	offset=2: [2,5,8]
	offset=3: [3,6,9]
	offset=4: [4,7,10]
	"""
	positions = []
	step_size = (seq_length + n_tokens_to_mask - 1) // n_tokens_to_mask
	for i in range(n_tokens_to_mask):
		pos = offset + i * step_size
		if pos < seq_length:
			positions.append(pos)
	return positions

def process_batch(model, tokenizer, batch, mlm_probability, output_file):
	# Create masked sequences
	# Process each sequence in the batch
	for i in range(len(batch)):
		# Process sequence
		return process_sequence_chunk(
			model, tokenizer,
			batch[i]['input_ids'], 
			batch[i]['attention_mask'], 
			batch[i]['offset_mapping'],
			batch[i]['file_id'],
			batch[i]['line_id'],
			mlm_probability, output_file)

def process_sequence_chunk(model, tokenizer, input_ids, attention_mask, offset_mapping, file_id, line_id, mlm_probability, output_file):
	# Convert inputs to tensors if they aren't already
	if not isinstance(input_ids, torch.Tensor):
		input_ids = torch.tensor(input_ids)
	if not isinstance(attention_mask, torch.Tensor):
		attention_mask = torch.tensor(attention_mask)
	
	# Get model's device
	device = next(model.parameters()).device
	
	# Get sequence length excluding special tokens
	seq_length = len(input_ids) - 2  # Exclude CLS and SEP tokens
	
	# Calculate number of tokens to mask per sequence
	n_tokens_to_mask = max(1, int(seq_length * mlm_probability))
	
	# Calculate number of offset patterns needed to cover all positions
	n_offset_patterns = (seq_length + n_tokens_to_mask - 1) // n_tokens_to_mask
	
	# Get chunk length in base pairs
	chunk_start = min(start for start, _ in offset_mapping)
	chunk_end = max(end for _, end in offset_mapping)	
	chunk_length = chunk_end - chunk_start
	
	# Initialize probability array for the chunk
	# Each position will store the probability of the ground truth token
	chunk_probs = np.full(chunk_length, np.nan)
	# chunks_nucleotides_pred = [""]*seq_length
	# chunks_nucleotides_true = [""]*seq_length
	
	# Process each offset pattern
	all_masked_positions = [] # for checking if all positions were masked
	overall_tokens_accuracy = []
	logger.debug(f"Processing chunk {file_id} {line_id} {chunk_start} {chunk_end}")
	for offset in range(n_offset_patterns):
		logger.debug(f"Processing offset {offset}")
		# Store ground truth tokens for masked positions
		ground_truths = []
		mask_positions = []
		
		# Get positions to mask for this offset (skip CLS token)
		# positions are in withCLS coordinate system (same as input_ids)
		positions = [p + 1 for p in get_mask_positions(seq_length, n_tokens_to_mask, offset)]
		
		# Store ground truth tokens
		# _ = input_ids.detach().cpu().numpy()
		# logger.debug(f"{_[:10]}")
		# for pos in positions:
		# 	if input_ids[pos].item() == 1:
		# 		logger.debug(f"input_ids[pos]==1 in positions: {pos}")
		ground_truths.extend([input_ids[pos].item() for pos in positions])
		# assert not 1 in ground_truths, "1 is in ground_truths"
		# raise Exception("Stop here")
		logger.debug(f"ground_truths:\n {tokenizer.decode(ground_truths)[:250]}")
		mask_positions.extend(positions)
		all_masked_positions.extend([p - 1 for p in positions])  # Convert back to non-special token positions
		
		# Create masked sequence
		masked_input_ids = input_ids.clone()
		masked_input_ids[positions] = tokenizer.mask_token_id
		
		inputs = {
			'input_ids': masked_input_ids.unsqueeze(0).to(device),  # Add batch dimension and move to device
			'attention_mask': attention_mask.unsqueeze(0).to(device),
		}

		# Get model predictions
		logger.debug("Getting model predictions")
		with torch.no_grad():
			outputs = model(inputs)
			logits = outputs.logits.reshape(inputs['input_ids'].shape[0],
										  inputs['input_ids'].shape[1],
										  -1)
			logger.debug("Got model predictions")
			# Get probabilities for masked positions
			# probs = torch.softmax(logits[0, positions], dim=1)
			probs = torch.argmax(logits[0, positions], dim=1)
			logger.debug("Done softmax")
			probs = probs.cpu().numpy()
			logger.debug("Moved to cpu")

		# print (probs.shape)
		# raise Exception("Stop here")
		
		# Update chunk_probs with ground truth token probabilities
		logger.debug("Filling chunk_probs")

		for i, pos in enumerate(positions):
			# Get character position for this token
			start, end = offset_mapping[pos - 1]  # -1 to account for CLS token
			# Get probability of ground truth token
			# gt_prob = probs[i, ground_truths[i]]

			gt_prob = probs[i]==ground_truths[i]
			overall_tokens_accuracy.append(gt_prob)

			# Update all positions in the token's span
			chunk_probs[start - chunk_start:end - chunk_start] = gt_prob
			# save predicted nucleotides
			assert end-start > 0, f"end-start={end-start} for pos={pos}, ground_truths[i]={ground_truths[i]}, decoded: {tokenizer.decode(ground_truths[i])}"
			# chunks_nucleotides_pred[pos - 1] = tokenizer.decode(probs[i])
			# chunks_nucleotides_true[pos - 1] = tokenizer.decode(ground_truths[i])	
			# print (tokenizer.decode(probs[i]), tokenizer.decode(ground_truths[i]))
		logger.debug("Filled chunk_probs")
	logger.debug("Saving results")
	# assert all([len(s)>0 for s in chunks_nucleotides_true]), "Some nucleotides are empty"
	# raise Exception("Stop here")
	# Save results
	output_file.write(json.dumps({
		"file_id": file_id,
		"line_id": line_id,
		"chunk_start": chunk_start,
		"chunk_end": chunk_end,
		"probabilities": chunk_probs.tolist(),
	}) + "\n")
	# "nucleotides_pred": chunks_nucleotides_pred,
	# "nucleotides_true": chunks_nucleotides_true,

	logger.debug("Saved results")
	# Verify all positions were masked
	all_masked_positions = np.array(all_masked_positions)
	assert len(np.unique(all_masked_positions)) == seq_length, "Mask positions do not cover all positions"
	return {
		"nucleotide_accuracy": np.mean(chunk_probs),
		"token_accuracy": np.mean(overall_tokens_accuracy)
	}

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
	parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
	parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
	parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
	# parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
	parser.add_argument("--mlm_probability", type=float, default=0.3, help="Probability of masking tokens")
	parser.add_argument("--output_file", type=str, required=True, help="Output file path")
	parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
	args = parser.parse_args()

	# Set logging level
	log_level = getattr(logging, args.log_level.upper())
	logger = get_logger(log_level)

	# Load model and tokenizer
	logger.info(f"Loading model and tokenizer from {args.model_path}")
	model, tokenizer = load_model_and_tokenizer(args.model_path)
	# model, tokenizer =  None, AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")

	
	logger.info("Creating dataset")	
	# Create dataset
	batched_dataset = NoStreamingGenomeDatasetAndDataloader(
		local=args.data_dir,
		split=args.split,
		max_seq_len=args.max_seq_len,
		tokenizer=tokenizer,
		pad_sequences=True
	)
	
	# Process sequences
	processed = 0
	N_to_process = 5000
	# create progress bar
	progress_bar = tqdm(total=N_to_process, desc="Processing batches")
	nucleotide_accuracy = []
	token_accuracy = []
	human = False
	randomize = True

	assert not human or not randomize, "Cannot process human data and randomize at the same time"

	with open(args.output_file, 'w') as f:
		if human:
			batch_id = 468_000 # approximately here starts human data; remove this line to process all data
		else:
			batch_id = 0

		if randomize:
			# sample N_to_process samples from the dataset without replacement
			batch_ids = random.sample(range(len(batched_dataset)), N_to_process)
		else:
			batch_ids = range(batch_id, batch_id + N_to_process)

		for batch_id in batch_ids:
			batch = batched_dataset[batch_id]
			logger.debug(f"Processing batch {batch_id}")
			if not batch[0]['file_id'].startswith('GCF_000001405') and human: # batch_id is 470000
				logger.debug(f"Skipping batch {batch[0]['file_id']}")
				continue
			logger.debug(f"Processing batch {batch_id}")
			results = process_batch(
				model, 
				tokenizer,
				batch,
				args.mlm_probability,
				f,
			)
			nucleotide_accuracy.append(results["nucleotide_accuracy"])
			token_accuracy.append(results["token_accuracy"])
			progress_bar.update(1)
			logger.debug(f"Done processing batch {processed}")			
			processed += 1
			if processed > N_to_process:
				break
	
	print (f"Nucleotide accuracy: {np.mean(nucleotide_accuracy):.3f}+-{np.std(nucleotide_accuracy):.3f}")
	print (f"Token accuracy: {np.mean(token_accuracy):.3f}+-{np.std(token_accuracy):.3f}")

if __name__ == "__main__":
	main() 