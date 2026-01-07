#!/usr/bin/env python3

"""
Convert h5 files with MLM metrics to bedgraph format.

This script:
1. Extracts shard_id from h5 filename
2. Extracts shard_sample_ids from h5 file keys
3. Sets up shards similar to NoStreamingGenomeDataset
4. For each sample, gets text_sample and extracts file_id and line_id
5. Maps h5 data to genomic coordinates and saves as .bedgraph file
"""

import argparse
import os
import json
import h5py
import numpy as np
import re
import sys
from tqdm import tqdm

# Add ModernBERT to path
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("."))
from src.text_data import NoStreamingGenomeDataset
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner
from src.data.metrics2bedgraph import generate_chromosome_renamer


def extract_shard_id_from_filename(h5_filepath):
	"""Extract shard_id from h5 filename like 'shard_0.hdf5'."""
	basename = os.path.basename(h5_filepath)
	match = re.match(r'shard_(\d+)\.hdf5', basename)
	if match:
		return int(match.group(1))
	else:
		raise ValueError(f"Could not extract shard_id from filename: {basename}")


def setup_shards(data_dir, split):
	"""Set up shards similar to NoStreamingGenomeDataset."""
	if split is not None:
		split_path = os.path.join(data_dir, split)
	else:
		split_path = data_dir
	index_file_path = os.path.join(split_path, "index.json")
	obj = json.load(open(index_file_path))
	shards = []
	for info in obj["shards"]:
		shard = reader_from_json(data_dir, split, info)
		raw_filename = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
		assert os.path.isfile(raw_filename), f"Raw file {raw_filename} does not exist"
		shard.validate(True)
		shards.append(shard)
	return shards


def parse_args():
	parser = argparse.ArgumentParser(description='Convert h5 file with MLM metrics to bedgraph format')
	parser.add_argument('--h5_file', type=str, default="runs/test/mlm_efficiency/human/train/shard_184.hdf5",
					  help='Path to h5 file (e.g., shard_0.hdf5)')
	parser.add_argument('--data_dir', type=str, default="/mnt/nfs_dna/shadskiy/promoters/pretrena/mds_v2/",
					  help='Path to dataset directory (MDS format)')
	parser.add_argument('--split', type=str, default="train",
					  help='Dataset split (e.g., "train"). If None, uses data_dir directly')
	parser.add_argument('--N', type=int, default="0",
					  help='Subsample data to this number of samples. If 0, use all samples')
	parser.add_argument('--promoters_dir', type=str, default="/mnt/nfs_dna/shadskiy/promoters/pretrena/train/GCF_000001405.40/",
					  help='Directory containing original promoter JSON files')
	# parser.add_argument('--promoters_dir', type=str, default="/mnt/nfs_dna/minja/DNALM/promoter_pretrain/small_test_100_human_promoters/jsons/GCF_000001405.40/",
	# 				  help='Directory containing original promoter JSON files')
	parser.add_argument('--output_prefix', type=str, default="runs/test/mlm_efficiency/bedgraph/train/shard_0",
					  help='Prefix for output bedgraph file')
	return parser.parse_args()


def process_h5_file(args):
	"""Process h5 file and convert to bedgraph format."""
	# 1. Extract shard_id from filename
	shard_id = extract_shard_id_from_filename(args.h5_file)
	print(f"Extracted shard_id: {shard_id}")
	
	# 2. Extract shard_sample_ids from h5 file keys
	with h5py.File(args.h5_file, 'r') as f:
		shard_sample_ids = [int(key) for key in f.keys()]
	print(f"Found {len(shard_sample_ids)} samples in h5 file")

	if args.N > 0:
		shard_sample_ids = shard_sample_ids[:min(args.N, len(shard_sample_ids))]
		print(f"Subsampled to {len(shard_sample_ids)} samples")
	
	# 3. Set up shards
	print(f"Setting up shards from {args.data_dir}")
	shards = setup_shards(args.data_dir, args.split)
	if shard_id >= len(shards):
		raise ValueError(f"shard_id {shard_id} is out of range. Found {len(shards)} shards")
	shard = shards[shard_id]
	
	# 4. Load chromosome mapping
	chroms = generate_chromosome_renamer()
	
	# 5. Get list of promoter files
	jsonfiles = os.listdir(args.promoters_dir)
	
	# 6. Create output directory if needed
	if os.path.dirname(args.output_prefix) != '':
		os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
	
	output_bedgraph = f"{args.output_prefix}.bedGraph"
	output_regions = f"{args.output_prefix}_regions.bed"
	
	# 7. Process each sample
	with open(output_bedgraph, 'w') as out_f, \
		 open(output_regions, 'w') as out_region_f:
		
		# Load h5 file
		with h5py.File(args.h5_file, 'r') as h5_f:
			for shard_sample_id in tqdm(shard_sample_ids, desc="Processing samples"):
				# Get sample from shard
				sample = shard[shard_sample_id]
				
				# Extract file_id and line_id
				file_id = sample['file_id']
				line_id = sample['line_id']
				text = sample['text']
				chunk_offset = 0 # we don't have chunk offset for this dataset
				
				# Get probabilities from h5 file
				probabilities = h5_f[str(shard_sample_id)][:]
				
				# Validate that probabilities length matches text length
				if len(probabilities) != len(text):
					raise ValueError(f"Error: probabilities length ({len(probabilities)}) != text length ({len(text)}) for sample {shard_sample_id}, skipping")
				
				# Find and read the corresponding promoter file
				required_promoters_file = [f for f in jsonfiles if f.startswith(file_id)]
				if len(required_promoters_file) != 1:
					raise ValueError(f"Error: Found {len(required_promoters_file)} promoter files for {file_id}, skipping")
				required_promoters_file = os.path.join(args.promoters_dir, required_promoters_file[0])
				
				if not os.path.exists(required_promoters_file):
					raise ValueError(f"Error: Promoter file {required_promoters_file} not found, skipping")
				
				# Read promoter file and get the specific line
				with open(required_promoters_file, 'r') as pf:
					promoter_file_lines = pf.readlines()
				
				if line_id >= len(promoter_file_lines):
					raise ValueError(f"Error: line_id {line_id} is out of range for file {required_promoters_file}, skipping")
				
				j = json.loads(promoter_file_lines[line_id])
				
				# Extract required information
				promoter_strand = j['sample_strand']
				chromosome = j['chromosome']
				position = (j["start"], j["end"] + 1)
				
				# Validate position
				if position[0] >= position[1]:
					raise ValueError(f"Error: Invalid position {position} for sample {shard_sample_id}, skipping")
				
				if (position[1] - position[0]) != len(j["text"][0]):
					raise ValueError(f"Error: Position length mismatch for sample {shard_sample_id}, skipping")
				
				# Map chromosome name
				chrom_matches = chroms[chroms["RefSeq_seq_accession"] == chromosome]
				if len(chrom_matches) == 0:
					raise ValueError(f"Error: Chromosome {chromosome} not found in mapping, skipping")
				
				chromlen = int(chrom_matches["Seq_length"].values[0])
				chromosome_ucsc = chrom_matches["UCSC_style_name"].values[0]
				
				# Calculate genomic coordinates for each character position
				# The probabilities array corresponds to each character in the text
				# Map each character position to genomic coordinates
				# Similar to metrics2bedgraph.py, but here we map the entire text
				# chunk_offset is the offset within the promoter region where this text starts
				if promoter_strand == '+':
					# For + strand: start from promoter_start + chunk_offset
					genomic_start = position[0] + chunk_offset
					genomic_end = genomic_start + len(text)
				elif promoter_strand == '-':
					# For - strand: end is at promoter_end - chunk_offset
					# Start is end - text_length
					genomic_end = position[1] - chunk_offset
					genomic_start = genomic_end - len(text)
				else:
					raise ValueError(f"Error: Invalid strand {promoter_strand} for sample {shard_sample_id}, skipping")
				
				# Validate genomic coordinates
				if genomic_start < position[0] or genomic_end > position[1]:
					raise ValueError(f"Error: Genomic coordinates {genomic_start}-{genomic_end} outside promoter region {position} for sample {shard_sample_id}, skipping")
				
				if genomic_end > chromlen + 1:
					raise ValueError(f"Error: Genomic end {genomic_end} > chromlen {chromlen} for sample {shard_sample_id}, skipping")
				
				# Write bedgraph entries
				# For + strand: positions go from start to end (start + i)
				# For - strand: positions go from end to start (end - i), matching metrics2bedgraph.py
				for i, prob in enumerate(probabilities):
					if promoter_strand == '+':
						pos = genomic_start + i
					else:
						pos = genomic_end - i
					
					# Write in bedgraph format: chromosome start end value
					out_f.write(f"{chromosome_ucsc}\t{pos}\t{pos+1}\t{int(prob)}\n")
				
				# Write region
				out_region_f.write(f"{chromosome_ucsc}\t{genomic_start}\t{genomic_end}\t{promoter_strand}\n")
	
	print(f"Processed {len(shard_sample_ids)} samples")
	print(f"Output written to {output_bedgraph} and {output_regions}")


def main():
	args = parse_args()
	process_h5_file(args)


if __name__ == '__main__':
	main()

