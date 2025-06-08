#!/usr/bin/env python3

import argparse
import json
import os
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(description='Convert metrics JSON to bedgraph format')
	parser.add_argument('--metrics_json', type=str, required=True,
					  help='Path to metrics JSON file')
	parser.add_argument('--promoters_dir', type=str, required=True,
					  help='Directory containing original promoter JSON files')
	parser.add_argument('--output', type=str, required=True,
					  help='Output bedgraph file path')
	return parser.parse_args()

def process_metrics(args):
	# Read metrics JSON
	with open(args.metrics_json, 'r') as f:
		metrics_data = json.load(f)

	# Create output directory if it doesn't exist
	os.makedirs(os.path.dirname(args.output), exist_ok=True)

	# Process each record
	with open(args.output, 'w') as out_f:
		promoter_file = None
		for record in tqdm(metrics_data):
			file_id = record['file_id']
			line_id = record['line_id']
			chunk_start = record['chunk_start']
			chunk_end = record['chunk_end']
			probabilities = record['probabilities']

			# Find and read the corresponding promoter file
			requiered_promoters_file = os.path.join(args.promoters_dir, f"{file_id}.json")
			if promoter_file is None or promoter_file != requiered_promoters_file:
				promoter_file = requiered_promoters_file
				assert os.path.exists(promoter_file), f"Promoter file {promoter_file} not found"
				promoter_file_lines = open(promoter_file, 'r').readlines()

			j = json.loads(promoter_file_lines[line_id])
			# Extract required information
			promoter_strand = j['promoter_strand']
			chromosome = j['chromosome']
			position = j['position']
			
			# Calculate genomic coordinates
			if promoter_strand == '+':
				start = position + chunk_start
				end = position + chunk_end
			elif promoter_strand == '-':
				start = position - chunk_end
				end = position - chunk_start
			else:
				raise ValueError(f"Invalid promoter strand: {promoter_strand}")

			# Write bedgraph entries
			for i, prob in enumerate(probabilities):
				if promoter_strand == '+':
					pos = start + i
				else:
					pos = end - i
				
				# Write in bedgraph format: chromosome start end value
				out_f.write(f"{chromosome}\t{pos}\t{pos+1}\t{int(prob)}\n")

def main():
	args = parse_args()
	process_metrics(args)

if __name__ == '__main__':
	main() 