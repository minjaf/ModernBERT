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
	parser.add_argument('--output_prefix', type=str, required=True,
					  help='Prefix for output files. Will create is_true.bedGraph and predicted_nt.bed')
	return parser.parse_args()

def process_metrics(args):
	# Read metrics JSON
	# Create output directory if it doesn't exist
	output_prob = f"{args.output_prefix}_is_true.bedGraph"
	output_regions = f"{args.output_prefix}_regions.bed"
	output_regions_all = f"{args.output_prefix}_regions_all.bed"
	output_TSS_all = f"{args.output_prefix}_TSS_all.bed"
	
	if os.path.dirname(args.output_prefix) != '':
		os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
	
	jsonfiles = os.listdir(args.promoters_dir)
	with open(output_prob, 'w') as out_f, \
		open(output_regions, 'w') as out_region_f, \
		open(output_regions_all, 'w') as out_region_all_f, \
		open(output_TSS_all, 'w') as out_TSS_all_f:

		with open(args.metrics_json, 'r') as f:
			promoter_file = None
			n_lines = 0
			for line in f:
				n_lines += 1
				record = json.loads(line)
				file_id = record['file_id']
				line_id = record['line_id']
				chunk_start = record['chunk_start']
				chunk_end = record['chunk_end']
				probabilities = record['probabilities']

				# Find and read the corresponding promoter file
				requiered_promoters_file = [f for f in jsonfiles if f.startswith(file_id)]
				assert len(requiered_promoters_file) == 1, f"Multiple promoter files found for {file_id}: {requiered_promoters_file}"
				requiered_promoters_file = os.path.join(args.promoters_dir, requiered_promoters_file[0])
				if promoter_file is None or promoter_file != requiered_promoters_file:
					promoter_file = requiered_promoters_file
					print (f"Processing {promoter_file}")
					assert os.path.exists(promoter_file), f"Promoter file {promoter_file} not found"
					promoter_file_lines = open(promoter_file, 'r').readlines()
					for line in promoter_file_lines:
						j = json.loads(line)
						out_region_all_f.write(f"{j['chromosome']}\t{j['position'][0]}\t{j['position'][-1]}\t{j['gene_ID']+':'+j['transcript_ID']}\t.\t{j['promoter_strand']}\n")
						out_TSS_all_f.write(f"{j['chromosome']}\t{j['position'][1]}\t{j['position'][1]+1}\t{j['gene_ID']+':'+j['transcript_ID']}\t.\t{j['promoter_strand']}\n")

				j = json.loads(promoter_file_lines[line_id])
				# Extract required information
				promoter_strand = j['promoter_strand']
				chromosome = j['chromosome'].replace('NC_000001.11', 'chr1').replace('NC_000002.12', 'chr2')
				position = j['position']
				assert position[0]<position[-1], f"Position {position} is not valid"
				assert (position[-1]-position[0])==len(j["text"][0]), f"Position {position} is not valid"
				
				# Calculate genomic coordinates
				if promoter_strand == '+':
					start = position[0] + chunk_start
					end = position[0] + chunk_end
				elif promoter_strand == '-':
					start = position[-1] - chunk_end
					end = position[-1] - chunk_start
				else:
					raise ValueError(f"Invalid promoter strand: {promoter_strand}")
				
				assert end>start, f"End {end} is not greater than start {start}"
				assert end-start==len(probabilities), f"End {end} is not greater than start {start}"
				assert start>=position[0], f"Start {start} is not greater than position {position[0]}"
				assert end<=position[-1], f"End {end} is not less than position {position[-1]}"
				
				# Write bedgraph entries
				for i, prob in enumerate(probabilities):
					if promoter_strand == '+':
						pos = start + i
					else:
						pos = end - i
					
					# Write in bedgraph format: chromosome start end value
					out_f.write(f"{chromosome}\t{pos}\t{pos+1}\t{int(prob)}\n")
				out_region_f.write(f"{chromosome}\t{start}\t{end}\t{promoter_strand}\n")

		print (f"Processed {n_lines} lines")

def main():
	args = parse_args()
	process_metrics(args)

if __name__ == '__main__':
	main() 