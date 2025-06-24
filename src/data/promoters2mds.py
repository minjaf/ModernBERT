#!/usr/bin/env python3
# conda activate bert24
# split="valid"; python promoters2mds.py --json_folder /mnt/nfs_dna/shadskiy/promoters/pretrena/$split/ --out_dir /mnt/nfs_dna/shadskiy/promoters/pretrena/mds_v2/ --split_name $split --overwrite

import argparse
import json
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streaming import MDSWriter
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Convert promoter JSON files to MDS format')
    parser.add_argument('--json_folder', type=str, required=True,
                      help='Path to folder containing JSON files')
    parser.add_argument('--out_dir', type=str, default='./data/mds/promoters/',
                      help='Output directory for MDS files'
                      )
    parser.add_argument('--split_name', type=str, required=True,
                      help='Name of the split (e.g., train, valid, test) for output directory')
    parser.add_argument('--n_files', type=int, default=None,
                      help='Number of files to process (for debugging). If None, process all files')
    parser.add_argument('--min_len', type=int, default=1000,
                      help='Minimum length of chunks to keep')
    parser.add_argument('--chunk_len', type=int, default=37000,
                      help='Length of chunks to split sequences into')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite output directory if it exists')
    return parser.parse_args()

def process_files(args):
    # Create output directory
    if args.n_files is not None:
        full_out_dir = os.path.join(args.out_dir, str(args.n_files), args.split_name)
        stats_out_dir = os.path.join(args.out_dir, str(args.n_files))
    else:
        full_out_dir = os.path.join(args.out_dir, args.split_name)
        stats_out_dir = os.path.join(args.out_dir)
    
    if os.path.exists(full_out_dir):
        if args.overwrite:
            shutil.rmtree(full_out_dir)
        else:
            raise ValueError(f"Output directory {full_out_dir} already exists. Use --overwrite to delete it.")
    os.makedirs(full_out_dir)

    # Get list of files
    jsonfiles = os.listdir(args.json_folder)
    if args.n_files is not None:
        jsonfiles = jsonfiles[:args.n_files]

    # Required keys in JSON files
    keys = ['text', 'sample_strand', 'transcript_strand', 'start', 'end',
            'transcript_ID', 'gene_ID', 'transcript_type', 'chromosome', 'genome']

    # Process files
    chunks_lengths = []
    columns = {
        "text": "str",
        "file_id": "str",
        "line_id": "int",
    }

    with MDSWriter(columns=columns, out=full_out_dir, size_limit='64mb') as fout:
        for file in tqdm(jsonfiles):
            with open(os.path.join(args.json_folder, file + f"/{file}.jsonl"), 'r') as f:
                for line_id, line in enumerate(f):
                    j = json.loads(line)
                    assert all(key in j for key in keys), f"{[k for k in keys if k not in j]}"
                    text = j['text']
                    assert len(text) == 1, f"Multiple text items for file {file}"
                    text = text[0]
                    
                    for i in range(0, len(text), args.chunk_len):
                        chunk = text[i:i+args.chunk_len]
                        if len(chunk) > args.min_len:
                            sample = {
                                'text': str(chunk).upper(),
                                'file_id': file.split('.')[0],
                                'line_id': line_id
                            }
                            fout.write(sample)
                            chunks_lengths.append(len(chunk))

    # Save statistics
    stats_df = pd.Series(chunks_lengths).value_counts().reset_index()
    stats_df.columns = ['length', 'count']
    stats_path = os.path.join(stats_out_dir, f'{args.split_name}.statistics.csv')
    stats_df.to_csv(stats_path, index=False)

    # Create and save plot
    plt.figure(figsize=(10, 6))
    sns.histplot(chunks_lengths)
    plt.yscale('log')
    plt.title(f'Distribution of chunk lengths for {args.split_name}\ntot L: {sum(chunks_lengths)} bp, tot N: {len(chunks_lengths)}')
    plt.xlabel('Chunk length')
    plt.ylabel('Count (log scale)')
    plot_path = os.path.join(stats_out_dir, f'{args.split_name}.length_distribution.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Processed {len(jsonfiles)} files")
    print(f"Created {len(chunks_lengths)} chunks")
    print(f"Statistics saved to {stats_path}")
    print(f"Plot saved to {plot_path}")

def main():
    args = parse_args()
    process_files(args)

if __name__ == '__main__':
    main() 