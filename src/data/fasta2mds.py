#!/usr/bin/env python3
"""
Convert a FASTA file to MDS (MosaicML Streaming) format.
Writes each sequence with columns "text" (uppercased) and "id" (from header, before first '|').

Usage:
  python src/data/fasta2mds.py --fasta data/metavr/IMGVR5_UViG.fna --out_dir data/metavr/mds/ --overwrite
"""

import argparse
import gzip
import shutil
from pathlib import Path

from Bio import SeqIO
from streaming import MDSWriter
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert FASTA file to MDS format (streaming dataset)."
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to FASTA file (.fa, .fasta, .fa.gz, .fasta.gz)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/mds/fasta/",
        help="Output directory for MDS shards",
    )
    parser.add_argument(
        "--size_limit",
        type=str,
        default="64mb",
        help="Max size per MDS shard (e.g. 64mb)",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=1000,
        help="Skip sequences shorter than this (bp)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    return parser.parse_args()


def _seq_id(record) -> str:
    # e.g. >IMGVR_UViG_2582581227_000001|2582581227|2582690522
    return record.description

def _open_fasta(path: str):
    path = Path(path)
    if path.suffix == ".gz" or (len(path.suffixes) >= 2 and path.suffixes[-1] == ".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def main():
    args = parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise ValueError(
                f"Output directory {out_dir} already exists. Use --overwrite to delete it."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = {"text": "str", "id": "str"}
    n_written = 0
    n_skipped_short = 0

    with _open_fasta(args.fasta) as fh:
        with MDSWriter(
            columns=columns,
            out=str(out_dir),
            size_limit=args.size_limit,
        ) as fout:
            for record in tqdm(
                SeqIO.parse(fh, "fasta"),
                desc="FASTA → MDS",
                unit=" seq",
            ):
                seq = str(record.seq).upper()
                if len(seq) < args.min_len:
                    n_skipped_short += 1
                    continue
                seq_id = _seq_id(record)
                fout.write({"id": seq_id, "text": seq})
                n_written += 1

    print(f"Wrote {n_written} sequences to {out_dir}")
    if n_skipped_short:
        print(f"Skipped {n_skipped_short} sequences shorter than {args.min_len} bp.")


if __name__ == "__main__":
    main()
