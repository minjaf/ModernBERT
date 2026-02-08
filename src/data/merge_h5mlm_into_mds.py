# PYTHONPATH="$PYTHONPATH:$(realpath .)" python3 src/data/merge_h5mlm_into_mds.py --input_dir /mnt/nfs_dna/shadskiy/promoters/pretrena/mds_v2/ --split train --h5_dir  data/mlm_acc_from_base_ep129  --output_dir data/promoter_mds_v2_train_with_MLM/
"""
Merge per-shard HDF5 MLM scores into an existing MDS dataset.

For each sample in the original MDS dataset, it:
  - finds the corresponding shard and sample index,
  - reads MLM scores from an HDF5 file `shard_{shard_id}.hdf5`,
  - attaches them as an `MLM` column,
  - and writes out a new MDS dataset with the extra column.

Arguments:
  --input_dir      Path to the original MDS dataset root (directory that contains split subdirs).
  --split         Name of the split to process (e.g. train / valid / test).
  --h5_dir        Path to directory with `shard_{shard_id}.hdf5` files.
  --output_dir    Path to root directory where the new MDS dataset will be written.
"""

import argparse
import os
import shutil

import h5py
import numpy as np
from streaming import MDSWriter
from tqdm.auto import tqdm

from src.text_data import NoStreamingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge HDF5 MLM scores into an existing MDS dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory of the original MDS dataset (contains split subdirectories).",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to process (e.g. train, valid, test).",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing HDF5 files named `shard_{shard_id}.hdf5`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory where the MDS dataset with MLM scores will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output split directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_split_dir = os.path.join(args.input_dir, args.split)
    if not os.path.isdir(input_split_dir):
        raise ValueError(f"Input split directory does not exist: {input_split_dir}")

    output_split_dir = os.path.join(args.output_dir, args.split)
    if os.path.exists(output_split_dir):
        if args.overwrite:
            shutil.rmtree(output_split_dir)
        else:
            raise ValueError(
                f"Output directory {output_split_dir} already exists. "
                "Use --overwrite to delete it."
            )

    os.makedirs(output_split_dir, exist_ok=True)

    h5_split_dir = os.path.join(args.h5_dir, args.split)
    if not os.path.isdir(h5_split_dir):
        raise ValueError(f"HDF5 split directory does not exist: {h5_split_dir}")

    # We only need the shard structure and spanner; we never call __getitem__,
    # so tokenizer and max_seq_len are irrelevant here.
    ds = NoStreamingDataset(
        local=args.input_dir,
        split=args.split,
        max_seq_len=1,
        tokenizer=None,
        pad_sequences=True,
    )

    # Columns follow the promoters2mds.py layout, plus the new MLM column.
    columns = {
        "text": "str",
        "file_id": "str",
        "line_id": "int",
        "chunk_offset": "int",
        "MLM": "ndarray:uint8",
    }

    last_h5 = None
    f = None
    try:
        with MDSWriter(columns=columns, out=output_split_dir, size_limit="64mb") as fout:
            for sample_id in tqdm(range(len(ds)), desc=f"Merging MLM into {args.split}"):
                shard_id, shard_sample_id = ds.spanner[sample_id]
                shard = ds.shards[shard_id]
                sample = shard[shard_sample_id]

                hdf5_file = os.path.join(h5_split_dir, f"shard_{shard_id}.hdf5")
                if last_h5 is None or last_h5 != hdf5_file:
                    if not os.path.exists(hdf5_file):
                        raise FileNotFoundError(f"Missing HDF5 file: {hdf5_file}")
                    if f is not None:
                        f.close()
                    f = h5py.File(hdf5_file, "r")
                    last_h5 = hdf5_file

                if str(shard_sample_id) not in f:
                    raise KeyError(
                        f"Sample {shard_sample_id} not found in HDF5 file {hdf5_file}"
                    )
                mlm_probs = f[str(shard_sample_id)][:]  # numpy array

                mlm_uint8 = mlm_probs.astype(np.uint8)
                sample["MLM"] = mlm_uint8

                fout.write(sample)
    finally:
        if f is not None:
            f.close()


if __name__ == "__main__":
    main()

