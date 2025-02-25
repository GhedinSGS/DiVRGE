"""
DiVRGE.py

This script processes split CIGAR text files to identify and group deletions in viral genomes.

Arguments:
    --strain, -s         Indicate strain (required)
    --ref, -r            Directory path + reference name (required)
    --file, -f           Split CIGAR text file (required)
    --save_dir, -d       Save directory (default: ./)
    --align_length, -l   Length of each portion of the read (default: 28 bp)
    --gap_size, -g       Minimum size of the gap (default: 5)
    --total_mismatch, -m Number of mismatches allowed (X) in alignment portion (default: 2)
    --total_indel, -i    Number of indels (I, D) allowed in alignment portion (default: 2)
    --group_bw, -w       Bandwidth for the grouping script (default: 5)
    --njob, -j           Number of threads (default: 2)
    --nbams, -b          Number of samples being used in idx file (default: 0)
    --gap_type, -t       Type of deletion ("N" or "D") (default: "N")

Example run:
    python3 DiVRGE.py \
        --strain H9N2 \
        --ref /path/to/ref.fasta \
        --file /path/to/split.txt \
        --save_dir /path/to/save_dir \
        --align_length 28 \
        --total_mismatch 2 \
        --group_bw 5 \
        bams 1 --njob 2 \
        --gap_type N \
        --total_indel 2

Author:
    Katherine Johnson (kate.ee.johnson12@gmail.com)

Date:
    2025-02-25

"""
import argparse
import pandas as pd
import gc
import os
import time
import math
import warnings
import logging
from joblib import Parallel, delayed

from DiVRGE_functions import (
    prep_dvg_frame, prep_one_deletions, prep_freq,
    group_one_deletion_dvgs, merge_reads_groups, 
    count_grouped_dvgs, reduce_df, open_fasta
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strain', '-s', required=True, help='Indicate strain')
    parser.add_argument('--ref', '-r', required=True, help='Directory path + ref name')
    parser.add_argument('--file', '-f', required=True, help='Split CIGAR txt file')
    parser.add_argument('--save_dir', '-d', default='.', help='Save directory (default: ./)')
    parser.add_argument('--align_length', '-l', type=int, default=28, help='Length of each portion of the read (default 28 bp)')
    parser.add_argument('--gap_size', '-g', type=int, default=5, help='Minimum size of the gap (default 5)')
    parser.add_argument('--total_mismatch', '-m', type=int, default=2, help='# of mismatches allowed (X) in alignment portion')
    parser.add_argument('--total_indel', '-i', type=int, default=2, help='# of indels (I, D) allowed in alignment portion')
    parser.add_argument('--group_bw', '-w', type=int, default=5, help='Bandwidth for the grouping script (default: 5)')
    parser.add_argument('--njob', '-j', type=int, default=2, help='Number of threads')
    parser.add_argument('--nbams', '-b', type=int, default=0, help='Number of samples being used in idx file')
    parser.add_argument('--gap_type', '-t', type=str, default="N", help='Type of deletion ("N" or "D")')
    return parser.parse_args()

def load_data(file_path):
    try:
        return pd.read_csv(file_path, sep=',', keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def main():
    start_time = time.time()
    args = parse_arguments()

    logging.info("Loading in parameters and files")
    ref = open_fasta(args.ref)
    fdf = load_data(args.file)

    if fdf.empty:
        logging.warning(f'NOTE: {args.file} was empty. Skipping.')
        return

    prefix = list(set(fdf['name']))[0] if args.nbams == 1 else args.strain
    final_output_path = os.path.join(args.save_dir, f"{prefix}.DVG.FINAL.OneGap.{args.strain}.N{args.gap_size}.Mis{args.total_mismatch}.M{args.align_length}.G{args.group_bw}.csv")

    logging.info(f"INPUT FILE: {args.file}")
    logging.info(f"OUTPUT FILENAME: {prefix}")
    logging.info(f"STRAIN: {args.strain}")

    logging.info("Extracting CIGAR string information for filtering steps")
    df = prep_dvg_frame(fdf, args.align_length, args.gap_type)

    logging.info("Filtering for single deletions that pass set thresholds")
    one_gap = df[(df.number_N == 1) & (df.tup == bool('True')) & (df.total_indel <= args.total_mismatch) & (df.total_mismatch <= args.total_mismatch) & (df.N > args.gap_size)]

    del df, fdf
    gc.collect()

    if one_gap.empty:
        logging.warning(f"NOTE: No reads with a single-deletion passed for {prefix}")
        return

    logging.info("Prepping single-deletions")
    reads_1n = prep_one_deletions(one_gap, ref, args.strain, args.gap_size, args.total_mismatch, args.align_length, args.gap_type)

    del one_gap
    gc.collect()

    if reads_1n.empty:
        logging.warning(f"No reads passed input requirements for {prefix}")
        return

    df_merge = prep_freq(reads_1n, 1)
    logging.info("Grouping single-deletions")
    group_dels = group_one_deletion_dvgs(df_merge, args.group_bw, args.njob)

    del df_merge
    gc.collect()

    d1g_m = merge_reads_groups(reads_1n, group_dels, 1)

    if d1g_m is not None:
        d1c = count_grouped_dvgs(d1g_m, 1)
        logging.info(f"Final 1N output: {final_output_path}")
        d1c = reduce_df(d1c, 1)
        d1c.to_csv(final_output_path, index=False)

        del d1g_m, d1c
        gc.collect()
    else:
        logging.warning(f"No reads passed input requirements for {prefix}")

    logging.info("Finished running")
    end_time = time.time()
    logging.info(f'Total time: {end_time - start_time:.4f} s')

if __name__ == '__main__':
    main()
