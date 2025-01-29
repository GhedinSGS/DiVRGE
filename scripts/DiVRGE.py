"""
Written by: Kate Johnson
Email: kej310@nyu.edu

LAST UPDATE: 01/10/2025
Removed 2N
Removed FeatureCounts
Adjusted for sam 1.4 formats
Added gap type (though unnecessary... bc it works with N)

run:
python3 DiVRGE.py   \
    --strain ${STRAIN}   \
    --ref ${REF}   \
    --file "/home/kate/Lab/DiVRGE/testing/N_files/New_W17_1415_d02_rep2.H9N2.split.txt"   \
    --save_dir "/home/kate/Lab/DiVRGE/testing/DVG/"   \
    --align_length ${ALIGN_LEN}   \
    --total_mismatch ${MIS}   \
    --group_bw 5 \
    --nbams 1 \
    --njob 2 \
    --gap_type N \
    --total_indel 2

"""

import argparse
import pandas as pd
import gc
import os
from joblib import Parallel, delayed
import time, math

from DiVRGE_functions import (PrepDVGFrame, PrepOneDeletions,
                           PrepFreq, GroupOneDeletionDVGs,
                           MergeReadsGroups,
                           CountGroupedDVGs,
                           ReduceDF, open_fasta)

import warnings

start = time.time()

parser = argparse.ArgumentParser()

parser.add_argument('--strain', '-s', required=True, help='Indicate strain')

parser.add_argument('--ref', '-r', required=True,
                    help='Directory path + ref name')

parser.add_argument('--file', '-f', required=True, help='Split CIGAR txt file')

parser.add_argument('--save_dir', '-d', default='.',
                    help='Save directory (default: ./) ')

parser.add_argument('--align_length', '-l', type=int, default=28,
                    help='Give the length that each portion of the read must \
                    be (default 28 bp)')

parser.add_argument('--gap_size', '-g', type=int, default=5,
                    help='Give the minimum size of the gap (default 5)')

parser.add_argument('--total_mismatch', '-m', type=int, default=2,
                    help='# of mismatches allowed (X) in alignment portion')

parser.add_argument('--total_indel', '-i', type=int, default=2,
                    help='# of indels (I, D) allowed in alignment portion')

parser.add_argument('--group_bw', '-w', type=int, default=5,
                    help='Provide the bandwidth for the grouping script (default: 5)')

parser.add_argument('--njob', '-j', type=int, default=2,
                    help='Number of threads')

parser.add_argument('--nbams', '-b', type=int, default=0,
                    help='Provide number of samples being used in idx file')

parser.add_argument('--gap_type', '-t', type=str, default="N",
                    help='Provide if the deletion is "N" or "D"')

args = parser.parse_args()

if __name__ == '__main__':
    print("")
    print("Loading in parameters and files")
    bandwidth = args.group_bw
    infile = args.file  # read in split read file
    njob = args.njob
    ref = open_fasta(args.ref)
    try:
        fdf = pd.read_csv(infile, sep=',', keep_default_na=False)  # read split read file
    except pd.errors.EmptyDataError:
        fdf = pd.DataFrame()

    if fdf.shape[0] > 0 :
        prefix = list(set(list(fdf['name'])))[0] if args.nbams == 1 else args.strain
        print("INPUT FILE: {0}".format(infile))
        print("OUTPUT FILENAME: {0}".format(prefix))
        print('STRAIN: {0}'.format(args.strain))
        print("")
        print("")

        Final1N = "{0}/{1}.DVG.FINAL.OneGap.{6}.N{2}.Mis{3}.M{4}.G{5}.csv".format(
                args.save_dir, prefix, args.gap_size,
                args.total_mismatch, args.align_length, args.group_bw,
                args.strain)

        print("Extracting CIGAR string information for filtering steps")
        # if you are interested in single-deletion read specific information (not filtered, not grouped) -
        # then save this dataframe (df) below
        df = PrepDVGFrame(fdf, args.align_length, args.gap_type)  # prep for filt
        
        print("")
        print("Filtering for single deletions that pass set thresholds")
        print("MATCH LENGTH: {0}".format(args.align_length))
        print("GROUPING SIZE: {0}".format(args.group_bw))
        print("MIN. DELETION SIZE: {0}".format(args.gap_size))
        print("# OF INDEL ALLOWED: {0}".format(args.group_bw))
        print("# OF MISMATCH ALLOWED: {0}".format(args.group_bw))
        print("CIGAR DELETION: {0}".format(args.gap_type))
        # If you are interested in the 1N read information that is filtered - but NOT grouped - save the - 
        # following dataframe (one_gap)
        one_gap = df[(df.number_N == 1) & (df.tup == bool('True')) & (
                df.total_indel <= args.total_mismatch) & (
                df.total_mismatch <= args.total_mismatch) & (df.N > args.gap_size)]  # temp df of group data

        
        # remove from memory
        del df
        del fdf
        gc.collect()

        # SINGLE DELETIONS 
        print("Prepping single-deletions")

        if not one_gap.empty:
            # Returns candidate one deletion DVGs:
            Reads1N = PrepOneDeletions(one_gap, ref, args.strain,
                                        args.gap_size, args.total_mismatch,
                                        args.align_length, args.gap_type)

            del one_gap
            gc.collect()

            if not Reads1N.empty:
                # groups files which will be output from grouping
                df_merge = PrepFreq(Reads1N, 1)

                print("")
                print("Grouping single-deletions")
                group_dels = GroupOneDeletionDVGs(df_merge, bandwidth, njob)

                del df_merge
                gc.collect()

                # Merging read information with NewGap information
                d1g_m = MergeReadsGroups(Reads1N, group_dels, 1)

                if d1g_m is not None: 
                    # Count the New Gap information for each sample and segment
                    d1c = CountGroupedDVGs(d1g_m, 1)

                    print("Final 1N output: {0}".format(Final1N))
                    d1c = ReduceDF(d1c, 1)
                    d1c.to_csv(Final1N, index=False)

                    del d1g_m
                    del d1c
                    gc.collect()
                else:
                    print("No reads passed input requirements for {0}".format(prefix))
        else:
            print("NOTE: No reads with a single-deletion passed for {0}".format(prefix))

    elif fdf.shape[0] == 0:
        print('NOTE: {0} was empty. Skipping.'.format(infile))

print("Finished running")

end = time.time()

print('{:.4f} s'.format(end-start))
