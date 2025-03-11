"""
Title: Pull Ghost Reads

Description:
    This script caculates total read abundance across an alignment (BAM format).
    It pulls information from both 'matching' or 'aligned' reads and 'gap' regions ('ghost' reads)
    and generates a summary of gap-spanning read coverage at each position along the
    reference genome. The results are saved to a CSV file. 

Usage:
    python pull_ghost_reads.py --ref <reference.fasta> --infile <input.bam> \
                                        --qual_cutoff <quality_cutoff> --strain <strain_name> \
                                        --outputdir <output_directory> --njobs <num_jobs>

Arguments:
    --ref, -r          : Full path to the reference FASTA file. (required)
    --infile, -i       : Input BAM file. (required)
    --qual_cutoff, -q  : Phred quality cutoff (default: 30)
    --strain, -T       : Strain name (default: 'strain')
    --outputdir, -o    : Output directory (default: './')
    --njobs, -n        : Number of jobs to parallelize (default: 2)

Author:
    Kate Johnson (kate.ee.johnson12@gmail.com)

Date:
    2025-02-26

Version:
    1.0

License:
    This script is released under the MIT License.
"""
import sys
import os
import time
import argparse
import pysam
import logging
#import multiprocessing
import numpy as np
import pandas as pd
#from functools import partial
from collections import defaultdict
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--ref', '-r', required=True, help='Full path to reference file.')
parser.add_argument('--infile', '-i', help='Input single BAM file.')
parser.add_argument('--qual_cutoff', '-q', type=int, default=30, help='Phred quality cutoff (default is 30)')
parser.add_argument('--strain', '-T', type=str, default='strain', help='Strain name')
parser.add_argument('--outputdir', '-o', default='./', help='Output directory')
parser.add_argument('--njobs', '-n', default=2, type=int, help='Number of jobs to parallelize')
args = parser.parse_args()

def ensure_dir(directory):
    """Ensure output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def analyzer(is_reverse, updated_out, qual_cutoff, reverse_dict, forward_dict, consensus_dict, insertion_dict):
    """Analyze read data and update dictionaries."""
    cig, seq, ntpos, qual = updated_out
    temp_ins_dict = defaultdict(list)

    for c, nt, pos, q in zip(cig, seq, ntpos, qual):
        if pos is None and c in {'S', 'D', 'N'}:
            continue
        if c == 'I':
            temp_ins_dict[pos].append(nt)
        elif q == '-' or q >= qual_cutoff:
            target_dict = reverse_dict if is_reverse else forward_dict
            target_dict[pos][nt] += 1
            consensus_dict[pos][nt] += 1

    for ntpos, nts in temp_ins_dict.items():
        full_nt = ''.join(nts)
        insertion_dict[ntpos][full_nt] += 1


def seq_updater(cigartuple, read, readidx, readq):
    """Expand cigar tuple to the length of the read."""
    updated_cigar_seq = []
    updated_seq = []
    updated_idx = []
    updated_readq = []
    idxctr = 0

    for identifier, length in cigartuple:
        if identifier in {0, 7, 8}:  # match or mismatch
            updated_cigar_seq.extend('M' * length)
            updated_seq.extend(read[idxctr:idxctr + length])
            updated_readq.extend(readq[idxctr:idxctr + length])
            idxctr += length
        elif identifier == 1:  # insertion
            updated_cigar_seq.extend('I' * length)
            updated_seq.extend(read[idxctr:idxctr + length])
            updated_readq.extend(readq[idxctr:idxctr + length])
            idxctr += length
        elif identifier == 2:  # deletion
            updated_cigar_seq.extend('D' * length)
            updated_seq.extend('-' * length)
            updated_readq.extend('-' * length)
        elif identifier == 4:  # soft clip
            updated_cigar_seq.extend('S' * length)
            updated_seq.extend(read[idxctr:idxctr + length])
            updated_readq.extend(readq[idxctr:idxctr + length])
            idxctr += length
        elif identifier == 3:  # gap
            updated_cigar_seq.extend('N' * length)
            updated_seq.extend('N' * length)
            updated_readq.extend('-' * length)
        elif identifier == 5:  # hard clip
            continue
        else:
            sys.exit(f"Invalid cigar identifier: {identifier} in {cigartuple}")

    idxctr = 0
    last_delnt = last_insnt = last_skip = 0

    for i, j, q in zip(updated_cigar_seq, updated_seq, updated_readq):
        if i in {'D', 'N'}:
            last_delnt = last_delnt if last_delnt is not None else 0
            last_delnt += 1
            updated_idx.append(last_delnt)
        elif i == 'I':
            updated_idx.append(last_insnt)
            idxctr += 1
        else:
            updated_idx.append(readidx[idxctr])
            last_delnt = last_insnt = readidx[idxctr]
            idxctr += 1

    assert len(updated_cigar_seq) == len(updated_seq) == len(updated_idx) == len(updated_readq)

    return updated_cigar_seq, updated_seq, updated_idx, updated_readq


def read_fasta(fp):
    """Read input fasta."""
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield name, ''.join(seq)
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield name, ''.join(seq)


def open_fasta(filename):
    """Open, read, and return reference fasta as dictionary."""
    segdict = {}
    with open(filename) as fp:
        for name, seq in read_fasta(fp):
            segdict[name[1:]] = seq
    return segdict


def printer(consensus_dict, sampname, segment, ntpos):
    """Adjust the data to have accurate counts for each nt pos in the genome."""
    tempd = consensus_dict[ntpos]
    ntpos += 1  # change from 0-index to 1-index
    # create a DataFrame directly from tempd
    data = {
        'name': [sampname],
        'segment': [segment],
        'ntpos': [ntpos],
        'N': [tempd.get('N', 0)],  # if nothing - make zero
        'totalcount': [sum(tempd.values())],
        'mapped_totalcount': [sum(tempd.values()) - tempd.get('N', 0)]
    }
    return pd.DataFrame(data)


def process_chunk(region, bamname, ref_dict, qual_cutoff):
    """Process a chunk of the BAM file."""
    samfile = pysam.AlignmentFile(bamname, "rb")

    forward_dict = defaultdict(lambda: defaultdict(int))  # initializing dictionaries using default dict
    reverse_dict = defaultdict(lambda: defaultdict(int))
    insertion_dict = defaultdict(lambda: defaultdict(int))
    consensus_dict = defaultdict(lambda: defaultdict(int))

    for read in samfile.fetch(region=region):
        if not read.is_unmapped:
            cigartup = read.cigartuples
            unfiltreadidx = read.get_reference_positions(full_length=True)
            unfiltread = list(read.query_sequence)
            unfiltreadqual = read.query_qualities
            is_reverse = read.is_reverse
            analyzer(is_reverse, seq_updater(cigartup, unfiltread, unfiltreadidx, unfiltreadqual), qual_cutoff, reverse_dict, forward_dict, consensus_dict, insertion_dict)

    samfile.close()
    return forward_dict, reverse_dict, insertion_dict, consensus_dict


def merge_dicts(dicts):
    """Merge multiple dictionaries into one."""
    result = defaultdict(lambda: defaultdict(int))
    for d in dicts:
        for key, subdict in d.items():
            for subkey, value in subdict.items():
                result[key][subkey] += value
    return result


def main_function(sample_name, segment, ref_dict, bamname, qual_cutoff=30, njobs=4):
    """Main function to process BAM file and calculate read abundance in parallel."""
    # Split the segment into smaller regions for parallel processing
    seglen = len(ref_dict[segment])  # Pull segment length information to determine chunk size
    chunk_size = seglen // njobs
    # establishing naming of regions to pull - read by pysam
    regions = [f"{segment}:{i}-{min(i + chunk_size, seglen)}" for i in range(1, seglen, chunk_size)]
    results = Parallel(n_jobs=njobs, backend="loky")(delayed(process_chunk)(region, bamname, ref_dict, qual_cutoff) for region in regions)

    # Merge the results from all processes
    forward_dicts, reverse_dicts, insertion_dicts, consensus_dicts = zip(*results)
    forward_dict = merge_dicts(forward_dicts)
    reverse_dict = merge_dicts(reverse_dicts)
    insertion_dict = merge_dicts(insertion_dicts)
    consensus_dict = merge_dicts(consensus_dicts)


    # print out ntpos specific information 
    ntpos_list = sorted(list(range(0,seglen)))  # using segment length info - dict.keys() won't work if no reads/no coverage
    segout = [printer(consensus_dict, sample_name, segment, ntpos) for ntpos in ntpos_list]
    return pd.concat(segout)


if __name__ == '__main__':
    start_time = time.time()
    ensure_dir(args.outputdir)
    logging.info(f"Files will be saved in: {args.outputdir}")

    qual_cutoff = args.qual_cutoff
    ref_dict = open_fasta(args.ref)
    logging.info(f"Input reference: {args.ref}")

    sample_name = os.path.basename(args.infile).split('.')[0]
    logging.info(f"Processing sample: {sample_name}")
    strain = args.strain
    outfile = f"{args.outputdir}/{sample_name}.{strain}.q{qual_cutoff}.rd.csv"
    logging.info(f"Outfile location and name: {outfile}")

    result = []
    
    for segment in ref_dict:
        # run for each segment/chrom
        result.append(main_function(sample_name, segment, ref_dict, args.infile, qual_cutoff, args.njobs))
    
    result = pd.concat(result, ignore_index = True)

    result.to_csv(outfile, header=True, index=False)

    logging.info("Finished running")
    end_time = time.time()
    logging.info(f'Total time: {end_time - start_time:.4f} s')