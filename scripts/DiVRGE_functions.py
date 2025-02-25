"""
Written by: Kate Johnson

DiVRGE_functions.py

This module contains utility functions for the DiVRGE project, including functions for
reading and processing FASTA files, estimating fragment sizes, grouping deletions, and
more.

Functions:
- read_fasta(fp): Reads a FASTA file.
- open_fasta(filename): Opens a FASTA file and returns a DataFrame with segment sizes.
- frame_check(del_estimated_size): Checks if the open reading frame is maintained.
- est_size(segment_length, gap_size): Estimates the fragment size after deletion.
- mnm(start, m1, n, m2): Calculates the deletion location information.
- printer1(outfile, name, segment, readname, gap, gap_start, gap_end, estimated_size, length, segment_length, frame_flag, readflags, strain, cigar): Prints information to a file.
- prep_dvg_frame(input_df, min_align_length, gap_type): Prepares a DataFrame for DVG analysis.
- prep_one_deletions(one_gap, ref, strain, gap_size, total_mismatch, align_length, gap_type): Prepares deletions for one-gap analysis.
- prep_freq(df, gap_number): Prepares frequency information for gaps.
- grouping_one(segment, name, df, bandwidth): Groups deletions for a specific segment and sample.
- group_one_deletion_dvgs(indataframe, bandwidth, njob): Groups deletions for all samples.
- merge_reads_groups(df, d1g, gap_number): Merges read and group information.
- count_grouped_dvgs(df, gap_number): Counts grouped deletions.
- reduce_df(df, gap_number): Reduces the DataFrame for figure generation.
"""

import os
import re
import time
import math
import warnings
from typing import List, Tuple, Dict, Generator

import pandas as pd
import numpy as np
from scipy.stats.distributions import binom  # scikit-learn
from sklearn.cluster import MeanShift # scikit-learn
from joblib import Parallel, delayed


def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))


def open_fasta(filename) -> pd.DataFrame:
    segdict = {}
    with open(filename) as fp:
        for name, seq in read_fasta(fp):
            segdict[name[1:]] = len(seq)
    return pd.DataFrame({'segment': segdict.keys(), 'segment_size': segdict.values()})


def frame_check(del_estimated_size) -> str:
    """
    INPUT: the estimated frag. size
    OUTPUT: if in frame (T=true or F=false)
    """
    return 'T' if del_estimated_size % 3 == 0 else 'F'


def est_size(segment_length: int, gap_size: int) -> Tuple[str, int]:
    """
    INPUT: chrom/seg size, size of deletion
    OUTPUT: estimated frag. size remaining after deletion
    """
    cds_size = segment_length - gap_size
    frame_flag = frame_check(gap_size)
    return frame_flag, cds_size


def mnm(start:int, m1: int, n: int, m2: int) -> Tuple[str, int, int]:
    """
    INPUT: CIGAR string, left-most mapping position
    OUTPUT: info on deletion location
    """
    gap_start = start + m1
    gap_end = gap_start + (n-1)
    gap = f'{gap_start}_{gap_end}'
    return gap, gap_start, gap_end


def printer1(outfile, name: str, segment: str, readname: str, gap: str, gap_start: int, gap_end: int,
             estimated_size: int, length: int, segment_length: int, frame_flag: str,
             readflags: str, strain: str, cigar: str) -> None:
    """
    INPUT: info on reads that span deletion
    OUTPUT: print out info to file
    """
    print(f'{name},{segment},{readname},{gap},{gap_start},{gap_end},{estimated_size},{length},{segment_length},{frame_flag},{readflags},{strain},{cigar}', file=outfile)


def prep_dvg_frame(input_df, min_align_length, gap_type = "N") -> pd.DataFrame:
    """
    INPUT: split read file (split_df) and features count file (features_df)
    OUTPUT: df with length/cigar checks to be filtered
    """

    input_df = input_df.astype({"name": str})  # make name string
    
    cigs = list(input_df['cigar']) # list cigars

    input_df['number_N'] = [cigar.count("N") for cigar in cigs] # count the # of N/cigar
    
    input_df['number_D'] = [cigar.count("D") for cigar in cigs] # count the # of D/cigar
    
    df = input_df[(input_df.number_N == 1)].copy()  # early filter - how many "N" do we want to work with
    
    cigs = list(df['cigar'])

    break_cigar = [re.findall(r'(\d+)([A-Za-z=])', x) for x in cigs]

    # Set empty lists - will be added to df - fast
    total_indel, total_mismatch, cigar_len, tup, no_soft_list, n, m1, m2 = [], [], [], [], [], [], [], []

    for cig_tuple in break_cigar:

        total_indel.append(sum(int(x[0]) for x in cig_tuple if x[1] in {'I', 'D'}))
        
        total_mismatch.append(sum(int(x[0]) for x in cig_tuple if x[1] == 'X'))
    
        no_soft = [i for i in cig_tuple if i[1] not in {'S', 'I'}] # doesn't include soft clips or insertions in alignment

        no_soft_list.append(no_soft)

        cigar_len.append(len(no_soft))  # used to help find the 'matching' (x & =) before and after N
         
        n_idx = next(i for i, x in enumerate(no_soft) if x[1] == gap_type)  # used to know where N is located so we can determine match before and after
        
        # returns a tup (true/false) as to whether match lengths before after n match
        # match is old school match (X + =) 
        # added a new filter flag (mismatch number) to allow us to filter by that instead 
        align_pass = [
            (sum(int(no_soft[x][0]) for x in range(0, n_idx) if no_soft[x][1] in {'X', '=', 'M'}) - sum(int(no_soft[x][0]) for x in range(0, n_idx) if no_soft[x][1] == 'D')) >= min_align_length,
            (sum(int(no_soft[x][0]) for x in range(n_idx + 1, len(no_soft)) if no_soft[x][1] in {'X', '=', 'M'}) - sum(int(no_soft[x][0]) for x in range(n_idx + 1, len(no_soft)) if no_soft[x][1] == 'D')) >= min_align_length
        ]
        
        tup.append(all(align_pass))
        
        # m1, m2, n will be used to calculate other things like size, fragment length, etc
        m1.append(sum(int(no_soft[x][0]) for x in range(0, n_idx)))
        m2.append(sum(int(no_soft[x][0]) for x in range(n_idx + 1, len(no_soft))))
        n.append(sum(int(x[0]) for x in no_soft if x[1] == 'N'))

    df['cigar_len'] = cigar_len
    df['total_indel'] = total_indel
    df['no_soft'] = no_soft_list
    df['tup'] = tup
    df['total_mismatch'] = total_mismatch
    df['N'] = n
    df['M1'] = m1
    df['M2'] = m2

    return(df)


def prep_one_deletions(one_gap: pd.DataFrame, ref: pd.DataFrame, 
                    strain: str, gap_size: int, total_mismatch: int, 
                    align_length: int, gap_type: str = "N") -> pd.DataFrame:

    """    
    INPUT: the estimated size using the CDS coordinates
    OUTPUT: Whether the estimated size is divisible by 3 and
    therefore maintaining open read frame
    """
    one_gap = pd.merge(one_gap, ref, on='segment', how='left')

    header = ['name', 'segment', 'readname', 'gap', 'gap_start', 'gap_end', 'gap_size', 'estimated_length', 'segment_size', 'frame_flag', 'readflags', 'strain', 'cigar']
    
    updated_df = [] # Iterate through rows to generate gap info for one and two gapped reads

    for index, row in one_gap.iterrows():
        gap, gap_start, gap_end = mnm(row['left_pos'], row['M1'], row['N'], row['M2'])  # calculate the following using the function mnm for the pattern
        
        frame_flag, est_length = est_size(row['segment_size'], row['N'])

        updated_df.append([row['name'], row['segment'], row['readname'], gap, gap_start, gap_end, row['N'], est_length, row['segment_size'], frame_flag, row['flags'], strain, row['cigar']])

    return pd.DataFrame(updated_df, columns=header)

def prep_freq(df: pd.DataFrame, gap_number: int) -> pd.DataFrame:
    """
    INPUT: The candidate gap dataframe
    OUTPUT: Outputs dataframe to be written as a csv file
    """
    if gap_number == 1:
        groups = ['name', 'segment', 'gap_start', 'gap_end', 'gap', 'gap_size', 'estimated_length', 'frame_flag']
        dvg_freq = df.groupby(groups).size().reset_index(name="freq")
        return dvg_freq.drop_duplicates()

def grouping_one(segment: str, name: str, df: pd.DataFrame, bandwidth: int = 5) -> pd.DataFrame:
    """
    INPUT: Freq. information for each sample
    OUTPUT: Grouping information for each sample/segment
    """
    # if a segment lacks DVG information - this will throw a warning
    warnings.filterwarnings("ignore", category=UserWarning)
    #print(f"Grouping {segment} 1N DVGs for {name}")

    dfs = df[(df.segment == segment) & (df.name == name)].copy() # subset dataframe for given segment

    if not dfs.empty:
        starts = list(dfs['gap_start'])
        ends = list(dfs['gap_end'])
        dvgs = np.array(list(zip(starts, ends)))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(dvgs)
        labels = ms.labels_
        labels_unique = list(np.unique(labels))

        grouping_info = list(zip(
            dfs['segment'], starts, ends, dfs['gap'],
            dfs['gap_size'], dfs['estimated_length'],
            dfs['freq'], labels
        ))

        g_df = pd.DataFrame(grouping_info, columns=[
            'segment', 'gap_start', 'gap_end', 'gap', 'gap_size',
            'estimated_length', 'freq', 'DVG_group'
        ])

        for group in labels_unique:
            t = g_df[g_df.DVG_group == group]
            max_freq = t.freq.max()
            g_df.loc[g_df.DVG_group == group, 'Deletion'] = t[t.freq == max_freq]['gap'].iloc[0]
            g_df.loc[g_df.DVG_group == group, 'DeletionStart'] = t[t.freq == max_freq]['gap_start'].iloc[0]
            g_df.loc[g_df.DVG_group == group, 'DeletionEnd'] = t[t.freq == max_freq]['gap_end'].iloc[0]
            g_df.loc[g_df.DVG_group == group, 'GroupBoundaries'] = f"{t['gap_start'].min()}-{t['gap_start'].max()}_{t['gap_end'].min()}-{t['gap_end'].max()}"
            g_df.loc[g_df.DVG_group == group, 'DeletionSize'] = t[t.freq == max_freq]['gap_size'].iloc[0]
            g_df.loc[g_df.DVG_group == group, 'EstimatedFragLength'] = t[t.freq == max_freq]['estimated_length'].iloc[0]

        return g_df
    else:
        #print(f'No deletions {segment} {name}')
        return pd.DataFrame()


def group_one_deletion_dvgs(indataframe: pd.DataFrame, bandwidth: int, njob: int) -> pd.DataFrame:
    """
    INPUT: freq. dataframe for all samples
    OUTPUT: group dataframe for all samples
    """
    df = indataframe
    segments = list(set(df['segment']))
    names = list(set(df['name']))
    master_df = pd.concat(Parallel(n_jobs=njob)(
        delayed(grouping_one)(s, n, df, bandwidth) for s in segments for n in names
    ))
    #print("Grouping 1N finished")
    return master_df
    

def merge_reads_groups(df: pd.DataFrame, d1g: pd.DataFrame, gap_number: int) -> pd.DataFrame:
    """
    INPUT: Read file info, group file, and number of deletions
    OUTPUT: df with read and group info, used to calc: rpkm, percentage, etc.
    """
    if gap_number == 1:
        if not d1g.empty:
            d1g_m = pd.merge(df, d1g, how='left', on=[
                'segment', 'gap', 'gap_start', 'gap_end', 'gap_size', 'estimated_length'
            ])
            d1g_m = d1g_m.drop_duplicates()
            d1g_m = d1g_m.rename(columns={"freq": "FreqAcrossSamples"})
            return d1g_m
        else:
            #print('No gap 1 files to merge')
            return pd.DataFrame()


def count_grouped_dvgs(df: pd.DataFrame, gap_number: int) -> pd.DataFrame:
    """
    INPUT: Take in grouping/read merged dataframe from merged_read_groups,
    count the new gap information for each sample

    OUTPUT: new dataframe that has the freq info for total number of gaps and
    Count information for each individual dvg type within the samples.
    Counts calculated with "New" gaps generated by grouping script
    """
    if gap_number == 1 and df is not None:
        dvg_freq = df.groupby(['name', 'segment', 'DeletionStart', 'DeletionEnd', 'Deletion']).size().reset_index(name="deletion_count")
        total_dvg = df.groupby(['name', 'segment']).size().reset_index(name="SegTotalDVG")
        dvg_c = pd.merge(dvg_freq, total_dvg, on=['name', 'segment'])
        dvg_f = pd.merge(df, dvg_c, on=['name', 'segment', 'DeletionStart', 'DeletionEnd', 'Deletion'])
        dvg_f = dvg_f.drop_duplicates()
        return dvg_f
    else:
        #print("No single gaps to count")
        return pd.DataFrame()


def reduce_df(df: pd.DataFrame, gap_number: int) -> pd.DataFrame:
    """
    INPUT: Read dataframe
    OUTPUT: A reduced dataframe, with info that we care about for figures
    """
    if gap_number == 1:
        dropcols = ['readname', 'gap', 'gap_start', 'gap_end', 'gap_size', 'estimated_length', 'frame_flag', 'readflags', 'FreqAcrossSamples', 'cigar']
        df = df.drop(dropcols, axis=1)
        df = df.drop_duplicates()
        return df