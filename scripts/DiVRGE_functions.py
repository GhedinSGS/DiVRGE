"""
Written by: Kate Johnson
Functions for finding DVGs in NGS data
Updated 01/10/2025
Removed 2N sections
"""

import pandas as pd
from scipy.stats.distributions import binom  # scikit-learn
import re
import numpy as np
import os
from sklearn.cluster import MeanShift # scikit-learn
from joblib import Parallel, delayed
import time, math
import warnings


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


def open_fasta(filename):
    # filename = '../FILES/reference/'+someref
    segdict = {}
    with open(filename) as fp:
        for name, seq in read_fasta(fp):
            segdict[name[1:]] = len(seq)
    return pd.DataFrame({'segment': segdict.keys(), 'segment_size': segdict.values()})


def frame_check(del_estimated_size):
    """
    INPUT: the estimated frag. size
           will not work without CDS
    OUTPUT: if orf is maintained (T=true or F=false)
    """
    if del_estimated_size % 3 == 0:  # if the remainder is zero-ORF maintained
        frame_flag = 'T'
    elif del_estimated_size % 3 != 0:  # elif the remainder is not zero
        frame_flag = "F"
    return frame_flag


def EstSize(segment_length, gap_size):
    """
    INPUT: chrom/seg size, size of deletion
    OUTPUT: estimated frag. size remaining after deletion
    """
    CDS_size = int(segment_length) - int(gap_size)
    frame_flag = frame_check(int(gap_size))
    return frame_flag, CDS_size


def MNM(start, M1, N, M2):
    """
    INPUT: CIGAR string, Left-most mapping position
    OUTPUT: The deletion location information
    """
    gap_start = start + M1
    gap_end = gap_start + (N-1)
    gap = '{0}_{1}'.format(gap_start, gap_end)
    return gap, gap_start, gap_end


def printer1(outfile, name, segment, readname, gap, gap_start, gap_end,
             estimated_size, length, segment_length, frame_flag,
             readflags, strain, cigar):
    """
    # CHECK THAT WE USE FOR OTHER THINGS - OTHERWISE DELETE
    INPUT: info on reads that span deletion
    OUTPUT: print out info to file
    """
    print('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}'.format(
        name, segment, readname, gap, gap_start, gap_end,
        estimated_size, length, segment_length, frame_flag,
        readflags, strain, cigar), end="\n", file=outfile)


def PrepDVGFrame(input_df, min_align_length, gap_type = "N"):
    """
    INPUT: Split read file (split_df) and features count file (features_df)
    OUTPUT: DF with length/cigar checks to be filtered
    ONLY WORKS ON 1N - 2+ DELETIONS/READ WILL NOT WORK HERE
    """

    input_df = input_df.astype({"name": str})  # make name string
    
    cigs = list(input_df['cigar']) # list cigars

    input_df['number_N'] = [cigar.count("N") for cigar in cigs] # count the # of N/cigar
    input_df['number_D'] = [cigar.count("D") for cigar in cigs] # count the # of D/cigar
    #!!!!!
    df = input_df[(input_df.number_N == 1)].copy()  # early filter - how many "N" do we want to work with
    cigs = list(df['cigar'])

    break_cigar = [re.findall(r'(\d+)([A-Za-z=])', x) for x in cigs]

    # Set empty lists - will be added to df - fast
    # At end, lists and df should be same len. Can check with print(len(list))
    total_indel = []  # list of total indels - includes I and D
    total_mismatch =[] # list of total mismatches - only X
    Cigar_len = []  # Length of cigar
    tup = []  # List containing T/F info on Length pass check
    no_soft_list = []
    N = []
    M1 = []
    M2 = []

    for cig_tuple in break_cigar:
        total_indel.append(sum([int(x[0]) for x in cig_tuple if x[1] == 'I' or x[1] == 'D'])) ## UPDATED 20250109
        total_mismatch.append(sum([int(x[0]) for x in cig_tuple if x[1] == 'X']))

        # doesn't include soft clips or insertions in alignment
        no_soft = [i for i in cig_tuple if i[1] != 'S' and i[1] != 'I'] 
        no_soft_list.append(no_soft)        
        
        cigar_len = len(no_soft) # used to help find the 'matching' (x & =) before and after N
        Cigar_len.append(cigar_len)

        # used to know where N is located so we can determine match before and after
        where_n = [x for x, y in enumerate(no_soft) if y[1] == gap_type]
        N_idx = int(where_n[0])  # find the position where the gap is
        
        # returns a tup (true/false) as to whether match  lengths before after n match
        # match is old school match (X + =) 
        # added a new filter flag (mismatch number) to allow us to filter by that instead
        align_pass = []
        align_pass.append((sum([int(no_soft[x][0]) for x in range(0, N_idx) if no_soft[x][1] == "X" or no_soft[x][1] == '=' or no_soft[x][1] == 'M']) - sum([int(no_soft[x][0]) for x in range(0, N_idx) if no_soft[x][1] == "D"])) >= min_align_length)
        align_pass.append((sum([int(no_soft[x][0]) for x in range(N_idx+1, cigar_len) if no_soft[x][1] == "X" or no_soft[x][1] == '=' or no_soft[x][1] == 'M']) - sum([int(no_soft[x][0]) for x in range(N_idx+1, cigar_len) if no_soft[x][1] == "D"])) >= min_align_length)
        tup.append(all(align_pass))

        # m1, m2, n will be used to calculate other things like size, fragment length, etc
        M1.append(sum([int(no_soft[x][0]) for x in range(0, N_idx)]))  # new 'match length' (before N
        M2.append(sum([int(no_soft[x][0]) for x in range(N_idx+1, cigar_len)])) # new "match" length after 'N'
        N.append(sum([int(x[0]) for x in no_soft if x[1] == 'N'])) # N size - if you do not filtter for 1N early - this will be misleading

    df['Cigar_len'] = Cigar_len
    df['total_indel'] = total_indel
    df['no_soft'] = no_soft_list
    df['tup'] = tup
    df['total_mismatch'] = total_mismatch
    df['N'] = N
    df['M1'] = M1
    df['M2'] = M2

    return(df)


def PrepOneDeletions(one_gap, ref, strain, gap_size, total_mismatch, align_lengt, gap_type = "N"):
    """
    INPUT: the estimated size using the CDS coordinates
    OUTPUT: Whether the estimated size is divisible by 3 and
    therefore maintaining open read frame
    """
    
    one_gap = pd.merge(one_gap, ref, on='segment', how='left')

    HEADER1 = ['name','segment','readname','gap','gap_start','gap_end','gap_size','estimated_length','segment_size',\
               'frame_flag','readflags','strain','cigar']

    # Iterate through rows to generate gap info for one and two gapped reads
    updated_df = []

    for index, row in one_gap.iterrows():
        # calculate the following useing the function MNM for the pattern
        # start, m1, n, m2
        gap, gap_start, gap_end = MNM(row['left_pos'], row['M1'], row['N'], row['M2'])
        frame_flag, est_length = EstSize(row['segment_size'], row['N'])
        
        updated_df.append([row['name'], row['segment'], row['readname'],
                     gap, gap_start, gap_end, row['N'], est_length,
                     row['segment_size'], frame_flag,
                     row['flags'], strain, row['cigar']])
            
    df = pd.DataFrame(updated_df, columns = HEADER1)

    return(df)


def PrepFreq(df, GapNumber):
    """
    INPUT: The candidate gap dataframe
    OUTPUT: Outputs dataframe to be written as a csv file
    """
    if GapNumber == 1:

        groups = ['name', 'segment', 'gap_start', 'gap_end', 'gap', 'gap_size',
                  'estimated_length', 'frame_flag']
        
        # count number of reads that match particular dvg
        dvg_freq = df.groupby(groups).size().reset_index(name="freq")
        # drop any duplicates to not double count
        dvg_freq = dvg_freq.drop_duplicates()

        return dvg_freq  # return dataframe to be used in grouping script


def GroupingOne(s, n, df, bandwidth=5):
    """
    INPUT: Freq. information for each sample
    OUTPUT: Grouping information for each sample/segment
    """
    # if a segment lacks DVG information - this will throw a warning
    # comment out if interested
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Grouping {0} 1N DVGs for {1}".format(s, n))
    dfs = df[(df.segment == s) & (df.name == n)].copy()  # subset dataframe for given segment

    if not dfs.empty:
        s = list(dfs['gap_start'])  # list of starts
        e = list(dfs['gap_end'])  # list of ends
        # zip and make into an array
        DVGs = np.array(list(zip(s, e)))  # zip will work left to right
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(DVGs)
        labels = ms.labels_
        labels_unique = list(np.unique(labels))
        n_clusters_ = len(labels_unique)

        #print("Number of estimated clusters : {0}".format(n_clusters_))

        grouping_info = list(zip(list(dfs['segment']), s, e, list(dfs['gap']),
            list(dfs['gap_size']),
            list(dfs['estimated_length']),
            list(dfs['freq']),
            list(labels)))

        g_df = pd.DataFrame(grouping_info,
            columns=['segment', 'gap_start', 'gap_end',
                    'gap', 'gap_size', 'estimated_length',
                    'freq', 'DVG_group'])

        # now we need to iterate through all of the groups and add in info
        for group in labels_unique:
            t = g_df[g_df.DVG_group == group]  # temp df of group data
            g_df.loc[g_df.DVG_group == group, 'Deletion'] = list(t[t.freq == t.freq.max()]['gap'])[0]
            g_df.loc[g_df.DVG_group == group, 'DeletionStart'] = list(t[t.freq == t.freq.max()]['gap_start'])[0]
            g_df.loc[g_df.DVG_group == group, 'DeletionEnd'] = list(t[t.freq == t.freq.max()]['gap_end'])[0]
            g_df.loc[g_df.DVG_group == group, 'GroupBoundaries'] = '{0}-{1}_{2}-{3}'.format(t['gap_start'].min(), t['gap_start'].max(), t['gap_end'].min(), t['gap_end'].max())
            g_df.loc[g_df.DVG_group == group, 'DeletionSize'] = list(t[t.freq == t.freq.max()]['gap_size'])[0]
            g_df.loc[g_df.DVG_group == group, 'EstimatedFragLength'] = list(t[t.freq == t.freq.max()]['estimated_length'])[0]

        return(g_df)
    else:
        print('no deletions {0} {1}'.format(s, n))


def GroupOneDeletionDVGs(indataframe, bandwidth, njob):
    """
    ADD IN PARAMETER FOR NUMBER OF THREADS/JOBS
    INPUT: freq. dataframe for all samples
    OUTPUT: group dataframe for all samples
    """
    df = indataframe
    SEGMENTS = list(set(list(df['segment'])))
    NAMES = list(set(list(df['name'])))
    masterDF = pd.concat(Parallel(n_jobs=njob)(delayed(GroupingOne)(s, n, df, bandwidth) for s in SEGMENTS for n in NAMES))
    print("Grouping 1N finished")
    return(masterDF)


def MergeReadsGroups(df, d1g, GapNumber):
    """
    INPUT: Read file info, group file, and number of deletions
    OUTPUT: df with read and group info, used to calc: rpkm, percentage, etc.
    """

    if GapNumber == 1:
        r, c = d1g.shape

        if r > 1:
            # merge the two dataframes
            d1g_m = pd.merge(df, d1g, how='left', on=['segment', 'gap',
                                                      'gap_start', 'gap_end',
                                                      'gap_size',
                                                      'estimated_length'])

            # drop any duplicates that were generated by merge
            d1g_m = d1g_m.drop_duplicates()

            # rename the freq column to keep for later use
            d1g_m = d1g_m.rename(columns={"freq": "FreqAcrossSamples"})

            # return to be used for rpkm, percentages, etc.
            return d1g_m

        else:
            print('No gap 1 files to merge')


def CountGroupedDVGs(df, GapNumber):
    """
    INPUT: Take in grouping/read merged dataframe from MergedReadGroups,
    count the new gap information for each sample

    OUTPUT: new dataframe that has the freq info for total number of gaps and
    Count information for each individual dvg type within the samples.
    This will then be used as input into the binomial check.
    Counts calculated with "New" gaps generated by grouping script
    """
    if GapNumber == 1 and df is not None:
        # count num reads / dvg
        dvg_freq = df.groupby(['name', 'segment',
                               'DeletionStart', 'DeletionEnd',
                               'Deletion']).size().reset_index(name="deletion_count")

        # count num reads /sample segment
        total_dvg = df.groupby(['name', 'segment']).size().reset_index(name="SegTotalDVG")

        # merge the dvg freq and total dvg counts into df
        dvg_c = pd.merge(dvg_freq, total_dvg, on=['name', 'segment'])

        # merge the dvg freq, total dvg counts with the full df with read info
        dvg_f = pd.merge(df, dvg_c, on=['name', 'segment', 'DeletionStart', 'DeletionEnd', 'Deletion'])

        # drop any duplicates generated during merging
        dvg_f.drop_duplicates()

        return dvg_f
    else: 
        print("No single gaps to count")


def ReduceDF(df, GapNumber):
    """
    INPUT: Read dataframe
    OUTPUT: A reduced dataframe, with info that we care about for figures
    """
    if GapNumber == 1:
        dropcols = ['readname', 'gap', 'gap_start', 'gap_end',
                    'gap_size', 'estimated_length', 'frame_flag',
                    'readflags', 'FreqAcrossSamples', 'cigar']

        df = df.drop(dropcols, axis=1)

        df = df.drop_duplicates()

        return df