"""
Written by: Kate Johnson

Made to: 
Quantify the total number of reads spanning a region - 
both MATCHING and GAPPED (aka 'ghost' reads)

Input: 
1. Gapped-read alignment ("N" in CIGAR string)
2. Reference
3. Qual cutoff (default = 30)
4. Strain/virus name
5. Output directory

Output: 
1. Read depth counts across the genome for the input bam. CSV format. 

Example run: 
python3 pull_ghost_reads.py \
    --ref ../../../read_depth/reference/Cal07.09.cds.fasta \
    --infile ../../../read_depth/example_bam/S10_R1_D07_FD.H1N1.rmd.bbmap.bam  \
    --qual_cutoff 30 \
    --strain H1N1 \
    --outputdir ./ \
    --njobs 4

"""

import sys,os,glob
import numpy as np
from scipy.stats.distributions import binom
import pysam
import pandas as pd
import operator
import argparse
from joblib import Parallel, delayed
import time, math

start = time.time()

parser = argparse.ArgumentParser() #argparse always need this first statment and then the following add arguments below
parser.add_argument('--ref','-r',required=True,help='give full path to reference file. Needs full path if not local!') 
parser.add_argument('--infile','-i',help='input single bamfile. Needs full path if not local!')
parser.add_argument('--qual_cutoff','-q',type=int,default=30,help='phred quality cutoff (default is at 30)')
parser.add_argument('--strain','-T',type=str,default='strain',help='need strain')
parser.add_argument('--outputdir','-o',default='./',help='output directory')
parser.add_argument('--njobs','-n',default=2,type=int,help='number of jobs to parallelize')

args = parser.parse_args()


def ensure_dir(f):
    """
    Confirm output directory exists
    If it doesn't - make it
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def analyzer(isReverse,updatedOut,qual_cutoff,REVERSE_DICT,FORWARD_DICT,CONSENSUS_DICT,INSERTION_DICT): 
    """
    Input: 
    """
    cig = updatedOut[0]
    seq = updatedOut[1]
    ntpos = updatedOut[2]
    qual = updatedOut[3]

    tempinsdict = {}
    if isReverse:
        for c,nt,pos,q in zip(cig,seq,ntpos,qual): 
            if pos == None and (c == 'S' or c == 'D' or c == 'N'): # added 'N'
                pass
            else:
                if c == 'I':
                    if pos in tempinsdict:
                        tempinsdict[pos].append(nt)
                    else:
                        tempinsdict[pos] = [nt]
                elif q == '-' or q >= qual_cutoff: #quality passes threshold
                    if nt in REVERSE_DICT[pos]: #position populated
                        REVERSE_DICT[pos][nt] = REVERSE_DICT[pos][nt] + 1
                    else:
                        REVERSE_DICT[pos][nt] = 1
                    if nt in CONSENSUS_DICT[pos]:
                        CONSENSUS_DICT[pos][nt] = CONSENSUS_DICT[pos][nt] + 1
                    else:
                        CONSENSUS_DICT[pos][nt] = 1
        if tempinsdict:
            for ntpos in tempinsdict:
                fullnt = ''.join(tempinsdict[ntpos])
                if ntpos in INSERTION_DICT:
                    pass
                else:
                    INSERTION_DICT[ntpos] = {}

                if fullnt in INSERTION_DICT[ntpos]:
                    INSERTION_DICT[ntpos][fullnt] = INSERTION_DICT[ntpos][fullnt] + 1
                else:
                    INSERTION_DICT[ntpos][fullnt] = 1
    else:
        #FORWARD_DICT
        for c,nt,pos,q in zip(cig,seq,ntpos,qual):
            if pos == None and (c == 'S' or c == 'D' or c == 'N'):  # added 'N'
                pass

            else:
                if c == 'I': #insertion
                    if pos in tempinsdict:
                        tempinsdict[pos].append(nt)
                    else:
                        tempinsdict[pos] = [nt]
                elif q == '-' or q >= qual_cutoff:
                    if nt in FORWARD_DICT[pos]:
                        FORWARD_DICT[pos][nt] = FORWARD_DICT[pos][nt] + 1
                    else:
                        FORWARD_DICT[pos][nt] = 1

                    if nt in CONSENSUS_DICT[pos]:
                        CONSENSUS_DICT[pos][nt] = CONSENSUS_DICT[pos][nt] + 1
                    else:
                        CONSENSUS_DICT[pos][nt] = 1
        if tempinsdict:
            for ntpos in tempinsdict:
                fullnt = ''.join(tempinsdict[ntpos])
                if ntpos in INSERTION_DICT:
                    pass
                else:
                    INSERTION_DICT[ntpos] = {}

                if fullnt in INSERTION_DICT[ntpos]:
                    INSERTION_DICT[ntpos][fullnt] = INSERTION_DICT[ntpos][fullnt] + 1
                else:
                    INSERTION_DICT[ntpos][fullnt] = 1


def seqUpdater(cigartuple,read,readidx,readq,qual_cutoff): 
    """
    This function takes in the cigar tuple and expands it to be
    the length of the read. See pysam doc for what each # 
    in a cigar tuple represents.
    """
    updatedcigarseq = []
    updatedseq = []
    updatedidx = []
    updatedreadq = []
    idxctr = 0
    for CIGIDX, (identifier,length) in enumerate(cigartuple):
        if identifier == 0 or identifier == 7:  # match
            updatedcigarseq.extend('M'*length)
            for i in range(length):
                updatedseq.append(read[idxctr])
                updatedreadq.append(readq[idxctr])
                idxctr+=1

        elif identifier == 1:  # insertion
            updatedcigarseq.extend('I'*length)
            for i in range(length):
                updatedseq.append(read[idxctr]) 
                updatedreadq.append(readq[idxctr])
                idxctr+=1

        elif identifier == 8:  # mismatch
            updatedcigarseq.extend('X'*length)
            for i in range(length):
                updatedseq.append(read[idxctr]) 
                updatedreadq.append(readq[idxctr])
                idxctr+=1

        elif identifier == 2:  # deletion
            updatedcigarseq.extend('D'*length) 
            for i in range(length):
                updatedseq.append('-')
                updatedreadq.append('-')

        elif identifier == 4:  # Soft clip
            updatedcigarseq.extend('S'*length)
            for i in range(length):
                updatedseq.append(read[idxctr])
                updatedreadq.append(readq[idxctr])
                idxctr+=1

        elif identifier == 3:  # gap
            updatedcigarseq.extend('N'*length)
            for i in range(length):
                updatedseq.append("N")
                updatedreadq.append('-')
            
        elif identifier == 5: # hard clip
            pass
        else:
            sys.exit("cigartuple: invalid number encountered! %s in %s " % (identifier,cigartuple))
    
    idxctr = 0
    last_delnt = 0
    last_insnt = 0
    last_skip = 0
    
    for i,j,q in zip(updatedcigarseq,updatedseq,updatedreadq):
        if i == 'D' or i == 'N':  # deletions and gaps will not be apart of the total read length - skip
            if(last_delnt == None):
                pass
            else:
                last_delnt+=1
            updatedidx.append(last_delnt)
        elif i == 'I':
            updatedidx.append(last_insnt)
            idxctr+=1
        else:
            updatedidx.append(readidx[idxctr])
            last_delnt = readidx[idxctr]
            last_insnt = readidx[idxctr]
            idxctr+=1 
    
    assert len(updatedcigarseq) == len(updatedseq) == len(updatedidx) == len(updatedreadq)  # check that they are all the same length assert function checks if True

    return (updatedcigarseq,updatedseq,updatedidx,updatedreadq)


def read_fasta(fp):
    """
    Read input fasta
    """
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
    """
    Open, read, and return reference fasta as dictionary
    """
    segdict = {}
    with open(filename) as fp:
        for name, seq in read_fasta(fp):
            segdict[name[1:]] = seq
    return segdict


def printer(consensus_dict, sampname, SEGMENT, ntpos): 
    """
    Adjust the data to have accurate counts for each nt pos in the genome.
    """
    tempd = consensus_dict[ntpos]
    ntpos = ntpos + 1  # change from 0-index to 1-index
    updated_dict = {}
    updated_dict[ntpos] = {}
        
    A_nt = 0
    C_nt = 0
    G_nt = 0
    T_nt = 0
    del_nt = 0
    gap_nt = 0

    # update the total counts
    if 'A' in tempd:
        A_nt = tempd['A']
    if 'C' in tempd:
        C_nt = tempd['C']
    if 'G' in tempd:
        G_nt = tempd['G']
    if 'T' in tempd:
        T_nt = tempd['T']
    if '-' in tempd:
        del_nt = tempd['-']
    if 'N' in tempd: 
        gap_nt = tempd['N']
    
    totalcount = sum(tempd.values())

    
    # update the dictionary that is returned
    updated_dict[ntpos]['name'] = sampname
    updated_dict[ntpos]['segment'] = SEGMENT
    updated_dict[ntpos]['ntpos'] = ntpos
    #updated_dict[ntpos]['A'] = A_nt
    #updated_dict[ntpos]['C'] = C_nt
    #updated_dict[ntpos]['G'] = G_nt
    #updated_dict[ntpos]['T'] = T_nt
    #updated_dict[ntpos]['-'] = del_nt
    updated_dict[ntpos]['N'] = gap_nt
    updated_dict[ntpos]['totalcount'] = totalcount
    updated_dict[ntpos]['mapped_totalcount'] = totalcount - gap_nt

    return(pd.DataFrame.from_dict(updated_dict, orient='index'))
    #return(updated_dict)


def MainFunction(SEGMENT, REF_DICT, bamname, QUAL_CUTOFF=30, njobs=4): 
    SAMFILE = pysam.AlignmentFile(bamname, "rb") # "rb" = 'read' 'bam'

    SEGLEN = len(REF_DICT[SEGMENT]) # length of ref seg

    FORWARD_DICT = {}
    REVERSE_DICT = {}
    INSERTION_DICT = {}
    CONSENSUS_DICT = {}

    for idx in range(SEGLEN):
        FORWARD_DICT[idx] = {}
        REVERSE_DICT[idx] = {}
        CONSENSUS_DICT[idx] = {}
    counter = 0

    for read in SAMFILE.fetch(SEGMENT):
        if read.is_unmapped:
            pass
        else:
            counter += 1
            cigartup= read.cigartuples # each read take cigar string, indx position, quality, forward/reverse
            unfiltreadidx = read.get_reference_positions(full_length=True)  # where the read aligns 
            unfiltread = list(read.query_sequence)  # read sequence - need for AGTC-N counts
            unfiltreadqual = read.query_qualities  # qual
            isReverse = read.is_reverse  # tells if forward or reverse read
            analyzer(isReverse, seqUpdater(cigartup,unfiltread,unfiltreadidx,unfiltreadqual, QUAL_CUTOFF),
            QUAL_CUTOFF,REVERSE_DICT,FORWARD_DICT,CONSENSUS_DICT,INSERTION_DICT)
    
    SEGOUT = Parallel(n_jobs=njobs)(delayed(printer)(CONSENSUS_DICT, SAMPLENAME, SEGMENT, ntpos) for ntpos in sorted(CONSENSUS_DICT.keys()))
    
    SAMFILE.close()

    return(pd.concat(SEGOUT))


if __name__ == '__main__':
    FULLVARLIST_DIR = args.outputdir 
    ensure_dir(FULLVARLIST_DIR)

    # input values: 
    QUAL_CUTOFF = args.qual_cutoff
    REF_DICT = open_fasta(args.ref)  # reference dictionary
    SAMPLENAME = args.infile.split('/')[-1].split('.')[0]  # makes same assumptions of name as 'timo'
    STRAIN = args.strain  # strain name
    OUTFILE = "%s/%s.%s.q%s.rd.csv" % (FULLVARLIST_DIR,SAMPLENAME,STRAIN,str(QUAL_CUTOFF))  # writing the file name for the snplist

    for f in REF_DICT:
        print(f)

    pd.concat(Parallel(n_jobs=int(args.njobs))(delayed(MainFunction)(SEGMENT, REF_DICT, args.infile, QUAL_CUTOFF, args.njobs) for SEGMENT in REF_DICT), ignore_index=True).to_csv(OUTFILE, header=True, index=False)


    end = time.time()

    print('{:.4f} s'.format(end-start))
