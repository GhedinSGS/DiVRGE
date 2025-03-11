"""
Written by: Kate Johnson
python3 calculate_abundance.py \
    --path ../read_depth \
    --dvgs ../DVG/SARS2.DVG.FINAL.OneGap.N5.Mis2.M28.G5.csv \
    --coverage_filt 5 \
    --outfile updated_calcs.csv \
    --newfilepath ../updated_files

This assumes that all data for DVGs are in one file 
"""
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import argparse
import numpy as np
import warnings
from tqdm import tqdm
import time
from warnings import simplefilter


warnings.filterwarnings("ignore", category=RuntimeWarning)  # adding this because log(1) = 0 and thenw e divide for Hmax so it outputs: divide by zero encountered in log
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)  # this is a warning that occurs when a large # of deletions are in the sample


parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', required=True, help='path to read depth files')
parser.add_argument('--dvgs', '-d', required=True, help='path to read depth files')
parser.add_argument('--coverage_filter', '-c', default=1, type = int, help='Provide a min. number of reads that must have mapped to position for diversity calcs (default:1)')
parser.add_argument('--outfile', '-o', default="./rd_calculations.csv", help='directory path + outfile name')
parser.add_argument('--newfilepath', '-n', default="./shannon_site", help='directory path for updated rd files')
args = parser.parse_args()

def create_path(newfilepath):
    if not os.path.exists(newfilepath):
        os.makedirs(newfilepath)

create_path(args.newfilepath)

def site_shannon(row, columns_to_apply): 
    
    return(abs(sum([(row[col] * np.log(row[col])) for col in columns_to_apply if row[col]>0])))  # if DVG column is 0, then ignore. Can't calc shannon - no diversity


def total_shannon(grouped): 
    d = {}
    index_list = ['shannon_norm_length','nHmax','nH',"Hmax","H",'max_probability']
    d['shannon_norm_length'] = (grouped.shape[0])  # calculate sites that should normalize to
    d["nHmax"] =  (grouped['Hmax_site'].sum()) /(grouped.shape[0]) # sum max normalized shannon across all sites
    d["nH"] =  (grouped['Hsite'].sum()) /(grouped.shape[0])
    d["Hmax"] =  (grouped['Hmax_site'].sum()) 
    d["H"] =  (grouped['Hsite'].sum())
    d['max_probability'] = grouped['deletion_probability_site'].max()  # nt site with the max probability of a deletion 
    
    return pd.Series(d, index=index_list)


def update_values(dvgs, read_depth, name, coverage_filter = 1):

    for _, deletion in dvgs.iterrows():
        """
        For every deletion that passes our quality checks:
        - pull the deletion info
        - pull the # of reads for that deletion
        - generate a new colum on the read count df with DVG label and read count
        - Add zero's at positions where deletion is not at
        """
        chrom = deletion['segment']  # segment
        start = deletion['DeletionStart']  # deletion start - used to add to rd info
        end = deletion['DeletionEnd']  # deletion end
        value = deletion['deletion_count']  # number of reads spanning deletion
        deletion = "{0}_{1}".format(deletion['segment'], deletion['Deletion'])  # name of deletion for column segment + coordinates
        
        # S = richness of deletions at position
        if 'S' not in read_depth.columns:  # If this is the first instance of adding the S column to the df
            read_depth['S'] = 0  # add and fill with 0's

        # for every deletion that spans start and end region - add 1:
        read_depth.loc[(read_depth['segment'] == chrom) & (read_depth['ntpos'] >= start) & (read_depth['ntpos'] <= end), "S"] += 1

        # add the number of DVG reads that span the region we are looking at:
        read_depth.loc[(read_depth['segment'] == chrom) & (read_depth['ntpos'] >= start) & (read_depth['ntpos'] <= end), deletion] = value
        
        
    read_depth.iloc[:, 6:] = read_depth.iloc[:, 6:].fillna(0)  # for all of the new cols we just added (S + DVGs) fill na's with 0's
    columns_to_sum = read_depth.columns[7:]  # pull all DVG columns - will be 0 if not within seg
    read_depth['no_deletion_probability'] = read_depth['totalcount'] - (read_depth[columns_to_sum].sum(axis=1))  # determine the number of reads with no deletion at given site
    # no_deletions
    columns_to_divide = read_depth.columns[7:]     # Divide selected columns by the divisor column
    read_depth[columns_to_divide] = read_depth[columns_to_divide].div(read_depth['totalcount'], axis=0)  # divide everything by total count at each position
    read_depth.iloc[:, 7:] = read_depth.iloc[:, 7:].fillna(0)  # no longer including 'S' here - if na - then no mapping - 0

    read_depth['Hsite'] = read_depth[columns_to_divide].apply(lambda row: site_shannon(row, columns_to_divide), axis=1)  # caculate shan entropy at each site

    # prepping for max normalization
    read_depth.loc[(read_depth['no_deletion_probability'] > 0), "S"] += 1  # add 1 to the diversity if we have no deletions - if no deletions exist, and totalcount >0 then no deletions will equal 1
    read_depth['Hmax_site'] = read_depth['Hsite'] / np.log(read_depth['S'])  # normalized by max possible entropy normalization max ent = log2(# of deletions). if 1 then it is highly entropic? ha - if total count 0
    read_depth['deletion_probability_site'] = 1 - read_depth['no_deletion_probability']  # calc the probability of a site is deleted 

    temp = read_depth[(read_depth.totalcount >= coverage_filter)].copy()  # making this temp so we output updated files 


    if temp.shape[0] > 0:    # after filtering by read depth - do the following if the df is not empty
        # per segment calcs:
        d1 = temp.groupby(['name','segment'], as_index = False).apply(total_shannon)  # make ecology calcs for each segment and sample
        # per genome calcs: 
        d2 = temp.groupby(['name'], as_index = False).apply(total_shannon)  # make ecology calcs for each segment and sample
        d2['segment'] = 'genome'  # change segment to genome so we can rbind
        # cat into one df to write out
        d1 = pd.concat([d1, d2], ignore_index=True,axis = 0)
    
    elif temp.shape[0] == 0:  # if the read depth filter made it so there are no sites in the genome - then output an empty df
        #print(name)
        index_list = ['nHmax','nH',"Hmax","H",'max_probability']
        empty_dict = {key: [np.nan] for key in index_list}
        empty_dict['name'] = name
        empty_dict['segment'] = 'genome'
        empty_dict['shannon_norm_length'] = temp.shape[0]
        d1 = pd.DataFrame(empty_dict)
        #print(d1)

    
    
    return d1, read_depth[['name', 'segment', 'ntpos', "N", 'totalcount',"mapped_totalcount", "S", "no_deletion_probability", "Hsite", "Hmax_site", "deletion_probability_site"]].copy()

def seg_abundance(grouped):
    d = {}
    d['genome_length'] = (grouped['ntpos'].max())/1000
    if grouped['totalcount'].sum() > 0: ## if we have area: 
        d['area'] = round(((grouped['N'].sum()) /  (grouped['totalcount'].sum())) / ((grouped['ntpos'].max()) / 1000), 4)
    else: 
        d['area'] = np.nan
    
    return pd.Series(d, index=['genome_length', 'area'])


def genome_abundance(full):
    d = {}  
    d['genome_length'] = (full.shape[0]) / 1000
    
    if full['totalcount'].sum() > 0: 
        d['area'] = (full['N'].sum()) /  (full['totalcount'].sum())  # not normalized
    else: 
        d['area'] = np.nan
    d['segment'] = 'genome'

    return pd.Series(d, index=['genome_length', 'area','segment'])

def eco_calcs(grouped): 
    d = {}
    #d["H"] =  -(grouped['Hsite'].sum()) # shannon
    d["BP"] = grouped['estimated_freq'].max()  # berger-parker dominance
    d['S']  = grouped['estimated_freq'].count()  # richness
    d['mean_freq'] = grouped['estimated_freq'].mean()
    d['sd_freq'] = grouped['estimated_freq'].std()
    d['deletion_count'] = grouped['deletion_count'].sum()

    return pd.Series(d, index=['BP', 'S','mean_freq','sd_freq','deletion_count'])

def update_sample(sample):
    time.sleep(0.1)  # Simulate a delay in updating the sample

if __name__ == '__main__':
    filelist = [f for f in listdir(args.path) if f.endswith(".rd.csv") and isfile(join(args.path, f))]

    dvg_df = pd.read_csv(args.dvgs, sep=',', keep_default_na=False)

    updated_dvg = pd.DataFrame()  # adding in frequency info to the DVGs from the read depth files
    area_data = pd.DataFrame()  # 'area' data in which we calculate the probability of pulling a deleted site
    shannon_data = pd.DataFrame()  # shannon data for each sample and segment

    with tqdm(total=len(filelist), desc="Updating samples") as pbar:

        for f in filelist:

            update_sample(f)
            pbar.update(1)  # Update the progress bar

            df = pd.read_csv("{0}/{1}".format(args.path,f), sep=',', keep_default_na=False)  # read in read depth file
            n = df['name'].unique()[0]  # pull out name for other things
            genome_size =  (df.shape[0])
            #print(n)  # prin the name

            # calculate segment specific areas: 
            seg_info = df.groupby(['name','segment'], as_index = False).apply(seg_abundance)#, include_groups = False)  # apply function to calculate proportion INCLUDE GROUPS IS A NEW THING

            # calculate genome specific areas:
            genome_info = df.groupby(['name'], as_index = False).apply(genome_abundance)# , include_groups = False)  # on function - no grouping/no normalization INCLUDE GROUPS IS A NEW THING
            
            final_df = pd.concat([seg_info, genome_info], ignore_index=True,axis = 0)  # concat seg and genome info
            
            area_data = pd.concat([area_data, final_df], ignore_index=True)  # append each samples calculations to the major file

            temp_df = dvg_df[(dvg_df.name == n)].copy() # pull the dvg info for the sample to add to 

            if temp_df.shape[0] > 0: # if the sample has dvgs:
                s , r = update_values(temp_df, df, n, args.coverage_filter) # dvgs for sample, read count data for sample
                shannon_data = pd.concat([shannon_data, s], ignore_index=True)  # shannon info for sample 
                r.to_csv("{0}/updated_{1}".format(args.newfilepath, f), index=False)


            # add in total count information to calculate est. freqs for both start and end positions:
            temp2 = pd.merge(temp_df, df[['name','segment','ntpos','totalcount']], left_on = ['name','segment','DeletionStart'], right_on = ['name','segment','ntpos'], how = 'left')
            temp2 = pd.merge(temp2, df[['name','segment','ntpos','totalcount']], left_on = ['name','segment','DeletionEnd'], right_on = ['name','segment','ntpos'], how = 'left')
            temp2['genome_size'] = genome_size  # add genome size so we can use for normalization in our calcs
            updated_dvg = pd.concat([updated_dvg, temp2], ignore_index=True)  # add it to the updated DVG file

    # make row level calcs like estimated freq, site H, and total deletion counts after filtering 
    updated_dvg.drop(['ntpos_x', 'ntpos_y','SegTotalDVG'], axis=1, inplace=True)  # remove the ntpos cols which are there bc they are named diff than DeletionStart/DeletionEnd
    updated_dvg.rename(columns={'totalcount_x': 'totalcount_start', 'totalcount_y': 'totalcount_end'}, inplace=True)  # rename the totalcount col - may change given timo
    updated_dvg['estimated_freq1'] = updated_dvg['deletion_count'] / updated_dvg['totalcount_start']  # calculate estimated freq of start region
    updated_dvg['estimated_freq2'] = updated_dvg['deletion_count'] / updated_dvg['totalcount_end']  # calculate estimated freq of end region
    updated_dvg['estimated_freq'] =  (updated_dvg['estimated_freq1'] +  updated_dvg['estimated_freq2']) / 2  # calculate the avg est. freqs for start and end
    updated_dvg['total_deletion_count'] = updated_dvg.groupby(['name','segment'])['deletion_count'].transform('sum')  # recalculate the total deletion count given that we have filtered
    
    # genome specific calculations
    d2 = updated_dvg.groupby(['name','genome_size'], as_index = False).apply(eco_calcs)  # make ecology calcs across the genome for samples
    d2['S_norm'] = d2['S'] / (d2['genome_size'] / 1000)  # normalize richness by kb
    d2['segment'] = 'genome'  # add a 'segment' column to allow for rbinding
    #d2['genome_length'] = d2['genome_size'] / 1000  # Output the way in which we normalized data
    d2.drop(['genome_size'], axis=1, inplace=True)  # Drop - will be a duplicate in later merges
   
    # segment specific calculations:
    d1 = updated_dvg.groupby(['name','segment','segment_size'], as_index = False).apply(eco_calcs)  # make ecology calcs for each segment and sample
    d1['S_norm'] = d1['S'] / (d1['segment_size'] / 1000)
    #d1['genome_length'] = (d1['segment_size']) / 1000
    d1.drop(['segment_size'], axis=1, inplace=True) 

    d1 = pd.concat([d1, d2], ignore_index=True,axis = 0)  # concat the genome and segment data. 
    d1 = pd.merge(d1, shannon_data, left_on = ['name','segment'], right_on = ['name','segment'], how = 'outer')  # add shannon data 

    # d1 = diversity calculations, area_data = area information
    d1 = pd.merge(d1, area_data, left_on = ['name','segment'], right_on = ['name','segment'], how = 'outer')

    d1.to_csv(args.outfile, index=False)

    updated_dvg.to_csv(args.dvgs, index=False)  # write to file  UNDO WHEN FINISHED
