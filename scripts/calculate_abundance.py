"""
python3 calculate_abundance.py --path ../../../read_depth/test_data --dvgs ../../../read_depth/test_data/H1N1.DVG.FINAL.OneGap.N5.Mis2.M28.G5.csv --outfile test.csv

This assumes that all data for DVGs are in one file 
# update to merge and keep nas...
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', required=True, help='path to read depth files')
parser.add_argument('--dvgs', '-d', required=True, help='path to read depth files')
parser.add_argument('--outfile', '-o', default="./rd_calculations.csv", help='directory path + outfile name')
args = parser.parse_args()

def seg_abundance(grouped):
    n = grouped['N'].sum()
    t = grouped['totalcount'].sum()
    l = (grouped['ntpos'].max())/1000
    if t > 0: 
        segment_abundance = round((n / t) / l, 4)
    else: 
        segment_abundance = np.nan
    
    return segment_abundance


def genome_abundance(full):
    n = full['N']
    t = full['totalcount']
    abundance = round((n.sum() / t.sum()), 6)
    return abundance


if __name__ == '__main__':
    filelist = [f for f in listdir(args.path) if f.endswith(".rd.csv") and isfile(join(args.path, f))]

    dvg_df = pd.read_csv(args.dvgs, sep=',', keep_default_na=False)

    updated_dvg = pd.DataFrame()
    all_data = pd.DataFrame()

    for f in filelist:
        df = pd.read_csv("{0}/{1}".format(args.path,f), sep=',', keep_default_na=False)  # read split read file 
        n = df['name'].unique()[0]
        print(n)
        seg_info = df.groupby(['name','segment'], as_index = False).apply(seg_abundance) #, include_groups = False)  # apply function to calculate proportion INCLUDE GROUPS IS A NEW THING
        seg_info.columns = ['name','segment', 'norm_del_proportion']  # adjust column names so we can pivot 
        seg_info = seg_info.pivot(index = 'name', columns=['segment'], values = 'norm_del_proportion')  # pivot wider so segments are column names
        seg_info = seg_info.reset_index()  # reset index so segment names are columns and not indx's
        
        genome_info = df.groupby(['name'], as_index = False).apply(genome_abundance)# , include_groups = False)  # on function - no grouping/no normalization INCLUDE GROUPS IS A NEW THING
        genome_info.columns = ['name', 'total_del_proportion']
        
        final_df = pd.merge(seg_info, genome_info, on='name', how='left')
        all_data = pd.concat([all_data, final_df], ignore_index=True)
        temp_df = dvg_df[(dvg_df.name == n)].copy() 


        temp2 = pd.merge(temp_df, df[['name','segment','ntpos','totalcount']], left_on = ['name','segment','DeletionStart'], right_on = ['name','segment','ntpos'], how = 'left')
        temp2 = pd.merge(temp2, df[['name','segment','ntpos','totalcount']], left_on = ['name','segment','DeletionEnd'], right_on = ['name','segment','ntpos'], how = 'left')
        updated_dvg = pd.concat([updated_dvg, temp2], ignore_index=True)


    all_data.to_csv(args.outfile, index=False)

    updated_dvg.drop(['ntpos_x', 'ntpos_y'], axis=1, inplace=True)  # remove the ntpos cols which are there bc they are named diff than DeletionStart/DeletionEnd
    updated_dvg.rename(columns={'totalcount_x': 'totalcount_start', 'totalcount_y': 'totalcount_end'}, inplace=True)  # rename the totalcount col - may change given timo
    updated_dvg['estimated_freq1'] = updated_dvg['deletion_count'] / updated_dvg['totalcount_start']  # calculate estimated freq of start region
    updated_dvg['estimated_freq2'] = updated_dvg['deletion_count'] / updated_dvg['totalcount_end']  # calculate estimated freq of end region

    updated_dvg.to_csv(args.dvgs, index=False)  # write to file 
