<h1 align="center">DiVRGE<br> <em>Deletions in Viral Genomes Extractor</em> </h1>

<h2 align="center"> Running DiVRGE </h2>

DiVRGE has been tested using alignment outputs generated by BBmap [1] and STAR [2] and will work on any alignment file that uses 'N' to represent a split-read in the CIGAR string [3]. We highly recommend using BBmap as it is an intuitive aligner to work with and is more accurate at aligning split-reads from larger and more complex viral genomes, such as SARS-CoV-2. 

We have integrated DiVRGE into a snakemake workflow pipeline. For more information on how to use snakemake or how to integrate snakemake with computing clusters please see their website [4].

<h2 align="center"> Quick Start </h2><br>
1. Clone repo<br>
2. Setup Python Virtual Environment<br>
3. Adjust config file<br>
4. Run snakemake<br>

<h2 align="center"> Setup and Requirements </h2> <br>


<p align="center">
After cloning the DiVRGE repo, check that all required programs and packages listed below are installed.<br>
</p>

### Alignment + DiVRGE requirements:
```
snakemake/6.12.3
trimmomatic/0.36    
bbtools/38.91   
samtools/intel/1.11   
picard/2.23.8   
subread/intel/2.0.1   
python/intel/3.8.6    
```

### DiVRGE Python3 packages (in requirements.txt):
```
biopython==1.81
joblib==1.2.0
numpy==1.21.5
pandas==2.0.1
pysam==0.20.0
python-dateutil==2.8.2
pytz==2023.3
scikit-learn==1.2.1
scipy==1.10.0
six==1.16.0
sklearn==0.0.post1
threadpoolctl==3.1.0
tzdata==2023.3
```
### Setting up virtual environment named 'divrge' and installing packages: 
```
>cd $HOME  # select direct. to create environment
>python3 -m venv divrge  # create new python environment
>source divrge/bin/activate  # activate env
>python3 -m pip install -r requirements.txt  # install DiVRGE requirements
```

### Check that necessary Python packages are installed in virtual env. using *pip list --local*:<br> 
```
(divrge) [user ~]$ pip list --local
Package         Version
--------------- ---------
biopython       1.81
joblib          1.2.0
numpy           1.21.5
pandas          2.0.1
pip             21.1.1
pysam           0.20.0
python-dateutil 2.8.2
pytz            2023.3
scikit-learn    1.2.1
scipy           1.10.0
setuptools      56.0.0
six             1.16.0
sklearn         0.0.post1
threadpoolctl   3.1.0
tzdata          2023.3
```

### Deactivate env when finished:<br>
```
>deactivate 
```

<h2 align="center"> Adjusting config and running </h2><br>

## snakefile_main_align (recommended): Align + SNV + DiVRGE <br>

This workflow trims (trimmomatic) and aligns (BBmap and BWA) paired-end fastq data. BWA alignments are used as input into timo [5] to call single-nucleotide variants while BBmap alignment files are used to pull split-read information and call deletions with DiVRGE. This workflow will generate subdirectories in the defined working directory to organize and store all outputs.

- Download config_align.yaml file
- Adjust the config file using your inputs (see below for details)
- Do not rename the file!
- This workflow works with paired-end seq. data (must adjust for single-end data)
- Make sure you have indexed your reference file using BWA before starting.

```
bwa index reference.fasta
```

- run **snakefile_main_align**. Must include '--use-envmodules' flag. See below for an example of a simple snakemake run with 2 cores:

```
>sinteractive --cpus-per-task=4 --mem=8GB
>module load snakemake/6.12.3
>snakemake -c2 --use-envmodules -s snakefile_main_align
```

example run across multiple SLURM Jobs (see https://snakemake.readthedocs.io/en/stable/executing/cluster.html for cluster integration): 
```
>module load snakemake/6.12.3
>snakemake -j2 -c4 --use-envmodules --groups align=4 --cluster "sbatch --cpus-per-task=4 --mem=24GB --time=1:00:00" -s snakefile_main_align
```

#### config_align.yaml: 
```
---
params:  # general and alignment parameters
  ref: path and ref file used for alignmnet ("/path/to/reference/h9n2/H9N2_HK_1999_CDS.fasta") 
  reads: path to raw fastq data ("/path/to/rawfiles/")
  begin1: characters found before sample id for pair 1 reads (ex: "n01", "000000-ZJFPQ", "HT5MLBGXL_n01_")
  ending1: characters at end of read name, after the sample id for pair 1 reads (ex: "_read1.fq.gz", "fastq.gz", "fq.gz", "n01.fq.gz")
  begin2: characters found before sample id for pair 2 reads (ex: "n02", "000000-ZJFPQ", "HT5MLBGXL_n02_")
  ending2: characters at end of read name, after the sample id for pair 2 reads (ex: "_read2.fq.gz", "fastq.gz", "fq.gz", "n02.fq.gz")
  output: path to directory where files should be saved. MAKE SURE '/' PRESENT AT END. ("/path/to/save/direct/")
  maxindel: the max deletion size expected. BBmap input. (ex: "3000")
  minindel: the min deletion size allowed. Warning: deletion sizes <5nt will decrease accuracy of alignment. BBmap input. (ex: "5")
  adapters: path and filename to adapters used for trimming (ex: "/share/apps/trimmomatic/0.36/adapters/NexteraPE-PE.fa")
  strain: strain, virus, subtype, lineage, etc. for naming purposes (ex: "H1N1","H3N2","SARS2","COVID")
  gigs: the amt. of mem needed for alignment with BBmap. Lg. files will require more memory (ex: "4G", "8G", "16G") 
  gtf_dir: filepath to where gtf file should be saved (ex: "/path/to/reference/h9n2")
  script_dir: directory where 'timo' and 'divrge' scripts are located (ex: "/scratch/kej310/testit/scripts/")
  venv_path: path to python environment with installed packages (see above for more info on installing packages) ("/path/to/divrge/bin/activate")

timo_params:  # timo specific parameters
  segments: list of 'segments' or 'CHROM' in the reference file. Include the brackets. (["H9N2_PB2","H9N2_PB1","H9N2_PA","H9N2_HA","H9N2_NP","H9N2_MP","H9N2_NS"])
  freq_cutoff: min freq of minor SNV required (default: 0.001) (ex: "0.001")
  cov_cutoff: min cov. required for SNV (default: 1) (ex: "1")

divrge_param: # divrge specific parameters
  mismatch: number of mismatches allowed in read alignment (default: "2")
  gap_size: min. size of deletion (default: "5")
  align_length: min. alignment length before and after deletion (default: "28")
  group_size: grouping size (default: "5")
  ncores: number of cores used during grouping (default: "2")
  ninput: number of samples input (default: "1")

mod_params: # modules to load
  subread_mod: "subread/intel/2.0.1"
  samtools_mod: "samtools/intel/1.14"
  python_mod: "python/intel/3.8.6"
  trim_mod: "trimmomatic/0.36"
  trim_jar: "$TRIMMOMATIC_JAR" (path to trimmomatic jar file)
  bbtools: "bbtools/38.91"
  bwa: "bwa/intel/0.7.17"
  picard_mod: "picard/2.23.8"
  picard_jar: "$PICARD_JAR" (path to picard jar file)

```

## snakefile_divrge_only includes: DiVRGE only (individual samples)<br>
If you have already aligned your data using BBmap or a similar split-read aligner, you can run DiVRGE specific rules with 'snakefile_divrge_only'.<br>

- Download config_divrge.yaml file.
- Adjust the config file using your inputs.
- Do not rename the file!
- This workflow works on PREVIOUSLY ALIGNED data.
- run **snakefile_divrge_only**. See below for a simple example of a snakemake run using 2 cores. Must use the '--use-envmodules' flag: 

```
>module load snakemake/6.12.3
>snakemake -c2 --use-envmodules -s snakefile_divrge_only
```

#### config_divrge.yaml
Most parameters overlap with those outlined above in 'config_align.yaml', those that do not overlap include: 
```
params:
  bamdir: full directory path to alignments you want to use as input into DiVRGE specific scripts (ex: "/path/to/alignment/files/bamfiles/rmdups")
  bam_ending: string that follows the sample name (ex: ".H9N2.rmd.bbmap.bam", ".bam", ".sorted.bam")
```

**Not sure if your alignments pass DiVRGE requirements?**
Split-read aligners differ in their sam formats. If not specified in the aligners user manual, an easy check includes: 
```

```
<h2 align="center"> Output </h2><br>

If running the recommended **snakemake_main_align** pipeline: 
- **bamfiles**: alignments
- **trimmed**: trimmed fastqs
- **FILES/fullvarlist**: the timo snplist outputs. Timo "snplist" files will contain nucleotide information across all positions in the genome (see GhedinLab/timo for more details).
- metric: metric files
- **DVG**: deletion files. Each sample will have individual deletion csv files (labeled: "SAMPLE.DVG.STRAIN.FINAL.OneGap"). All samples that were aligned together will also be concatenated into one file named "STRAIN.DVG.FINAL.OneGap").
- **N_files**: split read information used for DiVRGE







<h2 align="center"> What can you do with the output? </h2><br>

- Calculate deletion richness<br>
- Identify hotspots of deletion start/end coordinates<br>
- Calculate relative frequency using coverage information<br>

<h2 align="center"> Reference </h2><br>



<h2 align="center"> Links </h2><br>
1. https://sourceforge.net/projects/bbmap/<br>
2. https://github.com/alexdobin/STAR<br>
3. https://samtools.github.io/hts-specs/SAMv1.pdf<br>
4. https://snakemake.github.io/<br>
5. https://github.com/GhedinLab/timo/blob/main/timo.v3.py<br>
