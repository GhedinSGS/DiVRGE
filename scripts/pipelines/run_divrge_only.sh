#!/bin/bash

module load snakemake/7.32.4

snakemake --jobs 10214 \
    --use-envmodules \
    --group-components divrge=455 \
    --cluster "sbatch --cpus-per-task=4 --mem=2G --time=24:00:00" \
    --max-jobs-per-second 1 --max-status-checks-per-second 0.01 \
    -s snakefile_divrge
