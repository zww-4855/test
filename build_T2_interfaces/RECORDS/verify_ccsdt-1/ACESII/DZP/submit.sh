#!/bin/bash
#SBATCH --job-name=ac9IP
#SBATCH --output=zzz.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=3gb

echo $SLURM_NODELIST
echo $SLURM_JOBID
export PATH=/apps/shared/bartlett/ACESII-2.14.0/bin:$PATH

#/blue/bartlett/z.windom/incBasisFxnACES/ACESII-2.13.0/bin/:$PATH
export TESTROOT=$SLURM_SUBMIT_DIR

module load intel

xaces2 > out.out

#xvee >> out.out
echo "`date`" >> out.out

