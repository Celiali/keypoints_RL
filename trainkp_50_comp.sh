#!/usr/bin/env bash
#SBATCH --mem  32GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --constrain "khazadum|rivendell|belegost|shire"
#SBATCH --mail-type FAIL
#SBATCH --mail-user zehang@kth.se
#SBATCH --output /Midgard/home/zehang/project/keypoint_humanoids/stdout/stdout_kp_50_comp_output.log
#SBATCH --error /Midgard/home/zehang/project/keypoint_humanoids/stdout/stdout_kp_50_comp_error.log

nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2cuda10

DATAPATH=/nas/zehang/keypoint_humanoids
LOCALDATAPATH=/local_storage/users/zehang/keypoint_humanoids

if [ ! -d $LOCALDATAPATH ]
then
 mkdir $LOCALDATAPATH
 mkdir $LOCALDATAPATH/checkpoint
 mkdir $LOCALDATAPATH/logs
fi

scp -r $DATAPATH/data $LOCALDATAPATH/

RUNPATH=/Midgard/home/zehang/project/keypoint_humanoids

# store checkpoint and logs
CKFOLDER=$LOCALDATAPATH/comp_checkpoint
if [ ! -d $CKFOLDERR ]
then
 mkdir $CKFOLDER
fi

LOGFOLDER=$LOCALDATAPATH/comp_logs

cd $RUNPATH
python src/trainKp.py --data_path $LOCALDATAPATH/data --log_dir $LOGFOLDER --ck_path $CKFOLDER --batch_size 32 --epoch 100 --model pcn_det_comp --augrot --augocc --augsca --savemodel --tasklist 8 18 10 20 --numkp 50

scp -r $LOCALDATAPATH/comp/checkpoint/* $DATAPATH/comp/checkpoint/
scp -r $LOCALDATAPATH/comp/logs/* $DATAPATH/comp/logs/