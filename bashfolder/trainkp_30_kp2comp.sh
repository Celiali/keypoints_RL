#!/usr/bin/env bash
#SBATCH --mem  32GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --constrain "khazadum|rivendell|belegost|shire"
#SBATCH --mail-type FAIL
#SBATCH --mail-user zehang@kth.se
#SBATCH --output /Midgard/home/zehang/project/keypoint_humanoids/stdout/stdout_kp_30_kp2comp_output.log
#SBATCH --error /Midgard/home/zehang/project/keypoint_humanoids/stdout/stdout_kp_30_kp2comp_error.log

nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2cuda10

DATAPATH=/nas/zehang/keypoint_humanoids
LOCALDATAPATH=/local_storage/users/zehang/keypoint_humanoids

# path to the pre-trained detector model
DETECTOR_CKFOLDER=$DATAPATH/detect_checkpoint

##############################################
if [ ! -d $LOCALDATAPATH ]
then
 mkdir $LOCALDATAPATH
fi

if [ ! -d $LOCALDATAPATH/kp2comp_checkpoint ]
then
 mkdir $LOCALDATAPATH/kp2comp_checkpoint
fi

if [ ! -d $LOCALDATAPATH/kp2comp_logs ]
then
 mkdir $LOCALDATAPATH/kp2comp_logs
fi

###############################################
#scp -r $DATAPATH/data $LOCALDATAPATH/

RUNPATH=/Midgard/home/zehang/project/keypoint_humanoids

#TODO: 8 18 10 20
cd $RUNPATH
python src/trainKp_det2comp.py --data_path $LOCALDATAPATH/data --log_dir $LOCALDATAPATH/kp2comp_logs --detector_ck_path $DETECTOR_CKFOLDER --decoder_ck_path $LOCALDATAPATH/kp2comp_checkpoint --batch_size 32 --epoch 100 --model pcn_det --augrot --augocc --augsca --savemodel --tasklist 2 4 6 8 10 12 14 16 18 20--numkp 30

scp -r $LOCALDATAPATH/kp2comp_checkpoint/* $DATAPATH/kp2comp_checkpoint/
scp -r $LOCALDATAPATH/kp2comp_logs/* $DATAPATH/kp2comp_logs/