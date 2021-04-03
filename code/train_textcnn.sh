#!/bin/bash
######## Part 1 #########
# Script parameters     #
#########################

# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu

# Specify the QOS, mandatory option
#SBATCH --qos=normal

# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=junogpu

# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=TextCNN

# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tc_medical_nlp/code/log/gpujob-%j.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=10GB

# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1

######## Part 2 ######
# Script workload    #
######################

#export PYTHONPATH=$PYTHONPATH:/hpcfs/juno/junogpu/xuhangkun/SoftWare/VsCode/.local/lib/python3.6/site-packages
source /afs/ihep.ac.cn/users/x/xuhangkun/.bashrc.anaconda
conda activate pytorch
export PROJTOP=/hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tc_medical_nlp
python train.py -model TextCNN -epoch 10
