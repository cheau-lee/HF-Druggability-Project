#!/bin/bash
#$ -cwd           
#$ -pe smp 5      
#$ -l h_rt=240:0:0  
#$ -l h_vmem=10G   
#$ -m bea

module load anaconda3
conda activate drugnome_env
cd /data/home/bt23020/HF_Project/DrugnomeAI-release

# I have used 4 different settings for optimising the speed and comparison between models (# accordingly for separate scripts)

# Fast setting with the top xgboost model only with iterations
drugnomeai \
  -c /data/home/bt23020/HF_Project/DrugnomeAI-release/drugnome_ai/conf/hf_config.yaml \
  -o fast_HF_287_xgboost_results \
  --superv-models xgb \
  -m \
  -l \
  -k /data/home/bt23020/HF_Project/GENE_LIST_0001_287.txt \
  -n 5 \
  -i 5
  
# Fast setting with all models with -f 
  drugnomeai \
  -c /data/home/bt23020/HF_Project/DrugnomeAI-release/drugnome_ai/conf/hf_config.yaml \
  -o fast_complete_hf_results \
  -f \
  -m \
  -l \
  -k /data/home/bt23020/HF_Project/GENE_LIST_0001_287.txt \

# Normal setting with the top xgboost model only 
drugnomeai -c /data/home/bt23020/HF_Project/DrugnomeAI-release/drugnome_ai/conf/hf_config.yaml -o 287_HF_xg_norm_results --superv-models xgb -m -l -k /data/home/bt23020/HF_Project/GENE_LIST_0001_287.txt

# Normal setting with all models
drugnomeai \
  -c /data/home/bt23020/HF_Project/DrugnomeAI-release/drugnome_ai/conf/hf_config.yaml \
  -o complete_hf_results \
  -m \
  -l \
  -k /data/home/bt23020/HF_Project/GENE_LIST_0001_287.txt \

