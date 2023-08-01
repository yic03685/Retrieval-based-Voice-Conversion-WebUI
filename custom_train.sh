#!/bin/bash

TRAINING_DATA_FOLDER="training_data"
TRAINING_WORKSPACE_FOLDER="logs"
MODEL_OUTPUT_FOLDER="trained"
MODEL_NAME=$1

WEIGHTS_FOLDER="weights" #defined in the original training script

mkdir -p $TRAINING_WORKSPACE_FOLDER/$MODEL_NAME

# Preprocess [slice / normalization]
printf "==Preprocess slice / normalization==\n"
python trainset_preprocess_pipeline_print.py $TRAINING_DATA_FOLDER 40000 22 $TRAINING_WORKSPACE_FOLDER/$MODEL_NAME False

# Preprocess [extract pitch]
printf "\n==Preprocess extract pitch==\n"
python extract_f0_print.py $TRAINING_WORKSPACE_FOLDER/$MODEL_NAME 22 harvest

# Preprocess [extract feature v2]
printf "\n==Preprocess extract feature==\n"
python extract_feature_print.py cuda:0 1 0 0 $TRAINING_WORKSPACE_FOLDER/$MODEL_NAME v2

# write filelist
printf "\n==Write filelist==\n"
python filelist_generator.py --model_name $MODEL_NAME

# start training [extract feature v2]
printf "\n==Start training==\n"
python train_nsf_sim_cache_sid_load_pretrain.py -e $MODEL_NAME -sr "40k" -f0 1 -bs 12 -g 0 -te 20 -se 5 -pg pretrained_v2/f0G40k.pth -pd pretrained_v2/f0D40k.pth -l 0 -c 0 -sw 0 -v "v2"

# start indexing
printf "\n==Create index==\n"
python index_generator.py --model_name $MODEL_NAME

# move models to the output folder
printf "\n==Move models to output==\n"
mkdir -p $MODEL_OUTPUT_FOLDER/$MODEL_NAME
cp $TRAINING_WORKSPACE_FOLDER/$MODEL_NAME/added_IVF21_Flat_nprobe_1_ian_test2_v2.index $MODEL_OUTPUT_FOLDER/$MODEL_NAME/model.index
cp $WEIGHTS_FOLDER/${MODEL_NAME}.pth $MODEL_OUTPUT_FOLDER/$MODEL_NAME/model.pth

printf "\n==Finished!!==\n"