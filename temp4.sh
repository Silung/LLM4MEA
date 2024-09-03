#!/bin/bash
list2=('0' '1' '2' '3' '4')

for j in "${list2[@]}"; do
  python distill.py --dataset_code ml-1m --model_code sas --bb_model_code sas --num_generated_seqs 1000 --generated_sampler llm_seq -k 100 --port 1962 --id ${j} --device cpu --gen_data_only
done