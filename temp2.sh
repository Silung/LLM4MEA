#!/bin/bash
list1=('ml-1m' 'steam')
list2=('narm' 'sas')

for i in "${list1[@]}"; do
  for j in "${list2[@]}"; do
    echo "--dataset_code ${i} --model_code ${j} --generated_sampler autoregressive"
    python distill.py --dataset_code ${i} --model_code ${j} --bb_model_code ${j} --num_generated_seqs 5000 --generated_sampler autoregressive -k 100
  done
done