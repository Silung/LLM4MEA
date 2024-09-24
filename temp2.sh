#!/bin/bash
sleep 12800
python distill.py --dataset_code beauty --model_code bert --bb_model_code bert --num_generated_seqs 5000 --generated_sampler llm_seq -k 100 --id 0