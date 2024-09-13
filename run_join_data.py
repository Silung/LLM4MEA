import os
import subprocess
import time
import pickle


def rename_old(path):
    if os.path.exists(path):
        target_path = path+'.old'
        rename_old(target_path)
        os.rename(path, target_path)
    else:
        return


# 配置
dis_gpu = [3]  # 可用显卡
seqs_per_proc = 500
id_start = 0
num_p = 50 - id_start
arch = 'sas'
dataset_name = 'steam'
total_seqs = seqs_per_proc * num_p    # num_generated_seqs的总和

step = 10

# llama 和 distill 程序路径
distill_dir = "/data/zhaoshilong/REA_with_llm"

root = f'gen_data/{dataset_name}/{arch}_{seqs_per_proc}_100'

for idx in range(num_p // step):
    paths = [f'llm_seq{i}_dataset.pkl' for i in range(idx * step , (idx + 1) * step)]
    print(paths)
    datasets = []
    for path in paths:
        with open(os.path.join(root, path), 'rb') as f:
            datasets.append(pickle.load(f))
        
    seqs, logits, candidates = [], [], []
    for dataset in datasets:
        seqs += dataset['seqs']
        logits += dataset['logits']
        candidates += dataset['candidates']
    dataset3 = {'seqs':seqs[:step*seqs_per_proc], 'logits':logits[:step*seqs_per_proc], 'candidates':candidates[:step*seqs_per_proc]}
    print(f'len(dataset3):{len(dataset3)}')
    print(f'len(dataset3["seqs"]):{len(dataset3["seqs"])}')
    
    if not os.path.exists(f'gen_data/{dataset_name}/{arch}_{step*seqs_per_proc}_100'):
        os.mkdir(f'gen_data/{dataset_name}/{arch}_{step*seqs_per_proc}_100')
    target_path = os.path.join(f'gen_data/{dataset_name}/{arch}_{step*seqs_per_proc}_100', f'llm_seq{idx}_dataset.pkl')
    rename_old(target_path)
    with open(target_path, 'wb') as f:
        pickle.dump(dataset3, f)
