import os
import subprocess
import time
import pickle
from collections import Counter
import math
import torch


def load_data(arch, dataset_name, sampler):
    distill_dir = "/data/zhaoshilong/REA_with_llm"

    root = f'{distill_dir}/gen_data/{dataset_name}/{arch}_5000_100'

    for i in range(5):
        path = f'{sampler}{i}_dataset.pkl'
        try:
            with open(os.path.join(root, path), 'rb') as f:
                dataset = pickle.load(f)
            break
        except:
            continue
    seqs = dataset['seqs']
    return seqs


def extract_ngrams(sequences, n):
    ngrams = []
    for seq in sequences:  # Convert to list if tensor
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i + n])
            ngrams.append(ngram)
    return ngrams

def compute_kl_divergence(p_dist, q_dist, eps=1e-10):
    kl_div = 0.0
    all_ngrams = set(p_dist.keys()).union(set(q_dist.keys()))
    
    for ngram in all_ngrams:
        p_prob = p_dist.get(ngram, eps)
        q_prob = q_dist.get(ngram, eps)
        kl_div += p_prob * math.log(p_prob / q_prob)
    
    return kl_div

def ngram_distribution(ngrams):
    total_count = len(ngrams)
    ngram_count = Counter(ngrams)
    return {ngram: count / total_count for ngram, count in ngram_count.items()}

def stat_ngram(arch, dataset_name, samplers, n):
    print(f'arch={arch}\tdataset_name={dataset_name}\tn={n}')
    seqs1 = load_data(arch, dataset_name, samplers[0])
    ngrams1 = extract_ngrams(seqs1, n)
    p_dist = ngram_distribution(ngrams1)

    for sampler in samplers[1:]:
        seqs2 = load_data(arch, dataset_name, sampler)
        ngrams2 = extract_ngrams(seqs2, n)
        q_dist = ngram_distribution(ngrams2)
        kl_divergence = compute_kl_divergence(p_dist, q_dist)
        print(f'p:{samplers[0]}\t q:{sampler}\t\t kl:{kl_divergence}')

samplers = ['self', 'random', 'autoregressive', 'llm_seq']
dataset_names = ['ml-1m', 'steam', 'beauty']
archs = ['narm', 'sas', 'bert']

# for i in range(1,4):
#     for dataset_name in dataset_names:
#         stat_ngram('bert', dataset_name, samplers, i)

# stat_ngram(archs[2], dataset_names[0], samplers, 2)

print(compute_kl_divergence({'a':1}, {'b':1}))
print(compute_kl_divergence({'a':1}, {'a':0.7, 'b':0.3}))