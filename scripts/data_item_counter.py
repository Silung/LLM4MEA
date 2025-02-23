import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

most_common10 = None

def stat(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    seqs = dataset['seqs']

    cc = []
    for seq in seqs:
        cc += seq
    data = set(cc)
    return data

def stat_org(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    cc = []
    for k, v in dataset['train'].items():
        cc += v
    
    data = set(cc)
    return data

def diversity(path):
    try:
        data = stat(path)
    except:
        data = stat_org(path)

    print(len(data))
    
# dataset_names = ['ml-1m', 'steam', 'beauty']
# for dataset_name in dataset_names:
#     print(dataset_name)
#     print("LLM")
#     diversity(f'gen_data.old/{dataset_name}/narm_5000_100/llm_seq_dataset.pkl')
#     print("Auto")
#     diversity(f'gen_data.old/{dataset_name}/narm_5000_100/autoregressive_dataset.pkl')
#     print("Org")
#     diversity(f'data/preprocessed/{dataset_name}_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
# diversity(f'/data/zhaoshilong/REA_with_llm/gen_data/steam/narm_5001_100/llm_seq0_dataset.pkl')
# diversity(f'data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
diversity(f'/data/zhaoshilong/REA_with_llm/gen_data/steam/narm_5001_100/autoregressive1_dataset.pkl')


def info(path):
    print(f'Path: {path}')
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
        
    user_cc = len(dataset['train'])
    print(f'Users: {user_cc}')
    
    cc = []
    max_len = -1
    for k, v in dataset['train'].items():
        cc += v
        max_len = max(max_len, len(v))
    for k, v in dataset['val'].items():
        cc += v
        max_len = max(max_len, len(v))
    for k, v in dataset['test'].items():
        cc += v
        max_len = max(max_len, len(v))
    data = set(cc)
    item_cc = len(data)
    print(f'Items: {item_cc}')
    print(f'Interaction: {len(cc)}')
    print(f'avg len: {len(cc)/user_cc}')
    print(f'max len: {max_len}')
    print(f'Sparsity: {1 -  len(cc)/ (item_cc*user_cc)}')
    return data

info(f'data/preprocessed/beauty_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
info(f'data/preprocessed/games_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
info(f'data/preprocessed/steam_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
