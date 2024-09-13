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
    
diversity('gen_data.old/beauty/narm_5000_100/llm_seq_dataset.pkl')
diversity('gen_data.old/beauty/narm_5000_100/autoregressive_dataset.pkl')
diversity('data/preprocessed/beauty_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
