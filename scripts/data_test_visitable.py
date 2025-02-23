import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random

def stat_org(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    # smap = dataset['smap']
    train = set()
    for k, v in dataset['train'].items():
        train = train.union(v)
    
    test = set()
    for k, v in dataset['val'].items():
        test = test.union(v)

    print(len(dataset['test']))
    return train, test

def stat_gen(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    # smap = dataset['smap']
    seq = set()
    # print(dataset['seqs'])
    for s in dataset['seqs']:
        seq = seq.union(s)
    # print(dataset.keys())
    print(len(dataset['seqs']))
    return seq


org_train, test = stat_org('data/preprocessed/beauty_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
train = stat_gen('gen_data/beauty/bert_5001_100/autoregressive0_dataset.pkl')

print(f'item count of test: {len(test)}')
print(len(test - train))
print(len(test - org_train))