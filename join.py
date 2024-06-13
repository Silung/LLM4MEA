import os
import pickle

root = 'gen_data/ml-1m/bert_250_100'
paths = ['llm_pfl4_dataset.pkl','llm_pfl5_dataset.pkl']
datasets = []
for path in paths:
    with open(os.path.join(root, path), 'rb') as f:
         datasets.append(pickle.load(f))
    
k=500
seqs, logits, candidates = [], [], []
for dataset in datasets:
    seqs += dataset['seqs']
    logits += dataset['logits']
    candidates += dataset['candidates']
dataset3 = {'seqs':seqs[:k], 'logits':logits[:k], 'candidates':candidates[:k]}
print(f'len(dataset3):{len(dataset3)}')
print(f'len(dataset3["seqs"]):{len(dataset3["seqs"])}')

with open(os.path.join(root, 'temp.pkl'), 'wb') as f:
    pickle.dump(dataset3, f)