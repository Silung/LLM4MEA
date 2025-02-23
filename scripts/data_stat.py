import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random

def stat(path, idxs=[10], split=True):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    print(dataset.keys())
    seqs = dataset['seqs']
    
    seqs = np.vectorize(r_smap.get)(seqs)
    titles = np.vectorize(item_info['title'].get)(seqs)
    data = None
    if dataset_name == 'ml-1m':
        genres = np.vectorize(item_info['genres'].get)(seqs)
    
        if split:
            genres_split = []
            for i in genres:
                for j in i:
                    genres_split += j.split('|')
            
            data = Counter(genres_split)
        else:
            data = Counter(genres.reshape(-1))
    
    d = ''
    for idx in idxs:
        t = list(titles[idx])
        if dataset_name == 'ml-1m':
            g = list(genres[idx])
            d += ', '.join([f'{t[i]}({g[i]})' for i in range(len(t))])
        else:
            d += ', '.join([f'{t[i]}' for i in range(len(t))])
        d += '\n'
    return data, d

def stat_org(path, idxs=[10], split=True):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    smap = dataset['smap']
    r_smap = {v: k for k, v in smap.items()}
    titles = []
    genres = []
    for k, v in dataset['train'].items():
        org_items = np.vectorize(r_smap.get)(v)
        title = np.vectorize(item_info['title'].get)(org_items)
        if dataset_name == 'ml-1m':
            genre = np.vectorize(item_info['genres'].get)(org_items)
            genre_split = []
            if split:
                for i in genre:
                    genre_split += i.split("|")
                genre = genre_split
        titles.append(title)
        if dataset_name == 'ml-1m':
            genres += list(genre)
    data = titles
    if dataset_name == 'ml-1m':
        data = Counter(genres)
    d = ''
    for idx in idxs:
        t = list(titles[idx])
        if dataset_name == 'ml-1m':
            g = list(genres[idx])
            d += ', '.join([f'{t[i]}({g[i]})' for i in range(len(g))])
        else:
            d += ', '.join([f'{t[i]}' for i in range(len(t))])
        d += '\n'
    return data, d

def fig():
    # data1, _ = stat(f'gen_data/ml-1m/{arch}_5000_100/autoregressive_dataset.pkl')
    data1, _ = stat(f'gen_data/ml-1m/{arch}_5000_100/llm_seq_dataset.pkl.old', split=split)
    data2, _ = stat_org('data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl', split=split)

    data_sum = (data1 + data2).most_common()[:10]
    data_sum_keys = dict(data_sum).keys()

    data1 = {k:dict(data1)[k] for k in data_sum_keys}
    data2 = {k:dict(data2)[k] for k in data_sum_keys}
    y1 = data1.values()
    y2 = data2.values()

    print(f'data1:\n {data1}')
    print(f'data2:\n {data2}')



    plt.figure(figsize = (10,8))
    plt.bar(np.arange(len(y1)), y1, width=0.4, color='tomato', label='LLM')
    # for x, y in enumerate(y1):
    #     plt.text(x, y, y)
        
    plt.bar(np.arange(len(y2)) + 0.4, y2, width=0.4, color='steelblue', label='Org')
    # for x, y in enumerate(y2):
    #     plt.text(x + 0.4, y, y)

    br_data_sum_keys = ['\n'.join(i.split('|')) for i in data_sum_keys]
    plt.xticks(np.arange(len(data_sum_keys)) + 0.5 / 2, br_data_sum_keys)
    plt.xticks(rotation=45)
    plt.xticks(fontsize = 8)

    # 设置标题和轴标签
    plt.title('Counts')
    plt.xlabel('genres')
    plt.ylabel('Count')
    plt.legend() 
    plt.savefig(f'pics/data_top10_{arch}_split={split}.jpg')
    
def cases():
    # _, s1 = stat(f'gen_data/ml-1m/bert_5000_100/self0_dataset.pkl', idxs=[23])
    _, s2 = stat(f'gen_data/steam/narm_5001_100/llm_seq0_dataset.pkl', idxs=[0,1,2])
    # _, s3 = stat(f'gen_data/ml-1m/bert_5000_100/autoregressive1_dataset.pkl', idxs=[23])
    # print('self\n'+s2)
    print('llm\n'+s2)
    # print('auto\n'+s3)
    
    
    
    # titles, s = stat_org(f'data/preprocessed/steam_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl', idxs=[i for i in range(30)])
    # print(s)
    
    # title_f = []
    # for title in titles:
    #     title_f += list(title)
    # random.shuffle(title_f)
    # print(title_f[:100])
    

arch = 'narm'
split= False
dataset_name = 'steam'

with open(f'data/preprocessed/{dataset_name}_text.pkl', 'rb') as f:
    text = pickle.load(f)
item_info = text['item']

with open(f'data/preprocessed/{dataset_name}_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl', 'rb') as f:
    smap = pickle.load(f)['smap']
r_smap = {v: k for k, v in smap.items()}
    
most_common10 = None

cases()
# fig()