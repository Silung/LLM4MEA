import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

arch = 'narm'
split= False

with open('data/preprocessed/ml-1m_text.pkl', 'rb') as f:
    text = pickle.load(f)
item_info = text['item']

most_common10 = None

def stat(path, idxs=[10], split=True):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    print(dataset.keys())
    seqs = dataset['seqs']

    titles = np.vectorize(item_info['title'].get)(seqs)
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
        g = list(genres[idx])
        d += ', '.join([f'{t[i]}({g[i]})' for i in range(len(g))])
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
        genre = np.vectorize(item_info['genres'].get)(org_items)
        genre_split = []
        if split:
            for i in genre:
                genre_split += i.split("|")
            genre = genre_split
        titles += list(title)
        genres += list(genre)
    
    data = Counter(genres)
    d = ''
    for idx in idxs:
        t = list(titles[idx])
        g = list(genres[idx])
        d += ', '.join([f'{t[i]}({g[i]})' for i in range(len(g))])
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
    _, s1 = stat(f'gen_data/ml-1m/bert_5000_100/self0_dataset.pkl', idxs=[23])
    _, s2 = stat(f'gen_data/ml-1m/bert_5000_100/llm_seq0_dataset.pkl', idxs=[23])
    print('self\n'+s2)
    print('llm\n'+s1)
    
cases()
# fig()