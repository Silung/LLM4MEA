import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

most_common10 = None

def stat(path, item_info):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    seqs = dataset['seqs']

    # print('8891' in item_info['genres'])
    item_info['genres'] = {int(k):v for k,v in item_info['genres'].items()}
    genres = np.vectorize(item_info['genres'].get)(seqs)

    cc = []
    for genre in genres:
        # if len(set(genre)) != 1:
        #     print(genre)
        cc.append(len(set(genre)))
        
    data = Counter(cc)
    return data

def stat_org(path, item_info):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    smap = dataset['smap']
    r_smap = {v: k for k, v in smap.items()}
    cc = []
    for k, v in dataset['train'].items():
        org_items = np.vectorize(r_smap.get)(v)
        genre = np.vectorize(item_info['genres'].get)(org_items)
        cc.append(len(set(list(genre))))
    
    data = Counter(cc)
    return data

def diversity(name, path):
    if 'ml-1m' in path:
        with open('data/preprocessed/ml-1m_text.pkl', 'rb') as f:
            text = pickle.load(f)
        item_info = text['item']
    elif 'steam' in path:
        with open('data/preprocessed/steam_text.pkl', 'rb') as f:
            text = pickle.load(f)
        item_info = text['item']
    else:
        raise("Not Impl..")


    try:
        data = stat(path, item_info)
    except:
        data = stat_org(path, item_info)

    data_keys = sorted(list(dict(data).keys()))

    data = {k:dict(data)[k] for k in data_keys}
    y = data.values()

    print(f'{name} data:\n {data}')


    plt.figure(figsize = (10,8))
    plt.bar(np.arange(len(y)), y, width=0.4, color='tomato', label='AG')
    # for x, y in enumerate(y1):
    #     plt.text(x, y, y)
        
    # plt.bar(np.arange(len(y2)) + 0.4, y2, width=0.4, color='steelblue', label='LLM')
    # for x, y in enumerate(y2):
    #     plt.text(x + 0.4, y, y)

    # br_data_sum_keys = ['\n'.join(i.split('|')) for i in data_sum_keys]
    plt.xticks(np.arange(len(data_keys)), data_keys)
    # plt.xticks(rotation=45)
    plt.xticks(fontsize = 8)

    # 设置标题和轴标签
    plt.title(f'Diversity of Data from {name}')
    plt.xlabel('Diversity')
    plt.ylabel('Count')
    plt.savefig(f'pics/data_diversity_{name}.jpg')
    
# diversity('narm', 'gen_data/ml-1m/narm_5000_100/llm_seq_dataset.pkl.old')
# diversity('narm-new', 'gen_data/ml-1m/narm_5000_100/llm_seq_dataset.pkl')
# diversity('narm-ag', 'gen_data/ml-1m/narm_5000_100/autoregressive_dataset.pkl')
# diversity('sas', 'gen_data/ml-1m/sas_5000_100/llm_seq_test_dataset.pkl')
# diversity('sas-ag', 'gen_data/ml-1m/sas_5000_100/autoregressive_dataset.pkl')
# diversity('ml-1m', 'data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
# diversity('bert', 'gen_data/ml-1m/bert_5000_100/llm_seq_dataset.pkl')
# diversity('bert-exam', 'gen_data/ml-1m/bert_5000_100/llm_exam_dataset.pkl')
# diversity('bert-random', 'gen_data/ml-1m/bert_5000_100/random_dataset.pkl')
# diversity('narm-org', 'gen_data/steam/narm_5000_100/llm_seq_dataset.pkl')
diversity('narm-ml-len=50', 'gen_data/ml-1m/narm_5000_100/llm_seq_dataset.pkl')