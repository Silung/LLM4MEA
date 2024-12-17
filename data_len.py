import pickle
from collections import Counter


def len_dis(path):
    cc = 0
    f = open(path,'rb')
    org_dataset = pickle.load(f)
    for k,v in org_dataset['train'].items():
        cc += len(v)

    org_data = org_dataset['train']
    org_data_len = [len(v) for v in org_data.values()]
    data = Counter(org_data_len)
    print(data)
    return data

len_dis('/data/zhaoshilong/REA_with_llm/data/preprocessed/games_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
len_dis('/data/zhaoshilong/REA_with_llm/data/preprocessed/beauty_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
len_dis('/data/zhaoshilong/REA_with_llm/data/preprocessed/steam_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')