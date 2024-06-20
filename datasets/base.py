import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
from abc import *
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    @abstractmethod
    def is_zipfile(cls):
        pass

    @classmethod
    @abstractmethod
    def is_7zfile(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset
    
    def load_text(self):
        text_path = self._get_preprocessed_text_path()
        text = pickle.load(text_path.open('rb'))
        return text

    '''
    description: 过滤掉互动小于min_sc和min_uc的用户和物品
    param {*} self
    param {pd} df
    return {*}
    '''
    def filter_triplets(self, df:pd.DataFrame) -> pd.DataFrame:
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    '''
    description: 从1开始重新生成User和Item的Index
    param {*} self
    param {pd} df
    return {tuple}
    '''
    def densify_index(self, df:pd.DataFrame):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    '''
    description: 将item根据user聚合，按照时间排序形成序列数据，再切分测试集（倒数第一个item）、验证集（倒数第二个item）和训练集（其余）。
    param {*} self
    param {pandas.DataFrame} df 待切分数据
    param {dict[int, int]} user_count 从原始UserID到新生成的Index的映射
    return {*}
    '''
    def split_df(self, df:pd.DataFrame, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            # 按user聚合items
            user_group = df.groupby('uid')
            # 根据timestamp排序
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
            # 划分数据
            train, val, test = {}, {}, {}
            for i in range(user_count):
                user = i + 1
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError
    
    # 一些路径操作
    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
    
    def _get_preprocessed_text_path(self):
        folder = self._get_preprocessed_root_path()
        return folder.joinpath(f'{self.code()}_text.pkl')
