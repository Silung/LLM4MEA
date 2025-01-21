from pathlib import Path
import ast
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

class TimeIntervalAnalyzer:
    def __init__(self, min_uc=5, min_sc=5, dataset_type='steam'):
        """初始化分析器
        Args:
            min_uc: 最少用户交互数
            min_sc: 最少物品交互数
            dataset_type: 数据集类型 ('steam', 'beauty', 'games')
        """
        self.min_uc = min_uc
        self.min_sc = min_sc
        self.dataset_type = dataset_type

    def load_ratings_df(self, file_path: Path) -> pd.DataFrame:
        """加载数据集
        Args:
            file_path: 数据文件路径
        Returns:
            包含用户交互信息的DataFrame
        """
        print('加载数据集...')
        
        if self.dataset_type == 'steam':
            return self._load_steam_data(file_path)
        elif self.dataset_type in ['beauty', 'games']:
            return self._load_amazon_data(file_path)
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")

    def _load_steam_data(self, file_path: Path) -> pd.DataFrame:
        """加载Steam数据集"""
        print('加载数据集...')
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                temp = ast.literal_eval(line)
                data.append([
                    temp['username'],
                    temp['product_id'],
                    pd.to_datetime(temp['date'])
                ])
        
        return pd.DataFrame(data, columns=['uid', 'sid', 'timestamp'])

    def _load_amazon_data(self, file_path: Path) -> pd.DataFrame:
        """加载Amazon Beauty/Games数据集"""
        print('加载数据集...')
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        # 转换时间戳为datetime对象
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    def filter_triplets(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤掉交互次数过少的用户和物品
        Args:
            df: 原始DataFrame
        Returns:
            过滤后的DataFrame
        """
        print('过滤数据...')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        
        return df

    def calculate_time_intervals(self, df: pd.DataFrame):
        """计算用户交互的时间间隔统计信息
        Args:
            df: 数据DataFrame
        Returns:
            统计信息字典
        """
        print('计算时间间隔...')
        
        # 按用户和时间戳排序
        df = df.sort_values(['uid', 'timestamp'])
        
        # 计算时间差（天）
        df['time_diff'] = df.groupby('uid')['timestamp'].diff()
        df['time_diff_days'] = df['time_diff'].dt.total_seconds() / (24 * 60 * 60)
        
        # 计算每个用户的平均间隔
        user_mean_intervals = df.groupby('uid')['time_diff_days'].mean()
        
        stats = {
            'overall_mean': df['time_diff_days'].mean(),
            'overall_median': df['time_diff_days'].median(),
            'overall_std': df['time_diff_days'].std(),
            'min_interval': df['time_diff_days'].min(),
            'max_interval': df['time_diff_days'].max(),
            'user_mean_interval': user_mean_intervals.mean(),
            'user_median_interval': user_mean_intervals.median()
        }
        
        return stats

    def analyze(self, file_path: Path):
        """执行完整的分析流程
        Args:
            file_path: 数据文件路径
        """
        # 加载数据
        df = self.load_ratings_df(file_path)
        print(f'原始数据统计:')
        print(f'用户数: {df["uid"].nunique()}')
        print(f'物品数: {df["sid"].nunique()}')
        print(f'交互数: {len(df)}')
        print(f'平均每用户交互数: {len(df)/df["uid"].nunique():.2f}')
        print(f'平均每物品交互数: {len(df)/df["sid"].nunique():.2f}')
        
        # 计算原始数据的时间间隔
        stats_before = self.calculate_time_intervals(df)
        print('\n过滤前时间间隔统计:')
        print(f'总体平均间隔: {stats_before["overall_mean"]:.2f} 天')
        print(f'总体中位数间隔: {stats_before["overall_median"]:.2f} 天')
        print(f'总体标准差: {stats_before["overall_std"]:.2f} 天')
        print(f'最小间隔: {stats_before["min_interval"]:.2f} 天')
        print(f'最大间隔: {stats_before["max_interval"]:.2f} 天')
        print(f'用户平均间隔的平均值: {stats_before["user_mean_interval"]:.2f} 天')
        print(f'用户平均间隔的中位数: {stats_before["user_median_interval"]:.2f} 天')
        
        # 过滤数据
        df = self.filter_triplets(df)
        print(f'\n过滤后数据统计:')
        print(f'用户数: {df["uid"].nunique()}')
        print(f'物品数: {df["sid"].nunique()}')
        print(f'交互数: {len(df)}')
        print(f'平均每用户交互数: {len(df)/df["uid"].nunique():.2f}')
        print(f'平均每物品交互数: {len(df)/df["sid"].nunique():.2f}')
        
        # 计算过滤后的时间间隔
        stats_after = self.calculate_time_intervals(df)
        print('\n过滤后时间间隔统计:')
        print(f'总体平均间隔: {stats_after["overall_mean"]:.2f} 天')
        print(f'总体中位数间隔: {stats_after["overall_median"]:.2f} 天')
        print(f'总体标准差: {stats_after["overall_std"]:.2f} 天')
        print(f'最小间隔: {stats_after["min_interval"]:.2f} 天')
        print(f'最大间隔: {stats_after["max_interval"]:.2f} 天')
        print(f'用户平均间隔的平均值: {stats_after["user_mean_interval"]:.2f} 天')
        print(f'用户平均间隔的中位数: {stats_after["user_mean_interval"]:.2f} 天')
        
        return stats_after

def main():
    # 设置数据路径和类型
    datasets = {
        'beauty': Path('data/beauty/beauty.csv'),
        'games': Path('data/games/games.csv'),
        'steam': Path('data/steam/steam.json')
    }
    
    # 为每个数据集执行分析
    for dataset_type, file_path in datasets.items():
        if file_path.exists():
            print(f'\n分析 {dataset_type} 数据集...')
            analyzer = TimeIntervalAnalyzer(min_uc=5, min_sc=5, dataset_type=dataset_type)
            stats = analyzer.analyze(file_path)
            print('-' * 50)

if __name__ == '__main__':
    main()