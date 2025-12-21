import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MovieLensDataHandler:
    def __init__(self, ratings_path, movies_path, dataset_type='ml-latest-small'):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.dataset_type = dataset_type  # 'ml-latest-small' or 'ml-1m'
        self.ratings_df = None
        self.movies_df = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """加载MovieLens数据集"""
        if self.dataset_type == 'ml-1m':
            # ml-1m数据集使用::作为分隔符
            self.ratings_df = pd.read_csv(
                self.ratings_path, 
                sep='::', 
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python',
                encoding='latin-1'
            )
            
            self.movies_df = pd.read_csv(
                self.movies_path, 
                sep='::', 
                names=['movieId', 'title', 'genres'],
                engine='python',
                encoding='latin-1'
            )
        else:
            # 默认处理ml-latest-small数据集
            self.ratings_df = pd.read_csv(self.ratings_path)
            self.movies_df = pd.read_csv(self.movies_path)
        
        print(f"评分数据形状: {self.ratings_df.shape}")
        print(f"电影数据形状: {self.movies_df.shape}")
        
        return self.ratings_df, self.movies_df
    
    def preprocess_data(self):
        """数据预处理"""
        # 合并评分和电影数据
        merged_df = pd.merge(self.ratings_df, self.movies_df, on='movieId')
        
        # 显示基本统计信息
        print("数据集基本信息:")
        print(f"用户数量: {merged_df['userId'].nunique()}")
        print(f"电影数量: {merged_df['movieId'].nunique()}")
        print(f"评分数量: {len(merged_df)}")
        print(f"评分范围: {merged_df['rating'].min()} - {merged_df['rating'].max()}")
        
        return merged_df
    
    def split_data(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        self.train_data, self.test_data = train_test_split(
            self.ratings_df, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {len(self.train_data)}")
        print(f"测试集大小: {len(self.test_data)}")
        
        return self.train_data, self.test_data
    
    def create_user_item_matrix(self, data):
        """创建用户-物品评分矩阵"""
        user_item_matrix = data.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        return user_item_matrix

# 示例使用
if __name__ == "__main__":
    # 初始化数据处理器
    data_handler = MovieLensDataHandler(
        ratings_path="/home/admin/myfile/buaa_myxxjs/film/personalized_ir_system/data/ml-1m/ratings.dat",
        movies_path="/home/admin/myfile/buaa_myxxjs/film/personalized_ir_system/data/ml-1m/movies.dat",
        dataset_type='ml-1m'
    )
    
    # 加载数据
    ratings, movies = data_handler.load_data()
    
    # 数据预处理
    merged_data = data_handler.preprocess_data()
    
    # 划分数据集
    train_data, test_data = data_handler.split_data()
    
    # 创建用户-物品矩阵
    user_item_matrix = data_handler.create_user_item_matrix(train_data)
    print(f"用户-物品矩阵形状: {user_item_matrix.shape}")