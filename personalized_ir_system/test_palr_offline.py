#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline test script for PALR Recommender
Tests PALR functionality without requiring internet connection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_handler import MovieLensDataHandler
from lightgcn_recommender import LightGCNRecommender
import pandas as pd
import numpy as np

# Mock classes to simulate LLM functionality without internet
class MockSentenceEncoder:
    """Mock sentence encoder for offline testing"""
    def encode(self, sentences):
        # Return random embeddings
        if isinstance(sentences, str):
            return np.random.rand(384)
        return [np.random.rand(384) for _ in sentences]

class MockPALRRecommender:
    """
    Mock PALR Recommender for offline testing
    """
    
    def __init__(self, base_recommender, movies_df):
        self.base_recommender = base_recommender
        self.movies_df = movies_df
        self.sentence_encoder = MockSentenceEncoder()
    
    def _get_movie_info(self, movie_id: int) -> str:
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_row.empty:
            return f"Movie ID: {movie_id}"
        title = movie_row.iloc[0]['title']
        genres = movie_row.iloc[0]['genres']
        return f"Title: {title}, Genres: {genres}"
    
    def recommend(self, user_id: int, user_history: list, 
                  candidate_items: list, top_k: int = 10) -> list:
        """
        Mock enhanced recommendation - randomly reorders candidates
        """
        # Just shuffle the candidates to simulate "enhancement"
        import random
        shuffled = candidate_items.copy()
        random.shuffle(shuffled)
        return shuffled[:top_k]

def test_palr_offline():
    """Test PALR functionality offline"""
    print("Testing PALR Recommender functionality (offline version)...")
    
    # 初始化数据处理器
    data_handler = MovieLensDataHandler(
        ratings_path="data/ml-1m/ratings.dat",
        movies_path="data/ml-1m/movies.dat",
        dataset_type='ml-1m'
    )
    
    # 加载数据
    ratings, movies = data_handler.load_data()
    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies")
    
    # 数据预处理
    merged_data = data_handler.preprocess_data()
    
    # 划分数据集
    train_data, test_data = data_handler.split_data()
    
    # 创建用户-物品矩阵
    user_item_matrix = data_handler.create_user_item_matrix(train_data)
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    
    # 初始化LightGCN推荐器
    print("Initializing LightGCN recommender...")
    lightgcn_recommender = LightGCNRecommender(user_item_matrix)
    
    # 加载预训练模型
    model_path = "models/lightgcn_ml1m.pth"
    if os.path.exists(model_path):
        try:
            lightgcn_recommender.load_model(model_path)
            print("Successfully loaded pre-trained LightGCN model")
        except Exception as e:
            print(f"Failed to load pre-trained model: {e}")
            return
    else:
        print("Pre-trained model not found")
        return
    
    # 初始化Mock PALR推荐器
    print("Initializing Mock PALR recommender (offline version)...")
    palr_recommender = MockPALRRecommender(lightgcn_recommender, movies)
    
    # 测试用户ID
    test_user_id = 1
    top_k = 5
    
    # 获取用户历史评分
    user_history = []
    user_ratings = ratings[ratings['userId'] == test_user_id]
    for _, row in user_ratings.iterrows():
        user_history.append((row['movieId'], row['rating']))
    
    print(f"User {test_user_id} has {len(user_history)} historical ratings")
    
    # 获取LightGCN的基础推荐
    print("Getting base recommendations from LightGCN...")
    base_recommendations = lightgcn_recommender.recommend(test_user_id-1, top_k*2)
    print(f"Base recommendations: {base_recommendations[:3]}...")  # 只显示前3个
    
    # 使用Mock PALR增强推荐
    print("Getting mock enhanced recommendations from PALR...")
    enhanced_recommendations = palr_recommender.recommend(
        test_user_id, user_history, base_recommendations, top_k
    )
    print(f"Enhanced recommendations: {enhanced_recommendations}")
    
    # 显示推荐结果
    print("\n--- Recommendation Results ---")
    print(f"Base recommendations (LightGCN):")
    for i, (item_id, score) in enumerate(base_recommendations[:top_k]):
        # 转换回1-based索引以匹配movies数据框
        movie_id = item_id + 1
        movie_info = movies[movies['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            print(f"  {i+1}. {title} (ID: {movie_id}, Score: {score:.4f})")
    
    print(f"\nEnhanced recommendations (Mock PALR):")
    for i, (item_id, score) in enumerate(enhanced_recommendations):
        movie_info = movies[movies['movieId'] == item_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            print(f"  {i+1}. {title} (ID: {item_id}, Score: {score:.4f})")
    
    print("\n✅ Offline PALR test completed successfully!")

if __name__ == "__main__":
    test_palr_offline()