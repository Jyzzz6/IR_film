import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class CollaborativeFilteringRecommender:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        
    def compute_user_similarity(self):
        """计算用户相似度矩阵（基于用户的协同过滤）"""
        # 将DataFrame转换为稀疏矩阵以提高计算效率
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # 计算用户之间的余弦相似度
        self.user_similarity_matrix = cosine_similarity(sparse_matrix)
        
        # 将相似度矩阵转换为DataFrame以便于操作
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("用户相似度矩阵计算完成")
        return self.user_similarity_matrix
    
    def compute_item_similarity(self):
        """计算物品相似度矩阵（基于物品的协同过滤）"""
        # 转置矩阵以计算物品之间的相似度
        sparse_matrix = csr_matrix(self.user_item_matrix.T.values)
        
        # 计算物品之间的余弦相似度
        self.item_similarity_matrix = cosine_similarity(sparse_matrix)
        
        # 将相似度矩阵转换为DataFrame
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        print("物品相似度矩阵计算完成")
        return self.item_similarity_matrix
    
    def predict_user_based(self, user_id, item_id):
        """基于用户的协同过滤预测评分"""
        if user_id not in self.user_item_matrix.index:
            return 0
        
        if item_id not in self.user_item_matrix.columns:
            return 0
            
        # 获取用户已评分的物品
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not rated_items:
            return 0
            
        # 计算该用户与其他用户的相似度
        user_similarities = self.user_similarity_matrix.loc[user_id]
        
        # 计算加权平均评分
        weighted_sum = 0
        similarity_sum = 0
        
        for other_user in self.user_item_matrix.index:
            if other_user == user_id:
                continue
                
            # 检查其他用户是否对目标物品评过分
            other_rating = self.user_item_matrix.loc[other_user, item_id]
            if other_rating > 0:
                similarity = user_similarities[other_user]
                weighted_sum += similarity * other_rating
                similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 0
            
        predicted_rating = weighted_sum / similarity_sum
        return predicted_rating
    
    def predict_item_based(self, user_id, item_id):
        """基于物品的协同过滤预测评分"""
        if user_id not in self.user_item_matrix.index:
            return 0
        
        if item_id not in self.user_item_matrix.columns:
            return 0
            
        # 获取用户已评分的物品
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not rated_items:
            return 0
            
        # 计算加权平均评分
        weighted_sum = 0
        similarity_sum = 0
        
        for rated_item in rated_items:
            # 获取物品之间的相似度
            if self.item_similarity_matrix is not None and item_id in self.item_similarity_matrix.columns:
                similarity = self.item_similarity_matrix.loc[rated_item, item_id]
                rating = user_ratings[rated_item]
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 0
            
        predicted_rating = weighted_sum / similarity_sum
        return predicted_rating
    
    def recommend_items_user_based(self, user_id, n_recommendations=5):
        """基于用户的协同过滤推荐"""
        if user_id not in self.user_item_matrix.index:
            return []
            
        # 获取用户未评分的物品
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        # 预测评分
        predictions = []
        for item_id in unrated_items[:100]:  # 限制计算数量以提高效率
            prediction = self.predict_user_based(user_id, item_id)
            predictions.append((item_id, prediction))
        
        # 按预测评分排序并返回前N个推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def recommend_items_item_based(self, user_id, n_recommendations=5):
        """基于物品的协同过滤推荐"""
        if user_id not in self.user_item_matrix.index:
            return []
            
        # 获取用户未评分的物品
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        # 预测评分
        predictions = []
        for item_id in unrated_items[:100]:  # 限制计算数量以提高效率
            prediction = self.predict_item_based(user_id, item_id)
            predictions.append((item_id, prediction))
        
        # 按预测评分排序并返回前N个推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# 完整测试示例
if __name__ == "__main__":
    # 创建一个更完整的测试矩阵
    test_data = {
        1: [5, 3, 0, 1, 4],
        2: [4, 0, 0, 1, 2],
        3: [1, 1, 0, 5, 3],
        4: [1, 0, 0, 4, 1],
        5: [0, 2, 5, 0, 0]
    }
    
    test_matrix = pd.DataFrame(test_data, index=[1, 2, 3, 4, 5])
    
    # 初始化推荐器
    recommender = CollaborativeFilteringRecommender(test_matrix)
    
    # 计算相似度矩阵
    user_sim = recommender.compute_user_similarity()
    item_sim = recommender.compute_item_similarity()
    
    print("\n用户相似度矩阵:")
    print(user_sim)
    
    print("\n物品相似度矩阵:")
    print(item_sim)
    
    # 测试预测功能
    prediction_user = recommender.predict_user_based(1, 3)
    prediction_item = recommender.predict_item_based(1, 3)
    
    print(f"\n基于用户的预测评分 (用户1对物品3): {prediction_user:.3f}")
    print(f"基于物品的预测评分 (用户1对物品3): {prediction_item:.3f}")
    
    # 测试推荐功能
    recommendations_user = recommender.recommend_items_user_based(1, 3)
    recommendations_item = recommender.recommend_items_item_based(1, 3)
    
    print(f"\n基于用户的推荐 (用户1): {recommendations_user}")
    print(f"基于物品的推荐 (用户1): {recommendations_item}")