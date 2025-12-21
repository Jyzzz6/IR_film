import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RecommendationEvaluator:
    def __init__(self, recommender, test_data, user_item_matrix):
        self.recommender = recommender
        self.test_data = test_data
        self.user_item_matrix = user_item_matrix
        
    def evaluate_predictions(self, method='user_based'):
        """
        评估推荐系统的预测准确性
        method: 'user_based' 或 'item_based'
        """
        predictions = []
        actuals = []
        
        # 遍历测试数据
        for _, row in self.test_data.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            actual_rating = row['rating']
            
            # 检查用户和物品是否在训练集中
            if user_id in self.user_item_matrix.index and item_id in self.user_item_matrix.columns:
                # 预测评分
                if method == 'user_based':
                    predicted_rating = self.recommender.predict_user_based(user_id, item_id)
                else:
                    predicted_rating = self.recommender.predict_item_based(user_id, item_id)
                
                # 只考虑有效的预测（非零预测）
                if predicted_rating > 0:
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
        
        # 计算评估指标
        if len(predictions) > 0:
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals, predictions)
            
            print(f"{method} 方法评估结果:")
            print(f"  预测数量: {len(predictions)}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            
            return {'rmse': rmse, 'mae': mae, 'count': len(predictions)}
        else:
            print(f"{method} 方法没有有效的预测结果")
            return {'rmse': 0, 'mae': 0, 'count': 0}
    
    def evaluate_top_n_recommendations(self, n=10, method='user_based'):
        """
        评估Top-N推荐的准确性
        """
        precision_list = []
        recall_list = []
        
        # 获取测试数据中的用户列表
        test_users = self.test_data['userId'].unique()
        
        for user_id in test_users[:50]:  # 限制用户数量以提高计算效率
            # 获取用户在测试集中的正向反馈（评分>=4）
            user_test_data = self.test_data[(self.test_data['userId'] == user_id) & 
                                           (self.test_data['rating'] >= 4)]
            relevant_items = set(user_test_data['movieId'].tolist())
            
            if len(relevant_items) == 0:
                continue
                
            # 获取推荐结果
            if method == 'user_based':
                recommendations = self.recommender.recommend_items_user_based(user_id, n)
            else:
                recommendations = self.recommender.recommend_items_item_based(user_id, n)
            
            # 提取推荐的物品ID
            recommended_items = set([item_id for item_id, score in recommendations])
            
            # 计算Precision和Recall
            if len(recommended_items) > 0:
                # 正确推荐的物品数
                relevant_recommended = len(relevant_items.intersection(recommended_items))
                
                precision = relevant_recommended / len(recommended_items)
                recall = relevant_recommended / len(relevant_items)
                
                precision_list.append(precision)
                recall_list.append(recall)
        
        # 计算平均Precision和Recall
        avg_precision = np.mean(precision_list) if precision_list else 0
        avg_recall = np.mean(recall_list) if recall_list else 0
        
        # 计算F1分数
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0
        
        print(f"{method} Top-{n} 推荐评估结果:")
        print(f"  平均 Precision: {avg_precision:.4f}")
        print(f"  平均 Recall: {avg_recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score
        }

# 示例使用
if __name__ == "__main__":
    # 这里需要与data_handler和recommender模块集成
    # 为了演示，我们创建一些模拟数据
    print("推荐系统评估模块")
    print("该模块用于评估推荐算法的性能，包括:")
    print("1. 预测评分准确性 (RMSE, MAE)")
    print("2. Top-N推荐准确性 (Precision, Recall, F1)")