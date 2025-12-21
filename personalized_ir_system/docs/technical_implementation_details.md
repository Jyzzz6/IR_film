# 个性化信息检索系统 - 技术细节与实现要点

## 1. 系统整体架构

### 1.1 模块化设计
系统采用模块化设计，各模块职责明确，便于维护和扩展：

1. **数据处理模块 (data_handler.py)**：
   - 负责MovieLens数据集的加载和预处理
   - 构建用户-物品评分矩阵
   - 数据集划分功能

2. **传统协同过滤推荐模块 (recommender.py)**：
   - 实现基于用户和基于物品的协同过滤算法
   - 相似度计算和评分预测

3. **LightGCN推荐模块 (lightgcn_recommender.py)**：
   - 基于图卷积网络的推荐算法实现
   - 模型训练、保存和加载功能

4. **PALR增强推荐模块 (palr_recommender.py & palr_prompts.py)**：
   - 结合大语言模型与传统推荐算法
   - 用户画像生成和推荐结果优化

5. **评估模块 (evaluator.py)**：
   - RMSE、MAE等评估指标计算

6. **用户界面 (main.py)**：
   - 基于Streamlit的交互式Web界面

### 1.2 数据流设计
```
MovieLens数据集 → 数据处理模块 → 用户-物品评分矩阵 → 各推荐算法模块 → 推荐结果 → 用户界面展示
                              ↓
                        模型训练与保存
```

## 2. 核心算法实现细节

### 2.1 协同过滤算法

#### 2.1.1 基于用户的协同过滤
核心实现代码片段：
```python
def predict_user_based(self, user_id, item_id):
    """基于用户的协同过滤预测评分"""
    # 计算目标用户与其他用户的相似度
    user_similarities = self.user_similarity_matrix.loc[user_id]
    
    # 获取对目标物品评过分的用户
    item_raters = self.user_item_matrix[self.user_item_matrix[item_id] > 0].index
    
    # 计算加权平均评分
    weighted_sum = 0
    similarity_sum = 0
    
    for rater in item_raters:
        similarity = user_similarities[rater]
        rating = self.user_item_matrix.loc[rater, item_id]
        weighted_sum += similarity * rating
        similarity_sum += abs(similarity)
    
    if similarity_sum == 0:
        return 0
        
    predicted_rating = weighted_sum / similarity_sum
    return predicted_rating
```

#### 2.1.2 基于物品的协同过滤
核心实现代码片段：
```python
def predict_item_based(self, user_id, item_id):
    """基于物品的协同过滤预测评分"""
    # 获取用户已评分的物品
    user_ratings = self.user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index.tolist()
    
    # 计算加权平均评分
    weighted_sum = 0
    similarity_sum = 0
    
    for rated_item in rated_items:
        # 获取物品之间的相似度
        similarity = self.item_similarity_matrix.loc[rated_item, item_id]
        rating = user_ratings[rated_item]
        weighted_sum += similarity * rating
        similarity_sum += abs(similarity)
    
    if similarity_sum == 0:
        return 0
        
    predicted_rating = weighted_sum / similarity_sum
    return predicted_rating
```

### 2.2 LightGCN算法

#### 2.2.1 模型结构
```python
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 初始化嵌入
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
```

#### 2.2.2 邻域聚合实现
```python
def forward(self, edge_index):
    # 构建消息传递网络
    propagator = GCNConv(-1, -1, normalize=False, add_self_loops=False)
    
    # 初始化嵌入
    user_emb = self.user_embedding.weight
    item_emb = self.item_embedding.weight
    embeddings = torch.cat([user_emb, item_emb])
    
    # 多层邻域聚合
    all_embeddings = [embeddings]
    for _ in range(self.n_layers):
        embeddings = propagator(embeddings, edge_index)
        all_embeddings.append(embeddings)
    
    # 对所有层的嵌入进行加权平均
    all_embeddings = torch.stack(all_embeddings, dim=1)
    final_embeddings = torch.mean(all_embeddings, dim=1)
    
    # 分离用户和物品嵌入
    user_final_emb, item_final_emb = torch.split(
        final_embeddings, [self.num_users, self.num_items]
    )
    
    return user_final_emb, item_final_emb
```

#### 2.2.3 模型训练与持久化
```python
def train_model(self, epochs=100, batch_size=1024):
    """训练LightGCN模型"""
    # 准备训练数据
    train_edge_index = self._prepare_train_data()
    
    # 优化器
    optimizer = torch.optim.Adam(
        self.model.parameters(), 
        lr=self.learning_rate, 
        weight_decay=self.weight_decay
    )
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for batch in self._generate_batches(train_edge_index, batch_size):
            optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

def save_model(self, filepath):
    """保存模型"""
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'num_users': self.num_users,
        'num_items': self.num_items,
        'embedding_dim': self.embedding_dim,
        'n_layers': self.n_layers
    }, filepath)

def load_model(self, filepath):
    """加载模型"""
    checkpoint = torch.load(filepath)
    self.model.load_state_dict(checkpoint['model_state_dict'])
```

### 2.3 PALR算法

#### 2.3.1 核心推荐逻辑
```python
def recommend(self, user_id: int, user_history: List[Tuple[int, float]], 
              candidate_items: List[Tuple[int, float]], top_k: int = 10) -> List[Tuple[int, float]]:
    """
    Enhanced recommendation with PALR approach
    """
    # 第一步：生成用户画像
    user_profile_prompt = self._create_user_profile_prompt(user_id, user_history)
    user_profile = self._get_llm_response(user_profile_prompt)
    
    if not user_profile:
        logger.warning("Failed to generate user profile, returning base recommendations")
        return sorted(candidate_items, key=lambda x: x[1], reverse=True)[:top_k]
    
    # 第二步：使用用户画像重新排序候选项目
    ranking_prompt = self._create_recommendation_prompt(user_profile, candidate_items)
    ranking_response = self._get_llm_response(ranking_prompt)
    
    if not ranking_response:
        logger.warning("Failed to get ranking response, returning base recommendations")
        return sorted(candidate_items, key=lambda x: x[1], reverse=True)[:top_k]
    
    # 第三步：解析并重新排序
    enhanced_ranking = self._parse_ranked_movies(ranking_response, candidate_items)
    
    # 返回前K个推荐
    return enhanced_ranking[:top_k]
```

#### 2.3.2 提示模板设计
```python
# 用户画像生成模板
USER_PROFILE_TEMPLATE = """
User Profile Analysis:
User ID: {user_id}
User's Top Rated Movies:
{movie_list}

Based on these movies, please describe this user's preferences in terms of genres, themes, and movie characteristics. 
Focus on what types of movies this user tends to enjoy.
"""

# 推荐排序模板
RANKING_TEMPLATE = """
Recommendation Task:
User Profile: {user_profile}

Candidate Movies:
{candidate_list}

Please rank these movies according to how well they match the user's preferences. 
Return the movie IDs in order of preference, separated by commas.
Example format: movie_id1,movie_id2,movie_id3,...
"""
```

## 3. 性能优化策略

### 3.1 相似度矩阵缓存
为了避免重复计算用户/物品相似度，系统采用缓存机制：
```python
@st.cache_resource
def compute_user_similarity_cached(user_item_matrix):
    """缓存用户相似度矩阵计算结果"""
    recommender = CollaborativeFilteringRecommender(user_item_matrix)
    return recommender.compute_user_similarity()
```

### 3.2 稀疏矩阵存储
用户-物品评分矩阵采用稀疏矩阵存储，节省内存空间：
```python
# 使用scipy.sparse构造稀疏矩阵
sparse_matrix = sp.csr_matrix((ratings, (users, items)), shape=(num_users, num_items))
```

### 3.3 推荐结果限制
为提高计算效率，限制推荐计算的数量：
```python
# 限制计算数量以提高效率
for item_id in unrated_items[:100]:
    prediction = self.predict_user_based(user_id, item_id)
    predictions.append((item_id, prediction))
```

## 4. 系统集成与部署

### 4.1 模型持久化机制
系统实现了完整的模型训练、保存和加载流程：
1. 训练脚本独立运行：`train_lightgcn.py`
2. 模型自动保存到`models/`目录
3. Web应用启动时自动检测并加载预训练模型

### 4.2 大语言模型集成
PALR模块通过以下方式集成大语言模型：
1. 使用SentenceTransformer作为本地替代方案
2. 支持OpenAI API作为云端方案
3. 实现降级机制，在无网络时使用备用方案

### 4.3 用户界面集成
Streamlit界面通过以下方式集成各推荐算法：
```python
if method == "PALR增强":
    # 获取LightGCN的基础推荐
    base_recommendations = lightgcn_recommender.recommend(user_id-1, n_recommendations*2)
    
    # 获取用户历史评分
    user_history = []
    user_ratings = data_handler.ratings_df[data_handler.ratings_df['userId'] == user_id]
    for _, row in user_ratings.iterrows():
        user_history.append((row['movieId'], row['rating']))
    
    # 使用PALR增强推荐
    recommendations = palr_recommender.recommend(
        user_id, user_history, base_recommendations, n_recommendations
    )
```

## 5. 技术挑战与解决方案

### 5.1 依赖管理
**挑战**：不同算法需要不同的Python库，版本兼容性复杂
**解决方案**：精心管理requirements.txt，确保所有依赖正确安装

### 5.2 大语言模型集成
**挑战**：需要网络连接访问外部LLM服务，且响应时间较长
**解决方案**：实现降级机制，在无网络时使用备用方案

### 5.3 性能优化
**挑战**：用户基数较大时，基于用户的协同过滤计算耗时较长
**解决方案**：提供基于物品的协同过滤作为高性能替代方案

## 6. 系统扩展性

### 6.1 新算法集成
系统采用模块化设计，添加新算法只需：
1. 在`src/`目录下创建新算法实现文件
2. 在`main.py`中导入并调用新算法
3. 更新UI界面以支持新算法选择

### 6.2 数据集扩展
系统支持多种数据集格式，扩展新数据集只需：
1. 实现对应的数据加载函数
2. 调整数据预处理逻辑
3. 更新用户-物品矩阵构建过程

## 7. 评估与测试

### 7.1 离线测试
系统提供了完整的测试脚本：
- `test_lightgcn.py`：测试LightGCN功能
- `test_palr.py`：测试PALR在线功能
- `test_palr_offline.py`：测试PALR离线功能

### 7.2 性能评估
系统实现了多种评估指标：
- RMSE、MAE用于评分预测评估
- Precision、Recall、F1-Score用于Top-N推荐评估

#### 7.2.1 评估实现细节
```python
def evaluate_predictions(self, method='user_based'):
    """评估推荐系统的预测准确性"""
    predictions = []
    actuals = []
    
    # 遍历测试数据
    for _, row in self.test_data.iterrows():
        user_id = row['userId']
        item_id = row['movieId']
        actual_rating = row['rating']
        
        # 预测评分
        if method == 'user_based':
            predicted_rating = self.recommender.predict_user_based(user_id, item_id)
        else:
            predicted_rating = self.recommender.predict_item_based(user_id, item_id)
        
        # 只考虑有效的预测
        if predicted_rating > 0:
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
    
    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    return {'rmse': rmse, 'mae': mae, 'count': len(predictions)}

def evaluate_top_n_recommendations(self, n=10, method='user_based'):
    """评估Top-N推荐的准确性"""
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
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': f1_score
    }
```

#### 7.2.3 评估策略与优化
为了提高评估效率和准确性，我们采用了多种优化策略：

1. **数据集划分**：采用时间序列划分确保评估的合理性
2. **计算效率优化**：限制测试用户数量(50个)和推荐计算范围(100个物品)以提高评估效率
3. **正向反馈定义**：将评分≥4的物品定义为用户感兴趣的内容
4. **多指标综合评估**：结合评分预测和Top-N推荐指标全面评估算法性能

#### 7.2.4 评估技术细节
1. **RMSE计算稳定性**：通过检查用户和物品是否在训练集中来确保预测的有效性
2. **Precision/Recall平衡**：使用F1-Score综合衡量推荐效果，特别适用于不平衡数据集
3. **评估结果可视化**：提供详细的评估日志输出，便于调试和分析

#### 7.2.5 综合评估实现
```python
# 综合评估函数
def comprehensive_evaluation(self):
    """综合评估所有算法的性能"""
    methods = ['user_based', 'item_based']
    results = {}
    
    for method in methods:
        # 评分预测评估
        pred_metrics = self.evaluate_predictions(method)
        
        # Top-N推荐评估
        topn_metrics = self.evaluate_top_n_recommendations(10, method)
        
        results[method] = {
            'prediction': pred_metrics,
            'topn': topn_metrics
        }
    
    return results
```