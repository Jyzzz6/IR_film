# 个性化信息检索系统 - 使用指南

## 1. 系统安装

### 1.1 环境准备

确保您的计算机上已安装Python 3.7或更高版本。

### 1.2 下载项目

从代码仓库下载项目代码到本地：

```bash
git clone <repository-url>
cd personalized_ir_system
```

### 1.3 安装依赖

使用pip安装项目所需的依赖包：

```bash
pip install -r requirements.txt
```

## 2. 数据准备

系统已经包含了MovieLens小型数据集，无需额外下载。数据集包含以下文件：
- ratings.csv: 用户对电影的评分数据
- movies.csv: 电影基本信息
- tags.csv: 用户对电影的标签

## 3. 启动系统

在项目根目录下执行以下命令启动推荐系统：

```bash
streamlit run src/main.py
```

启动成功后，系统会在控制台输出类似以下信息：
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

在浏览器中打开显示的地址即可访问系统。

## 4. 界面操作说明

### 4.1 主界面

系统启动后会显示主界面，包含以下组件：
1. 侧边栏：用户设置区域
2. 主显示区：推荐结果展示区域

### 4.2 获取推荐

1. 在侧边栏的"选择用户ID"下拉框中选择一个用户
2. 在"推荐电影数量"输入框中设置希望获得的推荐电影数量
3. 点击"获取推荐"按钮

### 4.3 查看推荐结果

推荐结果将以表格形式显示，包含以下信息：
- 电影标题
- 电影类型
- 预测评分

评分越高表示用户可能越喜欢这部电影。

## 5. 算法选择

系统支持两种推荐算法：
- 基于用户的协同过滤
- 基于物品的协同过滤

可以通过修改源代码中的参数来切换算法。

## 6. 系统评估

系统内置了评估模块，可以计算推荐算法的准确性和效果。评估结果包括：
- RMSE (均方根误差)
- MAE (平均绝对误差)
- Precision, Recall, F1-Score (用于Top-N推荐评估)

### 6.1 评估指标说明

**评分预测评估：**
- RMSE：衡量预测评分与实际评分的差异，值越小表示预测越准确
- MAE：平均绝对误差，另一种衡量预测准确性的指标

**Top-N推荐评估：**
- Precision：推荐结果中相关项目的比例，值越高表示推荐结果越精准
- Recall：相关项目中被成功推荐的比例，值越高表示推荐覆盖越全面
- F1-Score：Precision和Recall的调和平均数，综合衡量推荐效果

### 6.2 评估实现细节

评估模块实现了完整的评估流程：

**评分预测评估实现：**
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
        
        return {'rmse': rmse, 'mae': mae, 'count': len(predictions)}
    else:
        return {'rmse': 0, 'mae': 0, 'count': 0}
```

**Top-N推荐评估实现：**
```python
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

### 6.3 评估策略与优化

为了提高评估效率和准确性，系统采用了多种优化策略：

1. **数据集划分策略**：采用时间序列划分确保评估的合理性
2. **计算效率优化**：限制测试用户数量为50个，以减少计算时间；在推荐计算时限制物品范围为100个，避免计算所有物品
3. **正向反馈定义**：将评分≥4的物品定义为用户感兴趣的内容
4. **多指标综合评估**：结合评分预测和Top-N推荐指标全面评估算法性能

### 6.4 评估技术细节

1. **RMSE计算稳定性**：通过检查用户和物品是否在训练集中来确保预测的有效性
2. **Precision/Recall平衡**：使用F1-Score综合衡量推荐效果，特别适用于不平衡数据集
3. **评估结果可视化**：提供详细的评估日志输出，便于调试和分析

如需查看评估结果，请参考技术文档或直接运行评估代码。

## 7. 故障排除

### 7.1 无法导入模块

如果遇到类似以下错误：
```
ModuleNotFoundError: No module named 'pandas'
```

请确保已正确安装所有依赖包：
```bash
pip install -r requirements.txt
```

### 7.2 数据文件缺失

如果提示找不到数据文件，请确认以下文件存在：
- data/ml-latest-small/ratings.csv
- data/ml-latest-small/movies.csv

### 7.3 端口冲突

如果8501端口已被占用，Streamlit会自动选择其他可用端口，请注意查看启动时的输出信息。

## 8. 自定义配置

### 8.1 修改数据路径

如需使用其他数据集，请修改`src/main.py`中的数据路径：
```python
data_handler = MovieLensDataHandler(
    ratings_path="your_ratings_file_path",
    movies_path="your_movies_file_path"
)
```

### 8.2 调整推荐参数

可以在代码中调整推荐算法的相关参数，例如相似度计算方法、邻居用户数量等。

## 9. 扩展开发

### 9.1 添加新算法

要添加新的推荐算法，请按照以下步骤操作：
1. 在`src/recommender.py`中实现新算法
2. 在`src/main.py`中调用新算法
3. 更新UI界面以支持新算法的选择

### 9.2 添加新评估指标

要在评估模块中添加新的评估指标，请修改`src/evaluator.py`文件。

## 10. 联系支持

如有任何问题或建议，请联系项目维护者。