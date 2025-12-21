# 个性化信息检索系统 (Personalized Information Retrieval System)

一个基于协同过滤算法的电影推荐系统，实现了用户个性化推荐功能，集成了最新的大语言模型增强推荐技术(PALR)。

## 目录结构

```
personalized_ir_system/
├── data/
│   └── ml-latest-small/     # MovieLens数据集
├── docs/                    # 文档目录
│   ├── technical_doc.md     # 技术文档
│   ├── user_guide.md        # 使用指南
│   └── lightgcn_documentation.md  # LightGCN算法文档
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── data_handler.py      # 数据处理模块
│   ├── recommender.py       # 传统协同过滤推荐算法模块
│   ├── lightgcn_recommender.py  # LightGCN推荐算法模块
│   ├── evaluator.py         # 评估模块
│   └── main.py              # 主程序入口
├── requirements.txt          # 依赖包列表
└── README.md                # 项目说明文档
```

## 功能特性

1. **个性化电影推荐**：基于用户历史评分数据，为用户推荐可能感兴趣的电影
2. **多种推荐算法**：
   - 基于用户的协同过滤 (User-Based Collaborative Filtering)
   - 基于物品的协同过滤 (Item-Based Collaborative Filtering)
   - 基于图卷积网络的推荐算法 (LightGCN)
   - 大语言模型增强推荐 (PALR - Prompt-Augmented Language Model for Recommendation)
3. **推荐效果评估**：提供RMSE、MAE等评估指标
4. **交互式用户界面**：基于Streamlit的Web界面，方便用户体验

## 技术实现

### 核心算法

1. **协同过滤**：
   - 用户相似度计算：使用余弦相似度计算用户之间的相似性
   - 物品相似度计算：使用余弦相似度计算电影之间的相似性
   - 评分预测：根据相似用户或相似物品的历史评分预测目标用户对未观看电影的评分

2. **LightGCN（简化图卷积网络）**：
   - 基于用户-物品二部图的图神经网络
   - 通过多层邻域聚合捕获高阶协同信号
   - 移除特征变换和非线性激活，只保留邻域聚合操作

3. **PALR（提示增强语言模型推荐）**：
   - 结合大语言模型与传统推荐算法
   - 利用LLM的语义理解能力增强推荐质量
   - 通过自然语言提示与LLM交互，优化推荐结果

3. **数据处理**：
   - 数据清洗与预处理
   - 用户-物品评分矩阵构建
   - 训练集与测试集划分

4. **评估指标**：
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - Precision, Recall, F1-Score (用于Top-N推荐评估)
   
   评估实现细节：
   - 评分预测评估：遍历测试数据集中的每一条记录，使用相应的推荐算法预测评分，并与实际评分进行比较计算RMSE和MAE指标
   - Top-N推荐评估：定义评分≥4的物品为用户感兴趣的内容，为测试用户生成推荐列表，计算Precision、Recall和F1-Score指标
   - 评估策略优化：限制测试用户数量(50个)和推荐计算范围(100个物品)以提高评估效率

## 安装与运行

### 环境要求

- Python 3.7+
- pip

### 安装步骤

1. 克隆项目到本地：
   ```bash
   git clone <repository-url>
   cd personalized_ir_system
   ```

2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

### 运行应用

```bash
streamlit run src/main.py
```

运行后，在浏览器中打开显示的地址即可访问推荐系统界面。

## 使用说明

1. 在界面左侧选择一个用户ID
2. 设置要推荐的电影数量
3. 点击"获取推荐"按钮
4. 系统将显示为您推荐的电影列表及其预测评分

## 项目模块介绍

### data_handler.py
负责数据加载和预处理：
- 读取MovieLens数据集
- 构建用户-物品评分矩阵
- 数据集划分

### recommender.py
实现传统协同过滤推荐算法：
- 用户相似度计算
- 物品相似度计算
- 评分预测

### lightgcn_recommender.py
实现LightGCN推荐算法：
- 图卷积网络模型
- 邻域聚合操作
- 评分预测

### palr_recommender.py
实现PALR推荐算法：
- 整合大语言模型与传统推荐算法
- 用户画像生成
- 推荐结果增强

### palr_prompts.py
PALR提示模板：
- 用户画像生成提示
- 推荐排序提示
- 推荐解释提示

### evaluator.py
评估推荐效果：
- 计算RMSE、MAE等指标
- Top-N推荐评估

### main.py
主程序文件，包含Streamlit界面实现。

### train_lightgcn.py
专门用于训练LightGCN模型的脚本，可将训练好的模型保存到磁盘供后续使用。

## 数据集

本项目使用MovieLens数据集的1M版本(ml-1m)，包含：
- 电影评分数据 (ratings.dat)
- 电影信息 (movies.dat)
- 用户信息 (users.dat)

## PALR功能使用说明

要使用PALR（提示增强语言模型推荐）功能：

1. 在Web界面左侧边栏选择"PALR增强"推荐方法
2. 输入用户ID和推荐数量
3. 点击"获取推荐"按钮
4. 系统将显示经过大语言模型增强后的推荐结果

注意：PALR功能需要访问外部的大语言模型服务，确保网络连接正常。

## 模型训练与使用

系统支持两种方式使用LightGCN模型：

1. **预训练模型**：系统会自动尝试加载已保存的预训练模型，如果存在则直接使用
2. **即时训练**：如果没有找到预训练模型，系统会在启动时训练一个新的模型

要训练并保存模型，请运行：
```bash
python train_lightgcn.py
```

这将在`models/`目录下创建模型文件，下次运行Web应用时将自动加载该模型。

## 评估结果

系统提供了两种协同过滤算法的评估结果：
- 基于用户的协同过滤
- 基于物品的协同过滤

通过RMSE和MAE指标可以比较不同算法的预测准确性。