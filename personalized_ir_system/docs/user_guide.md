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