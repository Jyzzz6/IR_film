import streamlit as st
import pandas as pd
import numpy as np
import os

# 导入我们的模块
from data_handler import MovieLensDataHandler
from recommender import CollaborativeFilteringRecommender
from lightgcn_recommender import LightGCNRecommender

# 设置页面配置
st.set_page_config(
    page_title="个性化电影推荐系统",
    page_icon="🎬",
    layout="wide"
)

# 页面标题
st.title("🎬 个性化电影推荐系统")
st.markdown("---")

# 初始化数据处理器和推荐器
@st.cache_resource
def load_data_and_model():
    # 初始化数据处理器，使用ml-1m数据集
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
    
    # 初始化传统协同过滤推荐器
    cf_recommender = CollaborativeFilteringRecommender(user_item_matrix)
    
    # 计算相似度矩阵
    with st.spinner('正在计算用户相似度矩阵...'):
        cf_recommender.compute_user_similarity()
    
    with st.spinner('正在计算物品相似度矩阵...'):
        cf_recommender.compute_item_similarity()
    
    # 初始化LightGCN推荐器
    with st.spinner('正在初始化LightGCN模型...'):
        lightgcn_recommender = LightGCNRecommender(user_item_matrix)
        
        # 尝试加载预训练模型
        model_path = "models/lightgcn_ml1m.pth"
        if os.path.exists(model_path):
            try:
                with st.spinner('正在加载预训练模型...'):
                    lightgcn_recommender.load_model(model_path)
                st.success("成功加载预训练模型!")
            except Exception as e:
                st.warning(f"加载预训练模型失败: {str(e)}，将重新训练模型...")
                # 训练LightGCN模型
                with st.spinner('正在训练LightGCN模型...'):
                    lightgcn_recommender.train(epochs=50)  # 使用50个epoch进行训练
        else:
            # 训练LightGCN模型
            with st.spinner('正在训练LightGCN模型...'):
                lightgcn_recommender.train(epochs=50)  # 使用50个epoch进行训练
    
    return data_handler, cf_recommender, lightgcn_recommender, movies

# 加载数据和模型
try:
    data_handler, cf_recommender, lightgcn_recommender, movies_df = load_data_and_model()
except Exception as e:
    st.error(f"加载数据时出错: {str(e)}")
    st.stop()

# 侧边栏
st.sidebar.header("⚙️ 推荐设置")
user_id = st.sidebar.number_input("请输入用户ID", min_value=1, max_value=6040, value=1)
n_recommendations = st.sidebar.slider("推荐数量", min_value=1, max_value=20, value=5)
method = st.sidebar.radio("推荐方法", ("基于用户", "基于物品", "LightGCN"))

# 主要内容区域
col1, col2 = st.columns(2)

with col1:
    st.header("👤 用户信息")
    st.write(f"用户ID: {user_id}")
    
    # 显示用户的历史评分
    st.subheader("历史评分")
    user_ratings = data_handler.ratings_df[data_handler.ratings_df['userId'] == user_id].merge(
        movies_df, on='movieId'
    ).sort_values('rating', ascending=False)
    
    if not user_ratings.empty:
        st.dataframe(
            user_ratings[['title', 'rating', 'genres']].head(10),
            use_container_width=True
        )
    else:
        st.info("该用户暂无评分记录")

with col2:
    st.header("🍿 电影推荐")
    
    # 生成推荐
    if st.button("🔍 获取推荐", type="primary"):
        with st.spinner('正在生成推荐...'):
            try:
                if method == "基于用户":
                    recommendations = cf_recommender.recommend_items_user_based(user_id, n_recommendations)
                elif method == "基于物品":
                    recommendations = cf_recommender.recommend_items_item_based(user_id, n_recommendations)
                else:  # LightGCN
                    recommendations = lightgcn_recommender.recommend(user_id-1, n_recommendations)  # LightGCN uses 0-based indexing
                
                if recommendations:
                    # 获取推荐电影的详细信息
                    if method == "LightGCN":
                        # LightGCN返回的是(item_id, score)元组
                        recommended_movie_ids = [item_id+1 for item_id, score in recommendations]  # Convert back to 1-based indexing
                        score_dict = {item_id+1: score for item_id, score in recommendations}
                    else:
                        # 传统方法返回的是(item_id, score)元组
                        recommended_movie_ids = [item_id for item_id, score in recommendations]
                        score_dict = dict(recommendations)
                    
                    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
                    
                    # 合并推荐分数
                    recommended_movies = recommended_movies.copy()
                    recommended_movies['推荐分数'] = recommended_movies['movieId'].map(score_dict)
                    recommended_movies = recommended_movies.sort_values('推荐分数', ascending=False)
                    
                    st.subheader(f"为您推荐的电影 ({method})")
                    st.dataframe(
                        recommended_movies[['title', 'genres', '推荐分数']],
                        use_container_width=True
                    )
                else:
                    st.warning("暂无推荐结果，请尝试其他用户ID或其他推荐方法")
                    
            except Exception as e:
                st.error(f"生成推荐时出错: {str(e)}")
    else:
        st.info("点击上方按钮获取个性化推荐")

# 数据集统计信息
st.markdown("---")
st.header("📊 数据集统计")
col3, col4, col5, col6 = st.columns(4)

with col3:
    st.metric("用户总数", data_handler.ratings_df['userId'].nunique())
with col4:
    st.metric("电影总数", data_handler.ratings_df['movieId'].nunique())
with col5:
    st.metric("评分总数", len(data_handler.ratings_df))
with col6:
    st.metric("评分范围", f"{data_handler.ratings_df['rating'].min()} - {data_handler.ratings_df['rating'].max()}")

# 说明信息
st.markdown("---")
st.header("ℹ️ 使用说明")
st.markdown("""
1. 在左侧边栏输入用户ID（1-610）
2. 选择推荐数量和推荐方法
3. 点击"获取推荐"按钮查看个性化推荐结果
4. 可以查看用户的历史评分记录

本系统基于MovieLens数据集，使用协同过滤算法实现个性化推荐。
""")