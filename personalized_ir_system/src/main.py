import streamlit as st
import pandas as pd
import numpy as np

# 导入我们的模块
from data_handler import MovieLensDataHandler
from recommender import CollaborativeFilteringRecommender

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
    # 初始化数据处理器
    data_handler = MovieLensDataHandler(
        ratings_path="/home/admin/myfile/buaa_myxxjs/film/personalized_ir_system/data/ml-latest-small/ratings.csv",
        movies_path="/home/admin/myfile/buaa_myxxjs/film/personalized_ir_system/data/ml-latest-small/movies.csv"
    )
    
    # 加载数据
    ratings, movies = data_handler.load_data()
    
    # 数据预处理
    merged_data = data_handler.preprocess_data()
    
    # 划分数据集
    train_data, test_data = data_handler.split_data()
    
    # 创建用户-物品矩阵
    user_item_matrix = data_handler.create_user_item_matrix(train_data)
    
    # 初始化推荐器
    recommender = CollaborativeFilteringRecommender(user_item_matrix)
    
    # 计算相似度矩阵
    with st.spinner('正在计算用户相似度矩阵...'):
        recommender.compute_user_similarity()
    
    with st.spinner('正在计算物品相似度矩阵...'):
        recommender.compute_item_similarity()
    
    return data_handler, recommender, movies

# 加载数据和模型
try:
    data_handler, recommender, movies_df = load_data_and_model()
except Exception as e:
    st.error(f"加载数据时出错: {str(e)}")
    st.stop()

# 侧边栏
st.sidebar.header("⚙️ 推荐设置")
user_id = st.sidebar.number_input("请输入用户ID", min_value=1, max_value=610, value=1)
n_recommendations = st.sidebar.slider("推荐数量", min_value=1, max_value=20, value=5)
method = st.sidebar.radio("推荐方法", ("基于用户", "基于物品"))

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
                    recommendations = recommender.recommend_items_user_based(user_id, n_recommendations)
                else:
                    recommendations = recommender.recommend_items_item_based(user_id, n_recommendations)
                
                if recommendations:
                    # 获取推荐电影的详细信息
                    recommended_movie_ids = [item_id for item_id, score in recommendations]
                    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
                    
                    # 合并推荐分数
                    recommended_movies = recommended_movies.copy()
                    score_dict = dict(recommendations)
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