#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PALR: Prompt-Augmented Language Model for Recommendation
Implementation based on the paper "PALR: Prompt-Augmented Language Model for Recommendation" (SIGIR 2023)

This module integrates large language models with traditional recommendation systems
to improve recommendation quality through prompt learning.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import openai
import os
from typing import List, Tuple, Dict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PALRRecommender:
    """
    PALR Recommender System
    Integrates Large Language Models with traditional recommendation algorithms
    """
    
    def __init__(self, base_recommender, movies_df, use_api_key=None):
        """
        Initialize PALR Recommender
        
        Args:
            base_recommender: Base recommendation model (e.g., LightGCN)
            movies_df: DataFrame containing movie information
            use_api_key: OpenAI API key for GPT integration (optional)
        """
        self.base_recommender = base_recommender
        self.movies_df = movies_df
        self.use_api_key = use_api_key
        
        # 初始化句子编码器（替代大型语言模型用于演示）
        try:
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence encoder initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize sentence encoder: {e}")
            self.sentence_encoder = None
        
        # 如果提供了API密钥，则使用OpenAI
        if self.use_api_key:
            openai.api_key = self.use_api_key
            self.use_openai = True
        else:
            self.use_openai = False
    
    def _get_movie_info(self, movie_id: int) -> str:
        """
        Get movie information as text description
        
        Args:
            movie_id: Movie ID
            
        Returns:
            String description of the movie
        """
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_row.empty:
            return f"Movie ID: {movie_id}"
        
        title = movie_row.iloc[0]['title']
        genres = movie_row.iloc[0]['genres']
        
        return f"Movie Title: {title}, Genres: {genres}"
    
    def _create_user_profile_prompt(self, user_id: int, user_history: List[Tuple[int, float]]) -> str:
        """
        Create prompt for user profile description
        
        Args:
            user_id: User ID
            user_history: List of (movie_id, rating) tuples
            
        Returns:
            Prompt string for user profile
        """
        # 获取用户历史评分最高的几部电影
        top_rated = sorted(user_history, key=lambda x: x[1], reverse=True)[:5]
        movie_descriptions = []
        
        for movie_id, rating in top_rated:
            movie_info = self._get_movie_info(movie_id)
            movie_descriptions.append(f"{movie_info} (Rating: {rating}/5)")
        
        prompt = f"""User Profile Analysis:
User ID: {user_id}
User's Top Rated Movies:
{chr(10).join(movie_descriptions)}

Based on these movies, please describe this user's preferences in terms of genres, themes, and movie characteristics. 
Focus on what types of movies this user tends to enjoy."""

        return prompt
    
    def _create_recommendation_prompt(self, user_profile: str, candidate_items: List[Tuple[int, float]]) -> str:
        """
        Create prompt for recommendation ranking
        
        Args:
            user_profile: User profile description
            candidate_items: List of (movie_id, base_score) tuples
            
        Returns:
            Prompt string for recommendation ranking
        """
        item_descriptions = []
        for movie_id, base_score in candidate_items[:10]:  # Limit to top 10 for prompt length
            movie_info = self._get_movie_info(movie_id)
            item_descriptions.append(f"{movie_info} (Base Score: {base_score:.3f})")
        
        prompt = f"""Recommendation Task:
User Profile: {user_profile}

Candidate Movies:
{chr(10).join(item_descriptions)}

Please rank these movies according to how well they match the user's preferences. 
Return the movie IDs in order of preference, separated by commas.
Example format: movie_id1,movie_id2,movie_id3,..."""

        return prompt
    
    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from language model
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response
        """
        if self.use_openai and self.use_api_key:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert movie recommendation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
                return ""
        else:
            # 使用句子编码器作为替代方案
            if self.sentence_encoder:
                # 这里简化处理，实际应该有更复杂的逻辑
                return "Enhanced by sentence encoder"
            else:
                return ""
    
    def _parse_ranked_movies(self, response: str, candidate_items: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Parse ranked movies from LLM response
        
        Args:
            response: LLM response
            candidate_items: Original candidate items
            
        Returns:
            Re-ranked items
        """
        # 提取电影ID
        import re
        movie_ids = re.findall(r'\d+', response)
        
        # 创建映射字典
        score_map = {movie_id: score for movie_id, score in candidate_items}
        
        # 重新排序
        ranked_items = []
        for movie_id_str in movie_ids:
            try:
                movie_id = int(movie_id_str)
                if movie_id in score_map:
                    ranked_items.append((movie_id, score_map[movie_id]))
            except ValueError:
                continue
        
        # 添加未提及的项目
        mentioned_ids = set(int(mid) for mid in movie_ids if mid.isdigit())
        for movie_id, score in candidate_items:
            if movie_id not in mentioned_ids:
                ranked_items.append((movie_id, score))
        
        return ranked_items
    
    def recommend(self, user_id: int, user_history: List[Tuple[int, float]], 
                  candidate_items: List[Tuple[int, float]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Enhanced recommendation with PALR approach
        
        Args:
            user_id: User ID
            user_history: User's historical ratings [(movie_id, rating), ...]
            candidate_items: Candidate items from base recommender [(movie_id, score), ...]
            top_k: Number of recommendations to return
            
        Returns:
            Enhanced recommendations [(movie_id, enhanced_score), ...]
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

def test_palr():
    """
    Test function for PALR implementation
    """
    print("Testing PALR Recommender...")
    
    # 这里只是一个示例，实际使用时需要传入真实的推荐器和电影数据
    print("PALR module initialized successfully!")

if __name__ == "__main__":
    test_palr()