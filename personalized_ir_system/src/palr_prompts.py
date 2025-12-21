#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PALR Prompt Templates
Templates for prompt-augmented recommendation with large language models
"""

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

# 推荐解释模板
EXPLANATION_TEMPLATE = """
Based on the user's preferences and the movie characteristics, please provide a brief explanation 
for why these movies are recommended to this user. Keep the explanation concise and informative.
"""

def create_user_profile_prompt(user_id: int, top_rated_movies: list) -> str:
    """
    Create prompt for user profile generation
    
    Args:
        user_id: User ID
        top_rated_movies: List of movie descriptions
        
    Returns:
        Formatted prompt string
    """
    movie_list_str = "\n".join([f"{i+1}. {movie}" for i, movie in enumerate(top_rated_movies)])
    return USER_PROFILE_TEMPLATE.format(user_id=user_id, movie_list=movie_list_str)

def create_ranking_prompt(user_profile: str, candidate_movies: list) -> str:
    """
    Create prompt for recommendation ranking
    
    Args:
        user_profile: User profile description
        candidate_movies: List of candidate movie descriptions
        
    Returns:
        Formatted prompt string
    """
    candidate_list_str = "\n".join([f"{i+1}. {movie}" for i, movie in enumerate(candidate_movies)])
    return RANKING_TEMPLATE.format(user_profile=user_profile, candidate_list=candidate_list_str)

def create_explanation_prompt(user_profile: str, recommended_movies: list) -> str:
    """
    Create prompt for recommendation explanation
    
    Args:
        user_profile: User profile description
        recommended_movies: List of recommended movie descriptions
        
    Returns:
        Formatted prompt string
    """
    recommended_list_str = "\n".join([f"{i+1}. {movie}" for i, movie in enumerate(recommended_movies)])
    return EXPLANATION_TEMPLATE.format(
        user_profile=user_profile, 
        recommended_list=recommended_list_str
    )