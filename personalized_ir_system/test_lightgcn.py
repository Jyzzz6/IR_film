#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for LightGCN Recommender
This script tests the LightGCN implementation without using the Streamlit interface.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_handler import MovieLensDataHandler
from lightgcn_recommender import LightGCNRecommender

def test_lightgcn_recommender():
    """Test the LightGCN recommender implementation."""
    print("🎬 Testing LightGCN Recommender System")
    print("=" * 50)
    
    # Initialize data handler
    print("Loading MovieLens data...")
    data_handler = MovieLensDataHandler(
        ratings_path="data/ml-1m/ratings.dat",
        movies_path="data/ml-1m/movies.dat",
        dataset_type='ml-1m'
    )
    
    # Load data
    ratings, movies = data_handler.load_data()
    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies")
    
    # Preprocess data
    merged_data = data_handler.preprocess_data()
    print("Data preprocessing completed")
    
    # Split data
    train_data, test_data = data_handler.split_data()
    print(f"Split data: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Create user-item matrix
    user_item_matrix = data_handler.create_user_item_matrix(train_data)
    print(f"Created user-item matrix: {user_item_matrix.shape[0]} users × {user_item_matrix.shape[1]} items")
    
    # Initialize LightGCN recommender
    print("\nInitializing LightGCN model...")
    lightgcn_recommender = LightGCNRecommender(user_item_matrix, embedding_dim=32, n_layers=2)
    
    # Train the model
    print("Training LightGCN model...")
    lightgcn_recommender.train(epochs=10, verbose=True)  # Use fewer epochs for testing
    print("Training completed!")
    
    # Test recommendations for a sample user
    test_user_id = 1  # 1-based indexing in the dataset
    print(f"\nGenerating recommendations for user {test_user_id}...")
    
    try:
        # LightGCN uses 0-based indexing
        recommendations = lightgcn_recommender.recommend(test_user_id-1, top_k=5)
        print(f"Top 5 recommendations for user {test_user_id}:")
        
        # Get movie titles for recommended items
        for i, (item_id, score) in enumerate(recommendations):
            # Convert back to 1-based indexing for lookup
            movie_info = movies[movies['movieId'] == item_id+1]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                print(f"  {i+1}. {title} (Movie ID: {item_id+1}) - Score: {score:.4f}")
            else:
                print(f"  {i+1}. Movie ID: {item_id+1} - Score: {score:.4f}")
                
        print("\n✅ LightGCN recommender test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during recommendation generation: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_lightgcn_recommender()
    sys.exit(0 if success else 1)