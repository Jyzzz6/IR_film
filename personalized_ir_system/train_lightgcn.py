#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and save LightGCN model
This script trains the LightGCN model on the MovieLens-1M dataset and saves the trained model.
"""

import sys
import os
import torch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_handler import MovieLensDataHandler
from lightgcn_recommender import LightGCNRecommender

def train_and_save_model():
    """Train the LightGCN model and save it to disk."""
    print("🎬 Training LightGCN Model on MovieLens-1M Dataset")
    print("=" * 50)
    
    # Initialize data handler
    print("Loading MovieLens-1M data...")
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
    lightgcn_recommender = LightGCNRecommender(user_item_matrix, embedding_dim=64, n_layers=3, learning_rate=0.001)
    
    # Train the model
    print("Training LightGCN model...")
    print("This may take several minutes depending on your system performance.")
    lightgcn_recommender.train(epochs=100, verbose=True)
    print("Training completed!")
    
    # Save the trained model
    model_save_path = "models/lightgcn_ml1m.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"\nSaving model to {model_save_path}...")
    lightgcn_recommender.save_model(model_save_path)
    print("Model saved successfully!")
    
    # Save additional information needed for inference
    info_save_path = "models/lightgcn_info.pth"
    lightgcn_recommender.save_info(info_save_path)
    print(f"Model information saved to {info_save_path}")
    
    print("\n✅ Training and saving completed successfully!")
    print("\nTo use the trained model in the web application, simply run:")
    print("streamlit run src/main.py")
    
    return True

if __name__ == "__main__":
    try:
        success = train_and_save_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        sys.exit(1)