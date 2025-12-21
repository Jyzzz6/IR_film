#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGCN Recommender System Implementation

This module implements the LightGCN algorithm for recommendation systems,
which is a simplified and effective graph convolutional network approach
for collaborative filtering.

Reference:
    He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Chua, T. S. (2020).
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
    arXiv preprint arXiv:2002.02126.

Author: Implementation based on the paper by Xiangnan He et al.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class LightGCN(nn.Module):
    """
    LightGCN Model for Recommendation
    
    LightGCN simplifies the design of GCN to make it more concise and appropriate
    for recommendation. It removes feature transformation and nonlinear activation
    in GCN, which makes it more efficient and effective.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        """
        Initialize LightGCN model
        
        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int): Embedding dimension
            n_layers (int): Number of GCN layers
        """
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
    def forward(self, adj_mat):
        """
        Forward propagation of LightGCN
        
        Args:
            adj_mat (torch.sparse.FloatTensor): Normalized adjacency matrix
            
        Returns:
            tuple: User embeddings and item embeddings for all layers
        """
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb])
        
        # Store embeddings from all layers
        embs = [all_emb]
        
        # Multi-layer propagation
        for _ in range(self.n_layers):
            all_emb = torch.spmm(adj_mat, all_emb)
            embs.append(all_emb)
            
        # Combine embeddings from all layers
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        # Split user and item embeddings
        user_final_emb, item_final_emb = torch.split(final_emb, [self.num_users, self.num_items])
        
        return user_final_emb, item_final_emb
    
    def predict(self, users, items):
        """
        Predict ratings for given user-item pairs
        
        Args:
            users (torch.LongTensor): User indices
            items (torch.LongTensor): Item indices
            
        Returns:
            torch.FloatTensor: Predicted ratings
        """
        with torch.no_grad():
            # Create adjacency matrix (this would normally be precomputed)
            # For simplicity in this implementation, we'll use a basic approach
            user_emb = self.user_embedding(users)
            item_emb = self.item_embedding(items)
            scores = torch.sum(user_emb * item_emb, dim=1)
            return scores


class LightGCNRecommender:
    """
    LightGCN Recommender System
    
    This class implements the LightGCN algorithm for collaborative filtering
    recommendation. It handles data processing, model training, and recommendation
    generation.
    """
    
    def __init__(self, user_item_matrix, embedding_dim=64, n_layers=3, learning_rate=0.001, weight_decay=1e-4):
        """
        Initialize LightGCN Recommender
        
        Args:
            user_item_matrix (pd.DataFrame): User-item interaction matrix
            embedding_dim (int): Embedding dimension
            n_layers (int): Number of GCN layers
            learning_rate (float): Learning rate for optimization
            weight_decay (float): Weight decay for regularization
        """
        self.user_item_matrix = user_item_matrix
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Get number of users and items
        self.num_users, self.num_items = user_item_matrix.shape
        
        # Initialize model
        self.model = LightGCN(self.num_users, self.num_items, embedding_dim, n_layers)
        
        # Create adjacency matrix
        self.adjacency_matrix = self._create_adjacency_matrix()
        
    def _create_adjacency_matrix(self):
        """
        Create normalized adjacency matrix for the bipartite graph
        
        Returns:
            torch.sparse.FloatTensor: Normalized adjacency matrix
        """
        # Convert user-item matrix to scipy sparse matrix
        ui_matrix = sp.coo_matrix(self.user_item_matrix.values)
        
        # Create adjacency matrix for the bipartite graph
        # A = [0, R; R^T, 0]
        adj_mat = sp.bmat([[None, ui_matrix], [ui_matrix.T, None]], format='coo')
        
        # Add self-connections
        adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        
        # Normalize adjacency matrix using symmetric normalization
        row_sum = np.array(adj_mat.sum(1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv).tocoo()
        
        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def train(self, epochs=100, verbose=True):
        """
        Train the LightGCN model
        
        Args:
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress
        """
        # Prepare training data
        users, items, ratings = [], [], []
        for user_idx in range(self.num_users):
            for item_idx in range(self.num_items):
                rating = self.user_item_matrix.iloc[user_idx, item_idx]
                if rating > 0:  # Only consider interactions with ratings
                    users.append(user_idx)
                    items.append(item_idx)
                    ratings.append(rating)
        
        users = torch.LongTensor(users)
        items = torch.LongTensor(items)
        ratings = torch.FloatTensor(ratings)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            user_embs, item_embs = self.model(self.adjacency_matrix)
            
            # Compute predictions
            user_features = user_embs[users]
            item_features = item_embs[items]
            predictions = torch.sum(user_features * item_features, dim=1)
            
            # Compute loss (MSE)
            loss = F.mse_loss(predictions, ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    def predict_user_ratings(self, user_id, top_k=10):
        """
        Predict ratings for all items for a given user
        
        Args:
            user_id (int): User ID
            top_k (int): Number of top items to recommend
            
        Returns:
            list: Top k recommended items with predicted ratings
        """
        # Get user embedding
        user_emb = self.model.user_embedding(torch.LongTensor([user_id]))
        
        # Get all item embeddings
        item_embs = self.model.item_embedding.weight
        
        # Compute scores for all items
        scores = torch.matmul(user_emb, item_embs.t()).squeeze()
        
        # Get top-k items
        top_items = torch.topk(scores, top_k)
        
        # Return item IDs and predicted scores
        return [(item_id.item(), score.item()) for item_id, score in zip(top_items.indices, top_items.values)]
    
    def recommend(self, user_id, top_k=10):
        """
        Generate recommendations for a user
        
        Args:
            user_id (int): User ID
            top_k (int): Number of recommendations to generate
            
        Returns:
            list: Recommended item IDs
        """
        # Check if user exists
        if user_id >= self.num_users:
            raise ValueError(f"User ID {user_id} not found in the system")
        
        # Get user's rated items
        user_rated_items = set(
            self.user_item_matrix.iloc[user_id][self.user_item_matrix.iloc[user_id] > 0].index
        )
        
        # Get predictions for all items
        predictions = self.predict_user_ratings(user_id, self.num_items)
        
        # Filter out already rated items
        recommendations = [
            (item_id, score) for item_id, score in predictions 
            if item_id not in user_rated_items
        ]
        
        # Return top-k recommendations
        return recommendations[:top_k]
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath)
        self.num_users = checkpoint['num_users']
        self.num_items = checkpoint['num_items']
        self.embedding_dim = checkpoint['embedding_dim']
        self.n_layers = checkpoint['n_layers']
        
        # Reinitialize model with loaded parameters
        self.model = LightGCN(self.num_users, self.num_items, self.embedding_dim, self.n_layers)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_info(self, filepath):
        """
        Save additional information needed for inference
        
        Args:
            filepath (str): Path to save the information
        """
        torch.save({
            'user_item_matrix_shape': self.user_item_matrix.shape,
            'adjacency_matrix': self.adjacency_matrix
        }, filepath)


def test_lightgcn():
    """
    Test function for LightGCN implementation
    """
    # Create sample data
    import pandas as pd
    import numpy as np
    
    # Sample user-item matrix (5 users, 10 items)
    np.random.seed(42)
    sample_data = np.random.randint(0, 6, size=(5, 10))  # Ratings from 0-5
    # Set some values to 0 to represent unrated items
    sample_data[sample_data < 2] = 0
    
    user_item_df = pd.DataFrame(sample_data)
    
    print("Sample User-Item Matrix:")
    print(user_item_df)
    print()
    
    # Initialize and train LightGCN recommender
    recommender = LightGCNRecommender(user_item_df, embedding_dim=32, n_layers=2)
    
    print("Training LightGCN model...")
    recommender.train(epochs=50, verbose=False)
    print("Training completed!")
    print()
    
    # Generate recommendations for user 0
    user_id = 0
    recommendations = recommender.recommend(user_id, top_k=5)
    
    print(f"Top 5 recommendations for user {user_id}:")
    for item_id, score in recommendations:
        print(f"  Item {item_id}: Predicted rating = {score:.3f}")


if __name__ == "__main__":
    test_lightgcn()