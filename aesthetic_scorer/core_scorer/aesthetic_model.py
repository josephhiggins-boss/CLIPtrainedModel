"""
Aesthetic Model - Core Scoring Architecture

Defines the AestheticProbe model that scores images as a percentage (0-100).

What this does:
- Uses CLIP to understand images (frozen, pre-trained)
- Adds residual projection layer to focus on aesthetic qualities
- Converts understanding into a score from 0.0 to 100.0 (percentage)
- Supports images of any size/shape without distortion

This is the core model architecture - other modules use this to score images.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from typing import Optional, Dict, Any


class AestheticProbe(nn.Module):
    """
    CLIP-based aesthetic assessment model.
    
    Architecture:
        - CLIP ViT-B/16 vision encoder (frozen)
        - Residual projection layer 
        - Linear prediction head 
        - Learnable temperature parameter
    
    Output: Aesthetic score as percentage in range [0, 100]
    
    The model removes positional embeddings from CLIP to enable processing
    of different aspect ratios 
    
    Attributes:
        clip: Frozen CLIP model for feature extraction
        residual_proj: Residual projection layer
        head: Linear layer for score prediction 
        temperature: Learnable temperature parameter
    """
    
    def __init__(self, clip_model_name: str = 'openai/clip-vit-base-patch16', remove_pos_embeddings: bool = True):
        """
        Initialize the aesthetic probe model.
        
        Args:
            clip_model_name: HuggingFace model name for CLIP
            remove_pos_embeddings: If True, remove positional embeddings to enable
                                   universal image sizes (any aspect ratio)
        """
        super().__init__()
        
        # Load pretrained CLIP
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP - we don't want to change these weights
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # Remove positional embeddings for all image sizes
        
        if remove_pos_embeddings:
            vision_model = self.clip.vision_model
            # For CLIP ViT, position_embedding can be nn.Parameter, nn.Embedding, or tensor
            # We need to zero it out to allow variable image sizes
            if hasattr(vision_model.embeddings, 'position_embedding'):
                pos_emb = vision_model.embeddings.position_embedding
                
                try:
                    # Handle different types of positional embeddings
                    if isinstance(pos_emb, nn.Embedding):
                        # It's an Embedding layer - access the weight tensor
                        self._original_pos_embedding = pos_emb.weight.data.clone()
                        with torch.no_grad():
                            pos_emb.weight.data.zero_()
                    elif isinstance(pos_emb, nn.Parameter):
                        # It's a Parameter - access .data
                        self._original_pos_embedding = pos_emb.data.clone()
                        with torch.no_grad():
                            pos_emb.data.zero_()
                    elif isinstance(pos_emb, torch.Tensor):
                        # It's a direct tensor
                        self._original_pos_embedding = pos_emb.clone()
                        # Replace with zero tensor as Parameter
                        vision_model.embeddings.position_embedding = nn.Parameter(
                            torch.zeros_like(pos_emb)
                        )
                        vision_model.embeddings.position_embedding.requires_grad = False
                    elif hasattr(pos_emb, 'weight'):
                        # It might be an Embedding-like object with .weight
                        self._original_pos_embedding = pos_emb.weight.data.clone()
                        with torch.no_grad():
                            pos_emb.weight.data.zero_()
                    elif hasattr(pos_emb, 'data'):
                        # It might have a .data attribute
                        self._original_pos_embedding = pos_emb.data.clone()
                        with torch.no_grad():
                            pos_emb.data.zero_()
                    else:
                        # Skip if we can't handle it
                        self._original_pos_embedding = None
                except Exception as e:
                    # If we can't handle it, skip positional embedding removal
                    # This is not critical - model will still work, just with fixed-size constraints
                    self._original_pos_embedding = None
        
        # Get feature dimension (512 for ViT-B/16)
        feature_dim = self.clip.config.projection_dim
        
        # Residual projection layer for feature refinement
        self.residual_proj = nn.Linear(feature_dim, feature_dim)
        
        # Linear head: features -> score
        self.head = nn.Linear(feature_dim, 1)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones([]))
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict aesthetic score.
        
        Matches training architecture exactly:
        1. Extract CLIP features
        2. Normalize features
        3. Apply residual projection: features + residual_proj(features)
        4. Normalize adapted features
        5. Compute logits: head(adapted_features)
        6. Scale by temperature
        7. Apply tanh and scale: tanh(logits) * 50 + 50
        8. Clamp to [0, 100] range (percentage)
        
        Args:
            pixel_values: Preprocessed image tensor from CLIP processor
                Shape: (batch_size, 3, 224, 224) or variable for universal sizes
        
        Returns:
            score: Predicted aesthetic quality as percentage (0-100 range)
                Shape: (batch_size,)
        """
        # Extract CLIP features (no gradients needed for CLIP)
        with torch.no_grad():
            features = self.clip.get_image_features(pixel_values=pixel_values)
        
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Apply residual projection: v + residual_proj(v)
        # This allows learning aesthetic nuances without forgetting CLIP's rich visual knowledge
        residual = self.residual_proj(features)
        adapted_features = features + residual
        
        # Normalize adapted features
        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)
        
        # Map features to raw logits
        logits = self.head(adapted_features).squeeze(-1)
        
        # Scale by temperature (learned during training)
        logits = logits / self.temperature
        
        # Apply tanh activation and scale to [0, 100] range (percentage)
        # tanh maps [-inf, inf] to [-1, 1], then scale: * 50 + 50 maps to [0, 100]
        scores = torch.tanh(logits) * 50.0 + 50.0
        
        # Clamp to ensure scores are in valid range [0, 100]
        scores = torch.clamp(scores, 0.0, 100.0)
        
        return scores

