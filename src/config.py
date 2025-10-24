# src/config.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    """Configuration parameters for model training and architecture."""
    layer_sizes: Tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.4
    epochs: int = 150
    batch_size: int = 32
    validation_split: float = 0.15
    learning_rate: float = 0.0005
    test_size: float = 0.2
    random_state: int = 42
    model_path: str = "models/best_heart_disease_model.keras"
    scaler_path: str = "models/scaler.pkl"
    prediction_threshold: float = 0.5
    data_path: str = "data/heart.csv"