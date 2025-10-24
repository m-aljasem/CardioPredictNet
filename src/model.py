# src/model.py
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from config import ModelConfig



class TQDMProgressBar(Callback):
    """A custom Keras callback for a clean, single-line training progress bar."""
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.pbar = tqdm(total=self.epochs, desc="Training", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = {k: f"{v:.4f}" for k, v in logs.items()}
        self.pbar.set_postfix(metrics)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()



class HeartDiseasePredictor:
    """
    A neural network classifier for heart disease prediction.

    This class encapsulates the entire machine learning pipeline.
    """

    def _initialize_device_strategy(self):
        """
        Detects available hardware (TPU, GPU, CPU) and sets the appropriate
        TensorFlow distribution strategy. This makes training device-aware.
        """
        try:
            # 1. Attempt to connect to a TPU cluster
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            self.strategy = tf.distribute.TPUStrategy(tpu)
            print("✅ TPU found! Training will be distributed across the TPU.")
            print('🚀 Running on TPU ', tpu.master())

        except ValueError:
            # 2. If no TPU, check for GPUs
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # If GPUs are found, use MirroredStrategy for single or multi-GPU setups
                self.strategy = tf.distribute.MirroredStrategy()
                print(f"✅ GPU(s) found! Training will be accelerated on {len(gpus)} GPU(s).")
                for i, gpu in enumerate(gpus):
                    print(f'   -> GPU {i}: {gpu.name}')
            else:
                # 3. If no GPU, fall back to the default strategy (CPU)
                self.strategy = tf.distribute.get_strategy()
                print("⚠️ No TPU or GPU detected. Training will run on the CPU.")
                print("   For faster training, enable a GPU/TPU runtime in Colab.")

    def __init__(self, config: ModelConfig = ModelConfig()):
        """Initializes the predictor with a configuration object."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.strategy = None  # Add this line to initialize the attribute
        self._initialize_device_strategy() # Add this line to call the setup method

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Loads dataset from a CSV file."""
        data_path = Path(filepath)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        df = pd.read_csv(data_path)
        print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
        """
        Splits and preprocesses data for training, ensuring consistency
        with the features used by the Streamlit application.
        """
        # Define the exact feature columns the Streamlit app uses
        feature_columns = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]

        # Check if all required columns are in the dataframe
        if not all(col in df.columns for col in feature_columns + ['target']):
            raise ValueError(
                "The dataset is missing one of the required columns. "
                f"Required: {feature_columns + ['target']}"
            )

        print("✓ All required columns found in the dataset.")

        # Extract target variable (disease presence: 0 or 1)
        y = df['target'].values

        # Select ONLY the features that our Streamlit app will provide
        X_raw = df[feature_columns]

        # --- IMPORTANT: Feature Engineering ---
        # The training script MUST perform the same feature engineering as the app.
        # The Streamlit app does not have BMI, but the model might implicitly use it.
        # The original Kaggle dataset for this problem often has BMI engineered.
        # For simplicity and robustness, we will train on the 11 base features.
        # If you were to add BMI, you would do it here AND in the Streamlit app.

        X = X_raw.values

        # Split data with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y  # Ensure proportional class distribution
        )

        # Normalize features: critical for neural network convergence
        # Fit scaler on training data only to prevent data leakage
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)  # Use training statistics

        # The scaler will now be correctly fitted on 11 features
        print(f"✓ Scaler fitted on {self.scaler.n_features_in_} features.")
        print(f"✓ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test

    def build_model(self, input_dim: int) -> tf.keras.Model:
        """
        Constructs the Keras model architecture using the Functional API
        for maximum compatibility with conversion tools like tf2onnx.
        """
        # 1. Define the input layer. This is the entry point to the model graph.
        inputs = tf.keras.layers.Input(shape=(input_dim,), name='input_layer')

        # 2. Stack layers by calling them on the previous layer's output tensor.
        # The first hidden layer connects to the input tensor.
        x = Dense(self.config.layer_sizes[0], activation='relu', name='hidden_1')(inputs)
        x = Dropout(self.config.dropout_rate)(x)

        # Subsequent hidden layers connect to the output of the previous layer.
        x = Dense(self.config.layer_sizes[1], activation='relu', name='hidden_2')(x)
        x = Dropout(self.config.dropout_rate)(x)

        x = Dense(self.config.layer_sizes[2], activation='relu', name='hidden_3')(x)
        x = Dropout(self.config.dropout_rate)(x)

        # 3. Define the final output layer.
        outputs = Dense(1, activation='sigmoid', name='output')(x)

        # 4. Create the final Model object by specifying its inputs and outputs.
        # This creates a model with a clear, explicit graph structure.
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="HeartDiseaseClassifier")

        # The compilation step remains exactly the same.
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        print("✓ Model architecture built (Functional API)")
        model.summary()
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
            """Trains the model with callbacks."""
            # --- FIX IS HERE ---
            # Use the correct attribute 'model_path' instead of 'model_path_keras'
            model_dir = Path(self.config.model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)

            callbacks = [
                ModelCheckpoint(
                    # --- AND FIX IS HERE ---
                    self.config.model_path, # Use the correct attribute name
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True,
                    verbose=0
                ),
                EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
                TQDMProgressBar()
            ]

            self.model = self.build_model(X_train.shape[1])
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=0
            )

            print("✓ Training complete")
            self._save_scaler()

    def _save_scaler(self) -> None:
        """Saves the fitted StandardScaler to disk."""
        joblib.dump(self.scaler, self.config.scaler_path)
        print(f"✓ Scaler saved to: {self.config.scaler_path}")

    def load_trained_model(self) -> None:
        """Loads a pre-trained Keras model and scaler for inference."""
        # Use the correct attribute 'model_path' for the Keras model
        model_p = Path(self.config.model_path)
        scaler_p = Path(self.config.scaler_path)

        if not model_p.exists() or not scaler_p.exists():
            # Updated error message for clarity
            raise FileNotFoundError("Ensure both the Keras model (.keras) and scaler (.pkl) files exist before loading.")

        # --- AND FIX IS HERE ---
        # Load the Keras model directly, remove the call to the old ONNX loader
        self.model = load_model(model_p)
        self.scaler = joblib.load(scaler_p)
        print("✓ Trained Keras model and scaler loaded successfully")

            


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on test data, providing a comprehensive report
        with key classification metrics and visualizations.
        """
        if self.model is None:
            self.model = load_model(self.config.model_path)

        # Get model predictions and probabilities
        probabilities = self.model.predict(X_test)
        predictions = (probabilities > self.config.prediction_threshold).astype(int).flatten()

        # --- Calculate Core Metrics ---
        test_accuracy = np.mean(predictions == y_test)
        test_precision = precision_score(y_test, predictions)
        test_recall = recall_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions)

        # Calculate ROC Curve and AUC
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)

        # --- Print Comprehensive Report ---
        print("\n" + "="*60)
        print("      COMPREHENSIVE MODEL EVALUATION (TEST SET)")
        print("="*60)
        print(f"-> Test Accuracy:   {test_accuracy:.4f}")
        print(f"-> Test Precision:  {test_precision:.4f} (Disease class)")
        print(f"-> Test Recall:     {test_recall:.4f} (Disease class)")
        print(f"-> Test F1-Score:   {test_f1:.4f} (Disease class)")
        print(f"-> Test AUC Score:  {roc_auc:.4f}")
        print("-" * 60)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['No Disease', 'Disease']))
        print("="*60)

        # --- Generate Visualizations ---
        plt.figure(figsize=(18, 7))

        # 1. Confusion Matrix
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')

        # 2. ROC Curve
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.show()

        # Return a dictionary of all calculated metrics
        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'auc': roc_auc
        }


    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generates binary predictions and probabilities for new data."""
        if self.model is None:
            raise ValueError("Model not loaded. Call train() or load_trained_model() first.")

        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled)
        predictions = (probabilities > self.config.prediction_threshold).astype(int)

        # Return both values as a tuple
        return predictions, probabilities