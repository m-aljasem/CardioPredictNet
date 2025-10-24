# -*- coding: utf-8 -*-
"""
CardioPredictNet AI: Advanced Cardiovascular Risk Assessment
=========================================================

A sophisticated, interactive web application for cardiovascular disease risk
assessment using a deep learning model.

This Streamlit application provides an elegant and intuitive interface for
healthcare professionals and individuals to evaluate heart disease risk.

Technology Stack: Streamlit, TensorFlow/Keras, Plotly, Scikit-learn
Design Philosophy: Professional, data-driven, and visually compelling.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Tuple
from config import ModelConfig
from model import HeartDiseasePredictor
# ============================================================================
# Page Configuration and Professional Styling
# ============================================================================

def set_page_config():
    """Configure Streamlit page settings and inject custom CSS for a stunning look."""
    st.set_page_config(
        page_title="CardioPredictNet AI",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a sophisticated, dark-themed, "Harvard-worthy" UI
    st.markdown("""
        <style>
        /* Import a professional font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        /* Main app background */
        .stApp {
            background-color: #0f172a; /* Deep indigo background */
            font-family: 'Roboto', sans-serif;
        }

        /* Glassmorphism containers */
        .block-container {
            background: rgba(30, 41, 59, 0.6); /* Semi-transparent slate */
            border-radius: 20px;
            padding: 2rem 3rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 2rem;
        }

        /* Headers and Text */
        h1, h2, h3 {
            font-weight: 700;
            text-align: center;
        }
        h1 {
            background: linear-gradient(90deg, #5e72e4, #9f58e4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
        }
        h2 {
            color: #e2e8f0; /* Light slate gray */
            font-size: 2.25rem;
        }
        h3 {
            color: #94a3b8; /* Medium slate gray */
            font-size: 1.5rem;
        }
        p, .stMarkdown, .stNumberInput, .stSelectbox {
            color: #cbd5e0; /* Lighter slate gray for text */
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1e293b; /* Darker slate for sidebar */
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        .css-1d391kg .stMarkdown {
            color: #94a3b8;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #5e72e4, #9f58e4);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 700;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(94, 114, 228, 0.4);
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(94, 114, 228, 0.5);
        }

        /* Input widget styling */
        .stNumberInput input, .stSelectbox select {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e2e8f0;
            border-radius: 8px;
        }

        /* Metric card styling */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric-card h3 {
            margin: 0;
            font-size: 1.2rem;
            color: #94a3b8;
        }
        .metric-card h2 {
            margin: 0.5rem 0 0 0;
            font-size: 2.5rem;
            color: #e2e8f0;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Keras Model Loading and Prediction Class
# ============================================================================

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


    def predict(self, X: np.ndarray) -> np.ndarray:
      """Generates binary predictions for new data using the Keras model."""
      if self.model is None:
          raise ValueError("Model not loaded. Call train() or load_trained_model() first.")
      
      # Scale the input data using the loaded scaler
      X_scaled = self.scaler.transform(X)

      # Use the standard Keras model.predict() method
      probabilities = self.model.predict(X_scaled)
      
      # Convert probabilities to binary predictions (0 or 1)
      predictions = (probabilities > self.config.prediction_threshold).astype(int)
      
      return predictions


# ============================================================================
# Visualization Functions (Styled for Excellence)
# ============================================================================

def create_gauge_chart(probability: float) -> go.Figure:
    """Creates an elegant gauge chart for risk probability."""
    if probability < 0.3:
        risk_level, color = "Low Risk", "#2dce89"
    elif probability < 0.7:
        risk_level, color = "Moderate Risk", "#fb6340"
    else:
        risk_level, color = "High Risk", "#f5365c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{risk_level}</b>", 'font': {'size': 24, 'color': '#e2e8f0'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(45, 206, 137, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(251, 99, 64, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(245, 54, 92, 0.3)'}],
        }))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': '#e2e8f0', 'family': 'Roboto'},
        height=350
    )
    return fig


# ============================================================================
# Main Application Logic
# ============================================================================

def main():
    """The main function that runs the Streamlit application."""
    set_page_config()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<h1>CardioPredictNet AI</h1>", unsafe_allow_html=True)
        st.markdown("### A Deep Learning Approach to Cardiovascular Risk Assessment")
        st.markdown("""
            This tool leverages a trained deep neural network to predict the likelihood of
            cardiovascular disease based on key health metrics.
        """)
        st.markdown("---")
        st.markdown("#### **Instructions:**")
        st.markdown("""
            1.  Input patient data in the main panel.
            2.  Click **'Analyze Risk'** to process the data.
            3.  Review the comprehensive risk analysis report.
        """)
        st.markdown("---")
        st.markdown("#### **Model Details:**")
        st.markdown("""
            - **Model:** TensorFlow/Keras
            - **Architecture:** Feedforward Neural Network
            - **Features:** 11 clinical & lifestyle inputs
        """)
        st.markdown("---")
        st.info("**Disclaimer:** This tool is for informational purposes only and does not constitute medical advice. Consult a healthcare professional for diagnosis.")

    # --- Main Panel ---
    st.markdown("<h1>CardioPredictNet AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8;'>Enter patient metrics below to receive an AI-powered risk assessment.</p>", unsafe_allow_html=True)

    # Load the model
    config = ModelConfig()
    predictor = HeartDiseasePredictor(config)
    predictor.load_trained_model()


    with st.container():
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        st.markdown("<h3>Patient Health Metrics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", 18, 100, 55)
            height = st.number_input("Height (cm)", 100, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 300, 75)
        with col2:
            gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            ap_hi = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
            ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        with col3:
            cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
            gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
            active = st.selectbox("Physical Activity", [1, 0], format_func=lambda x: "Active" if x == 1 else "Inactive")
        
        lifestyle_col1, lifestyle_col2 = st.columns(2)
        with lifestyle_col1:
             smoke = st.selectbox("Smoker", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        with lifestyle_col2:
             alco = st.selectbox("Consumes Alcohol", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Analysis Button ---
    st.markdown("")
    analyze_button = st.button("🔍 Analyze Cardiovascular Risk")
    st.markdown("")

    # --- Results Section ---
    if analyze_button:
        # Prepare feature array for the model
        features = np.array([[
            age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active
        ]], dtype=np.float32)

        with st.spinner("Performing deep learning analysis..."):
            prediction, probability = predictor.predict(features)

        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            st.markdown("<h2>Risk Analysis Report</h2>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            with res_col2:
                st.markdown("<h3>Key Metrics</h3>", unsafe_allow_html=True)
                
                # --- Metric Cards ---
                result_text = "High Risk" if prediction == 1 else "Low Risk"
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Predicted Risk</h3>
                        <h2 style='color: {"#f5365c" if prediction == 1 else "#2dce89"};'>{result_text}</h2>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Prediction Confidence</h3>
                        <h2>{probability*100:.1f}%</h2>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            st.markdown("<h3>Interpretation & Recommendations</h3>", unsafe_allow_html=True)
            if prediction == 1:
                st.warning("""
                **High Risk Detected:** The model predicts a significant probability of cardiovascular disease.
                
                **Recommendations:**
                - **Immediate Consultation:** It is strongly advised to consult a healthcare professional for a comprehensive evaluation.
                - **Lifestyle Review:** Discuss diet, exercise, smoking, and alcohol consumption with your doctor.
                - **Monitoring:** Regular monitoring of blood pressure and cholesterol is crucial.
                """)
            else:
                st.success("""
                **Low Risk Detected:** The model predicts a low probability of cardiovascular disease based on the provided data.
                
                **Recommendations:**
                - **Maintain Healthy Habits:** Continue with a balanced diet, regular physical activity, and avoiding smoking.
                - **Regular Check-ups:** Continue with routine health check-ups to monitor your long-term cardiovascular health.
                """)

            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()