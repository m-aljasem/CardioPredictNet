# src/train.py
from model import HeartDiseasePredictor
from config import ModelConfig

def main():
    """Main function to run the model training and evaluation pipeline."""
    print("Initializing training pipeline...")
    
    # 1. Initialize configuration and the predictor
    config = ModelConfig()
    predictor = HeartDiseasePredictor(config)

    # 2. Load and prepare the data
    print(f"Loading data from {config.data_path}...")
    df = predictor.load_data(config.data_path)
    X_train, X_test, y_train, y_test = predictor.prepare_data(df)

    # 3. Train the model
    print("Starting model training...")
    predictor.train(X_train, y_train)
    print("Model training completed.")
    
    # 4. Evaluate the final model
    print("Evaluating model on the test set...")
    predictor.evaluate(X_test, y_test)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()