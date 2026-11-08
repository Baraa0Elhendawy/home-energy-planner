import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "nn_model.joblib")

class LoadForecaster:
    def __init__(self):
        # 2 hidden layers for non-linear daily/weekly interactions
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64), 
            activation='relu', 
            solver='adam', 
            max_iter=500, 
            random_state=42, 
            early_stopping=True # Prevents overfitting and speeds up training
        )
        self.features = ['hour', 'weekday', 'month', 'prev_24h', 'same_hour_last_week']
        self.target_cols = [f'target_{i}' for i in range(1, 25)]
        
    def train(self, train_df):
        logging.info("Preparing training data...")
        X_train = train_df[self.features]
        Y_train = train_df[self.target_cols]
        
        logging.info(f"Training MLPRegressor on {len(train_df)} hourly samples...")
        logging.info("This will take 1 to 3 minutes. Please wait...")
        self.model.fit(X_train, Y_train)
        logging.info("Training complete.")
        
    def save_model(self):
        joblib.dump(self.model, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")
        
    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            logging.info(f"Loaded existing model from {MODEL_PATH}")
        else:
            raise FileNotFoundError("Model artifact not found. Run training first.")

    def predict_24h(self, current_state_features):
        """Predicts the next 24 hours given a 1D array of the 5 features."""
        return self.model.predict([current_state_features])[0]
        
    def evaluate(self, test_df):
        logging.info("Evaluating model on the held-out test set...")
        X_test = test_df[self.features]
        Y_test = test_df[self.target_cols]
        
        preds = self.model.predict(X_test)
        mse_nn = mean_squared_error(Y_test, preds)
        
        actuals = test_df['Global_active_power']
        mse_yesterday = mean_squared_error(actuals[24:], actuals.shift(24)[24:])
        mse_last_week = mean_squared_error(actuals[168:], actuals.shift(168)[168:])
        
        logging.info("--- Test MSE Results ---")
        logging.info(f"Neural Network MSE:      {mse_nn:.4f}")
        logging.info(f"Same Hour Yesterday MSE: {mse_yesterday:.4f}")
        logging.info(f"Same Hour Last Week MSE: {mse_last_week:.4f}")
        
        return mse_nn, mse_yesterday, mse_last_week

if __name__ == "__main__":
    logging.info("Loading CSV files into memory...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col='Datetime', parse_dates=True)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col='Datetime', parse_dates=True)
    
    forecaster = LoadForecaster()
    forecaster.train(train_df)
    forecaster.save_model()
    forecaster.evaluate(test_df)