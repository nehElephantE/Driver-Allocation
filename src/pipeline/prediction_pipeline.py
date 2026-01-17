"""
Prediction pipeline module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import joblib
import json
from pathlib import Path

from src.utils.config import config
from src.utils.logger import logger
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer

class PredictionPipeline:
    
    def __init__(self):
        self.logger = logger
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.threshold = config.get('prediction.threshold', 0.5)
        
    def load_trained_artifacts(self):
        self.logger.info("Loading trained artifacts...")
        
        # Load feature engineer
        fe_path = config.artifact_paths['features'] / "feature_engineer.pkl"
        if fe_path.exists():
            self.feature_engineer.load(fe_path)
            self.logger.info(f"Loaded feature engineer from {fe_path}")
        else:
            raise FileNotFoundError(f"Feature engineer not found at {fe_path}")
        
        # Load best model
        model_path = config.artifact_paths['models'] / "best_model.pkl"
        if model_path.exists():
            self.model = joblib.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            self.logger.info(f"Model type: {type(self.model).__name__}")
        else:
            # Try to load any model
            model_files = list(config.artifact_paths['models'].glob("*_model.pkl"))
            if model_files:
                self.model = joblib.load(model_files[0])
                self.logger.info(f"Loaded model from {model_files[0]}")
            else:
                raise FileNotFoundError(f"No trained models found in {config.artifact_paths['models']}")
    
    def prepare_test_data(self) -> pd.DataFrame:
        self.logger.info("Preparing test data...")
        
        try:
            # Try to load processed test data
            test_df = self.data_loader.load_processed_data('test', format='parquet')
        except:
            # If not processed, load raw and process
            self.logger.info("Processed test data not found, loading raw data...")
            _, _, test_df = self.data_loader.load_raw_data()
            test_df = self.data_loader.prepare_test_dataset(test_df)
        
        self.logger.info(f"Test data shape: {test_df.shape}")
        
        return test_df
    
    def make_predictions(self, test_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Making predictions...")
        
        # Transform features
        X_test = self.feature_engineer.transform(test_df)
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_test)[:, 1]
        else:
            probabilities = self.model.predict(X_test)
        
        # Apply threshold
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Create results dataframe
        results = pd.DataFrame({
            'order_id': test_df['order_id'].values if 'order_id' in test_df.columns else range(len(test_df)),
            'driver_id': test_df['driver_id'].values if 'driver_id' in test_df.columns else range(len(test_df)),
            'match_probability': probabilities,
            'predicted_match': predictions
        })
        
        self.logger.info(f"Predictions made: {len(results)} rows")
        
        return results
    
    def save_predictions(self, results: pd.DataFrame):
        self.logger.info("Saving predictions...")
        
        predictions_dir = config.artifact_paths['predictions']
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Save required output (order_id, driver_id)
        required_output = results[['order_id', 'driver_id']].copy()
        required_output_path = predictions_dir / config.get('output.required_output', 'order_driver_pairs.csv')
        required_output.to_csv(required_output_path, index=False)
        self.logger.info(f"Saved required output to {required_output_path}")
        
        # Save detailed predictions
        detailed_output = results.sort_values('match_probability', ascending=False)
        detailed_output_path = predictions_dir / config.get('output.detailed_output', 'detailed_predictions.csv')
        detailed_output.to_csv(detailed_output_path, index=False)
        self.logger.info(f"Saved detailed predictions to {detailed_output_path}")
        
        # Save prediction statistics
        self._save_prediction_statistics(results)
    
    def _save_prediction_statistics(self, results: pd.DataFrame):
        stats = {
            'total_predictions': len(results),
            'predicted_matches': int(results['predicted_match'].sum()),
            'predicted_non_matches': int((results['predicted_match'] == 0).sum()),
            'match_rate': float(results['predicted_match'].mean()),
            'probability_stats': {
                'min': float(results['match_probability'].min()),
                'max': float(results['match_probability'].max()),
                'mean': float(results['match_probability'].mean()),
                'median': float(results['match_probability'].median()),
                'std': float(results['match_probability'].std())
            },
            'threshold_used': self.threshold
        }
        
        stats_path = config.artifact_paths['predictions'] / "prediction_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        self.logger.info(f"Saved prediction statistics to {stats_path}")
        
        # Log summary
        self.logger.info("\n" + "="*60)
        self.logger.info("PREDICTION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total predictions:      {stats['total_predictions']:,}")
        self.logger.info(f"Predicted matches:      {stats['predicted_matches']:,}")
        self.logger.info(f"Predicted non-matches:  {stats['predicted_non_matches']:,}")
        self.logger.info(f"Match rate:             {stats['match_rate']:.2%}")
    
    def run(self) -> pd.DataFrame:
        self.logger.info("="*60)
        self.logger.info("PREDICTION PIPELINE STARTED")
        self.logger.info("="*60)
        
        try:
            # Step 1: Load trained artifacts
            self.logger.info("\nSTEP 1: Loading trained artifacts...")
            self.load_trained_artifacts()
            
            # Step 2: Prepare test data
            self.logger.info("\nSTEP 2: Preparing test data...")
            test_df = self.prepare_test_data()
            
            # Step 3: Make predictions
            self.logger.info("\nSTEP 3: Making predictions...")
            results = self.make_predictions(test_df)
            
            # Step 4: Save predictions
            self.logger.info("\nSTEP 4: Saving predictions...")
            self.save_predictions(results)
            
            self.logger.info("="*60)
            self.logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    pipeline = PredictionPipeline()
    results = pipeline.run()
    return results

if __name__ == "__main__":
    main()