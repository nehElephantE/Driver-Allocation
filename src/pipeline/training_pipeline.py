"""
Training pipeline module
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
import joblib
from pathlib import Path

from src.utils.config import config
from src.utils.logger import logger
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.trainer import ModelTrainer

class TrainingPipeline:
    
    def __init__(self):
        self.logger = logger
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
    def run(self) -> Dict:
        self.logger.info("="*60)
        self.logger.info("TRAINING PIPELINE STARTED")
        self.logger.info("="*60)
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("\nSTEP 1: Loading and preparing data...")
            booking_df, participant_df, test_df = self.data_loader.load_raw_data()
            
            # Create training dataset
            training_data = self.data_loader.create_training_dataset(booking_df, participant_df)
            
            # Prepare test dataset
            test_data = self.data_loader.prepare_test_dataset(test_df)
            
            # Save processed data
            self.data_loader.save_processed_data(training_data, 'train', format='parquet')
            self.data_loader.save_processed_data(test_data, 'test', format='parquet')
            
            # Step 2: Feature engineering
            self.logger.info("\nSTEP 2: Feature engineering...")
            
            # Prepare features for training
            X_train = training_data.drop('target', axis=1)
            y_train = training_data['target']
            
            # Fit feature engineer
            X_features = self.feature_engineer.fit_transform(X_train, y_train)
            
            # Save feature engineer
            fe_path = config.artifact_paths['features'] / "feature_engineer.pkl"
            self.feature_engineer.save(fe_path)
            self.logger.info(f"Saved feature engineer to {fe_path}")
            
            # Step 3: Model training
            self.logger.info("\nSTEP 3: Model training...")
            metrics = self.model_trainer.train(
                X_features, y_train.values,
                models_to_train=config.get('model.models_to_train', ['random_forest', 'xgboost'])
            )
            
            # Step 4: Save pipeline artifacts
            self.logger.info("\nSTEP 4: Saving pipeline artifacts...")
            self._save_pipeline_artifacts()
            
            self.logger.info("="*60)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_pipeline_artifacts(self):
        artifacts = {
            'feature_engineer': self.feature_engineer,
            'model_trainer': self.model_trainer
        }
        
        artifacts_path = config.artifact_paths['models'] / "pipeline_artifacts.pkl"
        joblib.dump(artifacts, artifacts_path)
        self.logger.info(f"Saved pipeline artifacts to {artifacts_path}")

def main():
    pipeline = TrainingPipeline()
    metrics = pipeline.run()
    return metrics

if __name__ == "__main__":
    main()