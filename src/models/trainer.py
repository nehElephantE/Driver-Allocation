"""
Model training module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib
import json
from datetime import datetime
from pathlib import Path

from src.utils.config import config
from src.utils.logger import logger
from src.models.evaluator import ModelEvaluator

class ModelTrainer:
    
    def __init__(self):
        self.logger = logger
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        
    def prepare_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        self.logger.info("Preparing training data...")
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config.get('model.validation_size', 0.1),
            random_state=config.get('model.random_seed', 42),
            stratify=y
        )
        
        self.logger.info(f"Training set: {X_train.shape}")
        self.logger.info(f"Validation set: {X_val.shape}")
        self.logger.info(f"Train target distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        return X_train, X_val, y_train, y_val
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[RandomForestClassifier, Dict]:
        self.logger.info("Training Random Forest model...")
        
        # Get parameters from config
        rf_params = config.get('random_forest', {})
        
        # Create and train model
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluator.evaluate(model, X_val, y_val, "Random Forest")
        
        # Store feature importance
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, metrics
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Tuple[XGBClassifier, Dict]:
        self.logger.info("Training XGBoost model...")
        
        # Get parameters from config
        xgb_params = config.get('xgboost', {})
        
        # Create and train model
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=config.get('training.early_stopping_rounds', 50),
            verbose=False
        )
        
        # Evaluate model
        metrics = self.evaluator.evaluate(model, X_val, y_val, "XGBoost")
        
        # Store feature importance
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, metrics
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Tuple[lgb.LGBMClassifier, Dict]:
        self.logger.info("Training LightGBM model...")
        
        # Get parameters from config
        lgb_params = config.get('lightgbm', {})
        
        # Create and train model
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=config.get('training.early_stopping_rounds', 50),
            verbose=False
        )
        
        # Evaluate model
        metrics = self.evaluator.evaluate(model, X_val, y_val, "LightGBM")
        
        # Store feature importance
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, metrics
    
    def perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray, 
                                model_name: str, cv_folds: int = 5) -> Dict:
        self.logger.info(f"Performing {cv_folds}-fold cross-validation for {model_name}...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=config.get('model.random_seed', 42))
        
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=config.get('training.scoring_metric', 'roc_auc'),
            n_jobs=-1
        )
        
        cv_metrics = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max(),
            'all_scores': cv_scores.tolist()
        }
        
        self.logger.info(f"{model_name} CV {config.get('training.scoring_metric', 'roc_auc')}: "
                        f"{cv_metrics['mean_score']:.4f} (+/- {cv_metrics['std_score']:.4f})")
        
        return cv_metrics
    
    def select_best_model(self) -> Tuple[Any, str]:
        if not self.metrics:
            raise ValueError("No models have been trained yet")
        
        # Select based on ROC-AUC
        self.best_model_name = max(
            self.metrics.keys(),
            key=lambda x: self.metrics[x].get('roc_auc', 0)
        )
        
        self.best_model = self.models[self.best_model_name]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BEST MODEL SELECTED: {self.best_model_name.upper()}")
        self.logger.info(f"ROC-AUC: {self.metrics[self.best_model_name]['roc_auc']:.4f}")
        self.logger.info(f"{'='*60}")
        
        return self.best_model, self.best_model_name
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                        models_to_train: Optional[List[str]] = None) -> Dict:
        self.logger.info("Training all models...")
        
        if models_to_train is None:
            models_to_train = config.get('model.models_to_train', ['random_forest', 'xgboost'])
        
        # Prepare training data
        X_train, X_val, y_train, y_val = self.prepare_training_data(X, y)
        
        # Train each model
        for model_name in models_to_train:
            try:
                if model_name == 'random_forest':
                    model, metrics = self.train_random_forest(X_train, y_train, X_val, y_val)
                elif model_name == 'xgboost':
                    model, metrics = self.train_xgboost(X_train, y_train, X_val, y_val)
                elif model_name == 'lightgbm':
                    model, metrics = self.train_lightgbm(X_train, y_train, X_val, y_val)
                else:
                    self.logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                # Store model and metrics
                self.models[model_name] = model
                self.metrics[model_name] = metrics
                
                # Perform cross-validation
                cv_metrics = self.perform_cross_validation(
                    model, X, y, model_name, 
                    cv_folds=config.get('training.cv_folds', 5)
                )
                self.metrics[model_name]['cross_validation'] = cv_metrics
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Select best model
        if self.models:
            self.select_best_model()
        
        return self.metrics
    
    def save_models(self):
        self.logger.info("Saving models...")
        
        models_dir = config.artifact_paths['models']
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all models
        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = models_dir / "best_model.pkl"
            joblib.dump(self.best_model, best_model_path)
            self.logger.info(f"Saved best model to {best_model_path}")
    
    def save_metrics(self):
        self.logger.info("Saving model metrics...")
        
        metrics_dir = config.artifact_paths['metrics']
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = metrics_dir / "model_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for model_name, metrics in self.metrics.items():
                serializable_metrics[model_name] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                        v.item() if hasattr(v, 'item') else v)
                    for k, v in metrics.items()
                }
            json.dump(serializable_metrics, f, indent=4)
        
        self.logger.info(f"Saved metrics to {metrics_path}")
        
        # Save metrics as CSV for readability
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_csv_path = metrics_dir / "model_metrics.csv"
        metrics_df.to_csv(metrics_csv_path)
        self.logger.info(f"Saved metrics CSV to {metrics_csv_path}")
        
        # Save feature importance
        if self.feature_importance:
            for model_name, fi_df in self.feature_importance.items():
                fi_path = metrics_dir / f"{model_name}_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                self.logger.info(f"Saved {model_name} feature importance to {fi_path}")
                
                # Log top features
                if len(fi_df) > 0:
                    self.logger.info(f"\nTop 10 features for {model_name}:")
                    for idx, row in fi_df.head(10).iterrows():
                        self.logger.info(f"  Feature {int(row['feature'])}: {row['importance']:.4f}")
    
    def save_training_summary(self):
        self.logger.info("Saving training summary...")
        
        summary = {
            'training_date': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'best_model_metrics': self.metrics.get(self.best_model_name, {}),
            'all_models_trained': list(self.models.keys()),
            'config_used': config.config
        }
        
        summary_path = config.artifact_paths['metrics'] / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        self.logger.info(f"Saved training summary to {summary_path}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              models_to_train: Optional[List[str]] = None) -> Dict:
        self.logger.info("="*60)
        self.logger.info("MODEL TRAINING PIPELINE")
        self.logger.info("="*60)
        
        # Train models
        metrics = self.train_all_models(X, y, models_to_train)
        
        # Save results
        self.save_models()
        self.save_metrics()
        self.save_training_summary()
        
        # Create visualizations
        self.evaluator.plot_model_comparison(self.metrics)
        if self.best_model_name and self.best_model_name in self.metrics:
            self.evaluator.plot_roc_curve(
                self.models[self.best_model_name],
                X, y,
                self.best_model_name
            )
        
        self.logger.info("="*60)
        self.logger.info("MODEL TRAINING COMPLETED")
        self.logger.info("="*60)
        
        return metrics