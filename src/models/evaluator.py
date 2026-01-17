"""
Model evaluation module
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

# Now this works perfectly
data = {
    'timestamp': datetime.now().isoformat()
}

from src.utils.config import config
from src.utils.logger import logger

class ModelEvaluator:
    
    def __init__(self):
        self.logger = logger
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                 model_name: str = "Model") -> Dict:
        self.logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': None  # Placeholder for log loss if needed
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # Log metrics
        self.logger.info(f"\n{model_name} Performance:")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        self.logger.info(f"\n  Confusion Matrix:")
        self.logger.info(f"    TN: {cm[0,0]:>6} | FP: {cm[0,1]:>6}")
        self.logger.info(f"    FN: {cm[1,0]:>6} | TP: {cm[1,1]:>6}")
        
        return metrics
    
    def plot_model_comparison(self, metrics_dict: Dict):
        self.logger.info("Creating model comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(metrics_dict.keys())
        
        # 1. Metrics comparison
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metric_names):
            values = [metrics_dict[m].get(metric, 0) for m in models]
            axes[0, 0].bar(x + i*width, values, width, label=metric)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width*2)
        axes[0, 0].set_xticklabels([m.upper() for m in models])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC curves
        for model_name, metrics in metrics_dict.items():
            if 'roc_curve' in metrics:
                roc_data = metrics['roc_curve']
                axes[0, 1].plot(
                    roc_data['fpr'], roc_data['tpr'],
                    label=f"{model_name} (AUC = {metrics.get('roc_auc', 0):.3f})"
                )
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall comparison
        for model_name, metrics in metrics_dict.items():
            axes[1, 0].scatter(
                metrics.get('recall', 0),
                metrics.get('precision', 0),
                s=100, label=model_name.upper()
            )
        
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary table
        axes[1, 1].axis('off')
        
        summary_text = "MODEL PERFORMANCE SUMMARY\n\n"
        for model_name, metrics in metrics_dict.items():
            summary_text += f"{model_name.upper()}:\n"
            summary_text += f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n"
            summary_text += f"  Precision: {metrics.get('precision', 0):.4f}\n"
            summary_text += f"  Recall:    {metrics.get('recall', 0):.4f}\n"
            summary_text += f"  F1-Score:  {metrics.get('f1_score', 0):.4f}\n"
            summary_text += f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}\n\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = config.artifact_paths['metrics'] / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved model comparison plot to {plot_path}")
    
    def plot_roc_curve(self, model: Any, X: np.ndarray, y: np.ndarray, 
                       model_name: str = "Model"):
        self.logger.info(f"Creating ROC curve for {model_name}...")
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            y_pred_proba = model.predict(X)
        
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = config.artifact_paths['metrics'] / f"{model_name}_roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved ROC curve to {plot_path}")
    
    def plot_confusion_matrix(self, model: Any, X: np.ndarray, y: np.ndarray, 
                              model_name: str = "Model"):
        self.logger.info(f"Creating confusion matrix for {model_name}...")
        
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Match', 'Match'],
                   yticklabels=['No Match', 'Match'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Save plot
        plot_path = config.artifact_paths['metrics'] / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved confusion matrix to {plot_path}")
    
    def generate_evaluation_report(self, metrics_dict: Dict, 
                                   feature_importance: Optional[Dict] = None):
        self.logger.info("Generating evaluation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(metrics_dict.keys()),
            'performance_metrics': metrics_dict,
            'best_model': max(metrics_dict.keys(), 
                            key=lambda x: metrics_dict[x].get('roc_auc', 0)),
            'feature_importance': feature_importance
        }
        
        # Save report
        report_path = config.artifact_paths['metrics'] / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        self.logger.info(f"Saved evaluation report to {report_path}")
        
        return report