#!/usr/bin/env python
"""
Script to evaluate trained models
"""
import sys
import os
from pathlib import Path
import datetime

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import config
from src.utils.logger import logger
from src.models.evaluator import ModelEvaluator
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer

def evaluate_models(args):
    logger.info("="*60)
    logger.info("MODEL EVALUATION SCRIPT")
    logger.info("="*60)
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Load feature engineer
        fe_path = config.artifact_paths['features'] / "feature_engineer.pkl"
        if not fe_path.exists():
            logger.error(f"Feature engineer not found at {fe_path}")
            logger.info("Please run training first or provide trained models")
            sys.exit(1)
        
        feature_engineer = FeatureEngineer()
        feature_engineer.load(fe_path)
        logger.info(f"Loaded feature engineer from {fe_path}")
        
        # Load data for evaluation
        logger.info("\nLoading data for evaluation...")
        data_loader = DataLoader()
        
        # Load training data for cross-validation
        try:
            train_data = data_loader.load_processed_data('train', format='parquet')
            X_train_raw = train_data.drop('target', axis=1)
            y_train = train_data['target']
            
            # Transform features
            X_train = feature_engineer.transform(X_train_raw)
            logger.info(f"Training data loaded: {X_train.shape}")
        except:
            logger.warning("Could not load processed training data")
            X_train, y_train = None, None
        
        # Load models to evaluate
        models_to_evaluate = []
        model_names = []
        
        if args.models == ['all']:
            # Load all available models
            model_files = list(config.artifact_paths['models'].glob("*_model.pkl"))
            for model_file in model_files:
                model_name = model_file.stem.replace('_model', '')
                model = joblib.load(model_file)
                models_to_evaluate.append(model)
                model_names.append(model_name)
                logger.info(f"Loaded model: {model_name}")
        else:
            # Load specified models
            for model_name in args.models:
                model_path = config.artifact_paths['models'] / f"{model_name}_model.pkl"
                if model_path.exists():
                    model = joblib.load(model_path)
                    models_to_evaluate.append(model)
                    model_names.append(model_name)
                    logger.info(f"Loaded model: {model_name}")
                else:
                    logger.warning(f"Model {model_name} not found at {model_path}")
        
        if not models_to_evaluate:
            logger.error("No models found to evaluate")
            sys.exit(1)
        
        # Evaluate models
        logger.info(f"\nEvaluating {len(models_to_evaluate)} models...")
        
        all_metrics = {}
        feature_importance = {}
        
        for model, model_name in zip(models_to_evaluate, model_names):
            logger.info(f"\nEvaluating {model_name}...")
            
            if X_train is not None and y_train is not None:
                # Split for evaluation
                from sklearn.model_selection import train_test_split
                X_eval, X_test_eval, y_eval, y_test_eval = train_test_split(
                    X_train, y_train, 
                    test_size=0.2, 
                    random_state=config.get('model.random_seed', 42),
                    stratify=y_train
                )
                
                # Evaluate
                metrics = evaluator.evaluate(model, X_test_eval, y_test_eval, model_name)
                all_metrics[model_name] = metrics
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    fi = pd.DataFrame({
                        'feature': range(X_train.shape[1]),
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    feature_importance[model_name] = fi
                    
                    # Print top features
                    logger.info(f"\nTop 10 features for {model_name}:")
                    for idx, row in fi.head(10).iterrows():
                        logger.info(f"  Feature {int(row['feature'])}: {row['importance']:.4f}")
                
                # Create confusion matrix plot
                evaluator.plot_confusion_matrix(model, X_test_eval, y_test_eval, model_name)
                
                # Create ROC curve plot
                evaluator.plot_roc_curve(model, X_test_eval, y_test_eval, model_name)
        
        # Create comparison plot
        if len(all_metrics) > 1:
            evaluator.plot_model_comparison(all_metrics)
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(all_metrics, feature_importance)
        
        # Save detailed report
        save_evaluation_report(report, args.output_dir)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED")
        logger.info("="*60)
        
        if all_metrics:
            best_model = max(all_metrics.keys(), 
                           key=lambda x: all_metrics[x].get('roc_auc', 0))
            best_auc = all_metrics[best_model].get('roc_auc', 0)
            logger.info(f"\nBest performing model: {best_model.upper()}")
            logger.info(f"Best AUC-ROC: {best_auc:.4f}")
        
        return all_metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def save_evaluation_report(report, output_dir):
    report_dir = Path(output_dir) / "evaluation_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_path = report_dir / f"evaluation_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    logger.info(f"Saved evaluation report to {json_path}")
    
    # Save as markdown
    md_path = report_dir / f"evaluation_report_{timestamp}.md"
    save_evaluation_markdown(report, md_path)
    
    return json_path, md_path

def save_evaluation_markdown(report, md_path):
    with open(md_path, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Date:** {report.get('timestamp', datetime.now().isoformat())}\n\n")
        
        f.write("## Models Evaluated\n")
        for model in report.get('models_evaluated', []):
            f.write(f"- {model}\n")
        
        f.write("\n## Performance Metrics\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n")
        f.write("|-------|----------|-----------|--------|----------|---------|\n")
        
        metrics = report.get('performance_metrics', {})
        for model_name, model_metrics in metrics.items():
            f.write(f"| {model_name} | "
                   f"{model_metrics.get('accuracy', 0):.4f} | "
                   f"{model_metrics.get('precision', 0):.4f} | "
                   f"{model_metrics.get('recall', 0):.4f} | "
                   f"{model_metrics.get('f1_score', 0):.4f} | "
                   f"{model_metrics.get('roc_auc', 0):.4f} |\n")
        
        f.write("\n## Best Model\n")
        best_model = report.get('best_model', 'N/A')
        f.write(f"**{best_model.upper()}**\n\n")
        
        if best_model in metrics:
            best_metrics = metrics[best_model]
            f.write("### Performance\n")
            f.write(f"- **Accuracy**: {best_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"- **Precision**: {best_metrics.get('precision', 0):.4f}\n")
            f.write(f"- **Recall**: {best_metrics.get('recall', 0):.4f}\n")
            f.write(f"- **F1-Score**: {best_metrics.get('f1_score', 0):.4f}\n")
            f.write(f"- **AUC-ROC**: {best_metrics.get('roc_auc', 0):.4f}\n")
            
            # Confusion matrix
            cm = best_metrics.get('confusion_matrix', {})
            f.write("\n### Confusion Matrix\n")
            f.write("```\n")
            f.write(f"True Negatives:  {cm.get('true_negative', 0)}\n")
            f.write(f"False Positives: {cm.get('false_positive', 0)}\n")
            f.write(f"False Negatives: {cm.get('false_negative', 0)}\n")
            f.write(f"True Positives:  {cm.get('true_positive', 0)}\n")
            f.write("```\n")
    
    logger.info(f"Saved markdown report to {md_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    
    parser.add_argument('--models', nargs='+', 
                       default=['all'],
                       help='Models to evaluate (default: all)')
    
    parser.add_argument('--output-dir', type=str, default='artifacts',
                       help='Output directory for evaluation reports')
    
    parser.add_argument('--data-split', type=float, default=0.2,
                       help='Test split size for evaluation')
    
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation evaluation')
    
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_models(args)
    
    # Print final message
    print("\n" + "="*60)
    print("EVALUATION SCRIPT COMPLETED")
    print("="*60)
    
    if metrics:
        print("\nEvaluation Summary:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {model_metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {model_metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {model_metrics.get('recall', 0):.4f}")
            print(f"  F1-Score:  {model_metrics.get('f1_score', 0):.4f}")
            print(f"  AUC-ROC:   {model_metrics.get('roc_auc', 0):.4f}")
    
    print(f"\nOutput files:")
    print(f"1. Reports: {Path(args.output_dir)}/evaluation_reports/")
    print(f"2. Visualizations: {config.artifact_paths['metrics']}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())