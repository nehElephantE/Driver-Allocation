#!/usr/bin/env python
"""
Script to train models for ride-hailing matching prediction
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
from datetime import datetime

from src.utils.config import config
from src.utils.logger import logger
from src.pipeline.training_pipeline import TrainingPipeline
from src.models.trainer import ModelTrainer

def train_models(args):
    logger.info("="*60)
    logger.info("MODEL TRAINING SCRIPT")
    logger.info("="*60)
    
    try:
        # Create training pipeline
        pipeline = TrainingPipeline()
        
        # Run training pipeline
        logger.info(f"\nTraining models with configuration:")
        logger.info(f"  Models to train: {args.models}")
        logger.info(f"  Random seed: {args.random_seed}")
        logger.info(f"  Test size: {args.test_size}")
        
        # Update config with command line arguments
        if args.models:
            config.update('model.models_to_train', args.models)
        
        if args.random_seed:
            config.update('model.random_seed', args.random_seed)
        
        if args.test_size:
            config.update('model.test_size', args.test_size)
        
        # Run pipeline
        metrics = pipeline.run()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        
        # Find best model
        if hasattr(pipeline.model_trainer, 'best_model_name'):
            best_model = pipeline.model_trainer.best_model_name
            best_auc = metrics.get(best_model, {}).get('roc_auc', 0)
            logger.info(f"\nBest model: {best_model.upper()}")
            logger.info(f"Best AUC-ROC: {best_auc:.4f}")
        
        # Save training report
        save_training_report(metrics, args.output_dir)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def save_training_report(metrics, output_dir):
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report
    report = {
        'training_date': datetime.now().isoformat(),
        'models_trained': list(metrics.keys()),
        'performance_metrics': metrics,
        'best_model': max(metrics.keys(), key=lambda x: metrics[x].get('roc_auc', 0)) if metrics else None
    }
    
    # Save as JSON
    report_path = report_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    logger.info(f"Saved training report to {report_path}")
    
    # Also save as markdown for readability
    save_markdown_report(report, report_dir)

def save_markdown_report(report, report_dir):
    md_path = report_dir / "training_report.md"
    
    with open(md_path, 'w') as f:
        f.write("# Model Training Report\n\n")
        f.write(f"**Date:** {report['training_date']}\n\n")
        
        f.write("## Models Trained\n")
        for model in report['models_trained']:
            f.write(f"- {model}\n")
        
        f.write("\n## Performance Metrics\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n")
        f.write("|-------|----------|-----------|--------|----------|---------|\n")
        
        for model_name, metrics in report['performance_metrics'].items():
            f.write(f"| {model_name} | "
                   f"{metrics.get('accuracy', 0):.4f} | "
                   f"{metrics.get('precision', 0):.4f} | "
                   f"{metrics.get('recall', 0):.4f} | "
                   f"{metrics.get('f1_score', 0):.4f} | "
                   f"{metrics.get('roc_auc', 0):.4f} |\n")
        
        if report['best_model']:
            f.write(f"\n## Best Model\n")
            f.write(f"**{report['best_model'].upper()}**\n")
            best_metrics = report['performance_metrics'][report['best_model']]
            f.write(f"- Accuracy: {best_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"- Precision: {best_metrics.get('precision', 0):.4f}\n")
            f.write(f"- Recall: {best_metrics.get('recall', 0):.4f}\n")
            f.write(f"- F1-Score: {best_metrics.get('f1_score', 0):.4f}\n")
            f.write(f"- AUC-ROC: {best_metrics.get('roc_auc', 0):.4f}\n")
    
    logger.info(f"Saved markdown report to {md_path}")

def main():
    parser = argparse.ArgumentParser(description='Train models for ride-hailing matching prediction')
    
    parser.add_argument('--models', nargs='+', 
                       default=['random_forest', 'xgboost'],
                       help='Models to train (default: random_forest xgboost)')
    
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test size for validation')
    
    parser.add_argument('--output-dir', type=str, default='artifacts',
                       help='Output directory for artifacts')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Update config if custom config file is provided
    if args.config != 'config.yaml':
        config_path = Path(args.config)
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
            config.config.update(custom_config)
    
    # Run training
    metrics = train_models(args)
    
    # Print final message
    print("\n" + "="*60)
    print("TRAINING SCRIPT COMPLETED")
    print("="*60)
    print("\nOutput files:")
    print(f"1. Models: {config.artifact_paths['models']}/")
    print(f"2. Metrics: {config.artifact_paths['metrics']}/")
    print(f"3. Reports: {Path(args.output_dir)}/reports/")
    print(f"4. Feature engineer: {config.artifact_paths['features']}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())