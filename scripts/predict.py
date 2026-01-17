#!/usr/bin/env python
"""
Script to make predictions using trained models
"""
import sys
import os
from pathlib import Path
import datetime

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np

from src.utils.config import config
from src.utils.logger import logger
from src.pipeline.prediction_pipeline import PredictionPipeline

def make_predictions(args):
    logger.info("="*60)
    logger.info("PREDICTION SCRIPT")
    logger.info("="*60)
    
    try:
        # Create prediction pipeline
        pipeline = PredictionPipeline()
        
        # Update threshold if provided
        if args.threshold is not None:
            pipeline.threshold = args.threshold
            logger.info(f"Using custom threshold: {args.threshold}")
        
        # Run prediction pipeline
        logger.info(f"\nMaking predictions with:")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Threshold: {pipeline.threshold}")
        logger.info(f"  Output file: {args.output}")
        
        # Load specific model if requested
        if args.model != 'best':
            model_path = config.artifact_paths['models'] / f"{args.model}_model.pkl"
            if model_path.exists():
                import joblib
                pipeline.model = joblib.load(model_path)
                logger.info(f"Loaded specified model: {args.model}")
            else:
                logger.warning(f"Model {args.model} not found, using best model")
        
        # Run pipeline
        results = pipeline.run()
        
        # Save to custom output if specified
        if args.output:
            output_path = Path(args.output)
            results[['order_id', 'driver_id']].to_csv(output_path, index=False)
            logger.info(f"Saved predictions to custom location: {output_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PREDICTION COMPLETED")
        logger.info("="*60)
        
        print(f"\nTotal predictions: {len(results):,}")
        print(f"Predicted matches: {results['predicted_match'].sum():,}")
        print(f"Match rate: {results['predicted_match'].mean():.2%}")
        
        print(f"\nFirst 5 predictions:")
        print(results[['order_id', 'driver_id', 'predicted_match', 'match_probability']].head().to_string())
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_prediction_report(results, args):
    report_dir = config.artifact_paths['predictions'] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Create statistics
    stats = {
        'total_predictions': len(results),
        'predicted_matches': int(results['predicted_match'].sum()),
        'predicted_non_matches': int((results['predicted_match'] == 0).sum()),
        'match_rate': float(results['predicted_match'].mean()),
        'threshold_used': args.threshold if args.threshold is not None else config.get('prediction.threshold', 0.5),
        'model_used': args.model
    }
    
    # Save statistics
    import json
    stats_path = report_dir / "prediction_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    logger.info(f"Saved prediction statistics to {stats_path}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained models')
    
    parser.add_argument('--model', type=str, default='best',
                       choices=['best', 'random_forest', 'xgboost', 'lightgbm'],
                       help='Model to use for prediction (default: best)')
    
    parser.add_argument('--threshold', type=float, 
                       help='Prediction threshold (default: from config)')
    
    parser.add_argument('--output', type=str,
                       help='Custom output file path (default: artifacts/predictions/order_driver_pairs.csv)')
    
    parser.add_argument('--input', type=str,
                       help='Custom input test data file (default: data/raw/test_data.csv)')
    
    parser.add_argument('--sample', type=int,
                       help='Only process first N samples (for testing)')
    
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save predictions, just display')
    
    args = parser.parse_args()
    
    # Update input file if provided
    if args.input:
        input_path = Path(args.input)
        if input_path.exists():
            # Update config
            config.data_paths['raw']['test'] = input_path
        else:
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
    
    # Make predictions
    results = make_predictions(args)
    
    # Generate report if not disabled
    if not args.no_save:
        stats = generate_prediction_report(results, args)
    
    # Print final message
    print("\n" + "="*60)
    print("PREDICTION SCRIPT COMPLETED")
    print("="*60)
    
    if not args.no_save:
        print("\nOutput files:")
        print(f"1. Predictions: {config.artifact_paths['predictions']}/order_driver_pairs.csv")
        print(f"2. Detailed: {config.artifact_paths['predictions']}/detailed_predictions.csv")
        print(f"3. Statistics: {config.artifact_paths['predictions']}/reports/prediction_statistics.json")
    else:
        print("\nPredictions displayed (not saved)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())