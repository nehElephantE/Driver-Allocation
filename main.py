"""
Main entry point for the ride-hailing matching system
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.utils.logger import logger
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

# running training pipeline
def run_training():
    logger.info("="*60)
    logger.info("RUNNING TRAINING PIPELINE")
    logger.info("="*60)
    
    pipeline = TrainingPipeline()
    metrics = pipeline.run()
    return metrics

def run_prediction():
    logger.info("="*60)
    logger.info("RUNNING PREDICTION PIPELINE")
    logger.info("="*60)
    
    pipeline = PredictionPipeline()
    results = pipeline.run()
    return results

def main():
    logger.info("="*60)
    logger.info("RIDE-HAILING MATCHING PREDICTION SYSTEM")
    logger.info("="*60)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ride-Hailing Matching Prediction System")
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'predict', 'all'],
                       help='Run mode: train, predict, or all')
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['train', 'all']:
            metrics = run_training()
        
        if args.mode in ['predict', 'all']:
            results = run_prediction()
        
        logger.info("="*60)
        logger.info("SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        print(f"\nOutput files created in:")
        print(f"  - artifacts/models/ (trained models)")
        print(f"  - artifacts/predictions/ (predictions)")
        print(f"  - artifacts/metrics/ (performance metrics)")
        
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()