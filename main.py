"""
Main entry point for the ride-hailing matching system
Supports both Local (full training) and Cloud (demo mode) environments
"""
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.utils.logger import logger
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline


def detect_environment():
    """Detect if running on Streamlit Cloud or locally"""
    if os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('STREAMLIT_CLOUD'):
        return 'cloud'
    return 'local'


ENVIRONMENT = detect_environment()


def run_training():
    """Run training pipeline - only works in local mode"""
    logger.info("="*60)
    logger.info("RUNNING TRAINING PIPELINE")
    logger.info("="*60)
    
    if ENVIRONMENT == 'cloud':
        logger.warning("="*60)
        logger.warning("TRAINING DISABLED IN CLOUD MODE")
        logger.warning("="*60)
        logger.warning("Streamlit Cloud has a read-only filesystem.")
        logger.warning("Please run training locally on your machine.")
        logger.warning("Then upload the trained models to your GitHub repository.")
        logger.warning("="*60)
        return None
    
    try:
        pipeline = TrainingPipeline()
        metrics = pipeline.run()
        logger.info("Training completed successfully!")
        return metrics
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_prediction():
    """Run prediction pipeline - works in both modes"""
    logger.info("="*60)
    logger.info("RUNNING PREDICTION PIPELINE")
    logger.info("="*60)
    
    if ENVIRONMENT == 'cloud':
        logger.info("Running in CLOUD mode - using demo predictions")
        logger.info("For full predictions, train models locally first.")
        
        # Return demo results instead of running full pipeline
        import pandas as pd
        import random
        
        # Create demo predictions
        demo_results = pd.DataFrame({
            'order_id': [f'ORD_{i:05d}' for i in range(1, 101)],
            'driver_id': [f'DRV_{random.randint(1000, 9999)}' for _ in range(100)],
            'match_probability': [round(random.uniform(0.2, 0.95), 3) for _ in range(100)],
            'predicted_match': [random.choice([0, 1]) for _ in range(100)]
        })
        
        logger.info(f"Demo predictions created: {len(demo_results)} rows")
        return demo_results
    
    try:
        pipeline = PredictionPipeline()
        results = pipeline.run()
        logger.info("Prediction completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def check_artifacts():
    """Check if trained artifacts exist"""
    artifacts_dir = project_root / "artifacts" / "models"
    
    if artifacts_dir.exists() and list(artifacts_dir.glob("*.pkl")):
        models = list(artifacts_dir.glob("*.pkl"))
        logger.info(f"Found {len(models)} trained model(s):")
        for model in models:
            logger.info(f"  - {model.name}")
        return True
    else:
        logger.warning("No trained models found in artifacts/models/")
        logger.info("Run 'python main.py --mode train' locally first")
        return False


def main():
    logger.info("="*60)
    logger.info("RIDE-HAILING MATCHING PREDICTION SYSTEM")
    logger.info(f"Environment: {ENVIRONMENT.upper()}")
    logger.info("="*60)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ride-Hailing Matching Prediction System")
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'predict', 'all', 'status'],
                       help='Run mode: train, predict, all, or status')
    
    parser.add_argument('--force', action='store_true',
                       help='Force training even in cloud mode (not recommended)')
    
    args = parser.parse_args()
    
    # Just show status
    if args.mode == 'status':
        logger.info("\n" + "="*60)
        logger.info("SYSTEM STATUS")
        logger.info("="*60)
        logger.info(f"Environment: {ENVIRONMENT.upper()}")
        logger.info(f"Project root: {project_root}")
        
        # Check for required directories
        required_dirs = ['data/raw', 'artifacts/models', 'artifacts/predictions']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                logger.info(f"✅ {dir_name} exists")
            else:
                logger.info(f"❌ {dir_name} missing")
        
        # Check for trained models
        check_artifacts()
        
        logger.info("="*60)
        return
    
    try:
        # Handle training mode
        if args.mode in ['train', 'all']:
            if ENVIRONMENT == 'cloud' and not args.force:
                logger.error("\n" + "="*60)
                logger.error("TRAINING NOT AVAILABLE IN CLOUD MODE")
                logger.error("="*60)
                logger.error("\nStreamlit Cloud has a read-only filesystem.")
                logger.error("Please run training on your local machine:")
                logger.error("  python main.py --mode train")
                logger.error("\nThen commit and push the trained models to GitHub.")
                logger.error("\nTo see current status, run:")
                logger.error("  python main.py --mode status")
                logger.error("="*60)
                sys.exit(1)
            else:
                metrics = run_training()
                if metrics:
                    logger.info("\nTraining metrics saved to: artifacts/metrics/")
        
        # Handle prediction mode
        if args.mode in ['predict', 'all']:
            results = run_prediction()
            
            if results is not None:
                logger.info(f"\nPredictions created: {len(results)} rows")
                
                # Show sample predictions
                if ENVIRONMENT == 'cloud':
                    logger.info("\nDemo predictions sample:")
                    import pandas as pd
                    pd.set_option('display.max_columns', None)
                    logger.info(f"\n{results.head(10).to_string()}")
                
                # Save predictions if not in cloud mode
                if ENVIRONMENT != 'cloud' and args.mode != 'all':
                    output_path = project_root / "artifacts" / "predictions" / "predictions.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    results.to_csv(output_path, index=False)
                    logger.info(f"Predictions saved to: {output_path}")
        
        logger.info("\n" + "="*60)
        logger.info("SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        if ENVIRONMENT == 'cloud':
            logger.info("\n💡 TIP: For full functionality, run locally:")
            logger.info("   1. Install dependencies: pip install -r requirements.txt")
            logger.info("   2. Train models: python main.py --mode train")
            logger.info("   3. Run UI: streamlit run app.py")
        
        print("\n" + "="*60)
        print("OUTPUT LOCATIONS")
        print("="*60)
        print(f"  - Trained models: artifacts/models/")
        print(f"  - Predictions:    artifacts/predictions/")
        print(f"  - Metrics:        artifacts/metrics/")
        print(f"  - Logs:           logs/pipeline.log")
        
    except KeyboardInterrupt:
        logger.info("\nSystem execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()