"""
Data loading module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import pickle
import pyarrow.parquet as pq
import pyarrow as pa

from src.utils.config import config
from src.utils.logger import logger

class DataLoader:
    
    def __init__(self):
        self.logger = logger
        self.data_paths = config.data_paths
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.logger.info("Loading raw data...")
        
        try:
            # Load booking data
            booking_df = pd.read_csv(self.data_paths['raw']['booking'])
            self.logger.info(f"Booking data loaded: {booking_df.shape}")
            
            # Load participant data
            participant_df = pd.read_csv(self.data_paths['raw']['participant'])
            self.logger.info(f"Participant data loaded: {participant_df.shape}")
            
            # Load test data
            test_df = pd.read_csv(self.data_paths['raw']['test'])
            self.logger.info(f"Test data loaded: {test_df.shape}")
            
            return booking_df, participant_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            raise
    
    def parse_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'event_timestamp') -> pd.DataFrame:
        self.logger.info(f"Parsing timestamps in column: {timestamp_col}")
        
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column {timestamp_col} not found in dataframe")
            return df
        
        # Handle different timestamp formats
        try:
            # Remove timezone info for consistent parsing
            df[timestamp_col] = df[timestamp_col].astype(str).str.replace(' UTC', '', regex=False)
            
            # Try multiple parsing strategies
            for format_str in ['mixed', 'ISO8601', None]:
                try:
                    df[timestamp_col] = pd.to_datetime(
                        df[timestamp_col], 
                        format=format_str,
                        errors='raise'
                    )
                    self.logger.info(f"Successfully parsed timestamps with format: {format_str}")
                    break
                except Exception:
                    continue
            
            # Extract datetime features
            df[f'{timestamp_col}_dt'] = df[timestamp_col]
            df[f'{timestamp_col}_hour'] = df[timestamp_col].dt.hour
            df[f'{timestamp_col}_day'] = df[timestamp_col].dt.day
            df[f'{timestamp_col}_month'] = df[timestamp_col].dt.month
            df[f'{timestamp_col}_year'] = df[timestamp_col].dt.year
            df[f'{timestamp_col}_dayofweek'] = df[timestamp_col].dt.dayofweek
            
            self.logger.info("Timestamp parsing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error parsing timestamps: {e}")
            # Create dummy datetime features if parsing fails
            df[f'{timestamp_col}_hour'] = 12
            df[f'{timestamp_col}_dayofweek'] = 0
        
        return df
    
    def validate_data(self, df: pd.DataFrame, df_name: str) -> bool:
        self.logger.info(f"Validating {df_name} data...")
        
        validation_results = {
            'has_data': len(df) > 0,
            'has_duplicates': df.duplicated().any(),
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Log validation results
        self.logger.info(f"{df_name} Validation Results:")
        self.logger.info(f"  Has data: {validation_results['has_data']}")
        self.logger.info(f"  Has duplicates: {validation_results['has_duplicates']}")
        self.logger.info(f"  Missing values: {validation_results['missing_values']}")
        
        if validation_results['has_duplicates']:
            self.logger.warning(f"{df_name} contains duplicates")
        
        if validation_results['missing_values'] > 0:
            self.logger.warning(f"{df_name} contains {validation_results['missing_values']} missing values")
        
        return validation_results['has_data']
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, format: str = 'parquet'):
        save_path = config.data_paths['processed'][filename]
        
        try:
            if format == 'parquet':
                df.to_parquet(save_path, index=False)
            elif format == 'csv':
                df.to_csv(save_path, index=False)
            elif format == 'pickle':
                df.to_pickle(save_path)
            
            self.logger.info(f"Saved processed data to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise
    
    def load_processed_data(self, filename: str, format: str = 'parquet') -> pd.DataFrame:
        load_path = config.data_paths['processed'][filename]
        
        try:
            if format == 'parquet':
                df = pd.read_parquet(load_path)
            elif format == 'csv':
                df = pd.read_csv(load_path)
            elif format == 'pickle':
                df = pd.read_pickle(load_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Loaded processed data from {load_path}: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading processed data: {e}")
            raise
    
    def create_training_dataset(self, booking_df: pd.DataFrame, 
                                participant_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Creating training dataset...")
        
        # Parse timestamps
        booking_df = self.parse_timestamps(booking_df)
        participant_df = self.parse_timestamps(participant_df)
        
        # Get successful matches (COMPLETED orders with ACCEPTED drivers)
        completed_orders = booking_df[booking_df['booking_status'] == 'COMPLETED'].copy()
        accepted_drivers = participant_df[participant_df['participant_status'] == 'ACCEPTED'].copy()
        
        # Create positive samples
        positive_samples = pd.merge(
            completed_orders[[
                'order_id', 'customer_id', 'trip_distance',
                'pickup_latitude', 'pickup_longitude',
                'event_timestamp_hour', 'event_timestamp_dayofweek'
            ]],
            accepted_drivers[[
                'order_id', 'driver_id', 'driver_latitude',
                'driver_longitude', 'driver_gps_accuracy'
            ]],
            on='order_id',
            how='inner'
        )
        positive_samples['target'] = 1
        self.logger.info(f"Positive samples created: {len(positive_samples)}")
        
        # Create negative samples from rejected/ignored requests
        rejected_drivers = participant_df[
            participant_df['participant_status'].isin(['REJECTED', 'IGNORED'])
        ].copy()
        
        negative_samples = pd.merge(
            booking_df[[
                'order_id', 'customer_id', 'trip_distance',
                'pickup_latitude', 'pickup_longitude',
                'event_timestamp_hour', 'event_timestamp_dayofweek'
            ]],
            rejected_drivers[[
                'order_id', 'driver_id', 'driver_latitude',
                'driver_longitude', 'driver_gps_accuracy'
            ]],
            on='order_id',
            how='inner'
        )
        negative_samples['target'] = 0
        self.logger.info(f"Negative samples created: {len(negative_samples)}")
        
        # Balance the dataset
        min_samples = min(len(positive_samples), len(negative_samples))
        
        if min_samples < len(positive_samples):
            positive_samples = positive_samples.sample(
                n=min_samples, 
                random_state=config.get('model.random_seed', 42)
            )
        
        if min_samples < len(negative_samples):
            negative_samples = negative_samples.sample(
                n=min_samples, 
                random_state=config.get('model.random_seed', 42)
            )
        
        # Combine samples
        training_data = pd.concat([positive_samples, negative_samples], ignore_index=True)
        
        # Shuffle the data
        training_data = training_data.sample(
            frac=1, 
            random_state=config.get('model.random_seed', 42)
        ).reset_index(drop=True)
        
        self.logger.info(f"Final training dataset: {training_data.shape}")
        self.logger.info(f"Target distribution: {training_data['target'].value_counts().to_dict()}")
        
        return training_data
    
    def prepare_test_dataset(self, test_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Preparing test dataset...")
        
        # Parse timestamps
        test_df = self.parse_timestamps(test_df)
        
        # Add time features
        test_df['event_timestamp_hour'] = test_df['event_timestamp'].dt.hour
        test_df['event_timestamp_dayofweek'] = test_df['event_timestamp'].dt.dayofweek
        
        self.logger.info(f"Test dataset prepared: {test_df.shape}")
        
        return test_df