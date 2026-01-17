"""
Feature engineering for ride-hailing matching prediction
"""
import numpy as np
import pandas as pd
import math
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

from src.utils.config import config
from src.utils.logger import logger

class FeatureEngineer:
    
    def __init__(self):
        self.logger = logger
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_columns = []
        self.is_fitted = False
        self.train_stats = {}
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth radius in km
        return c * r
    
    def calculate_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                            'driver_latitude', 'driver_longitude']):
            df['pickup_driver_distance_km'] = df.apply(
                lambda x: self.haversine_distance(
                    x['pickup_latitude'], x['pickup_longitude'],
                    x['driver_latitude'], x['driver_longitude']
                ), axis=1
            )
        
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'event_timestamp' in df.columns:
            # Parse timestamp if it's a string
            if not pd.api.types.is_datetime64_any_dtype(df['event_timestamp']):
                try:
                    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], format='mixed')
                except:
                    # If parsing fails, create dummy hour
                    df['event_timestamp_hour'] = 12
                    return df
            
            # Extract features
            df['hour'] = df['event_timestamp'].dt.hour
            df['day_of_week'] = df['event_timestamp'].dt.dayofweek
            df['month'] = df['event_timestamp'].dt.month
            
            # Derived features
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_rush_hour'] = (
                ((df['hour'] >= 7) & (df['hour'] <= 10)) | 
                ((df['hour'] >= 16) & (df['hour'] <= 19))
            ).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
            
            # Circular encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Trip distance features
        if 'trip_distance' in df.columns:
            df['trip_distance_log'] = np.log1p(df['trip_distance'])
            df['trip_distance_squared'] = df['trip_distance'] ** 2
        
        # GPS features
        if 'driver_gps_accuracy' in df.columns:
            df['gps_accuracy_log'] = np.log1p(df['driver_gps_accuracy'].fillna(100))
            df['gps_good'] = (df['driver_gps_accuracy'] <= 20).astype(int).fillna(0)
        
        return df
    
    def extract_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if all(col in df.columns for col in ['pickup_driver_distance_km', 'trip_distance']):
            df['distance_to_trip_ratio'] = df['pickup_driver_distance_km'] / (df['trip_distance'] + 0.001)
            df['total_travel_distance'] = df['pickup_driver_distance_km'] + df['trip_distance']
        
        if all(col in df.columns for col in ['hour', 'pickup_driver_distance_km']):
            df['hour_distance_interaction'] = df['hour'] * df['pickup_driver_distance_km']
        
        return df
    
    def extract_location_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        
        if 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
            df['pickup_lat_bin'] = np.round(df['pickup_latitude'], 2)
            df['pickup_lon_bin'] = np.round(df['pickup_longitude'], 2)
        
        if 'driver_latitude' in df.columns and 'driver_longitude' in df.columns:
            df['driver_lat_bin'] = np.round(df['driver_latitude'], 2)
            df['driver_lon_bin'] = np.round(df['driver_longitude'], 2)
        
        return df
    
    def extract_historical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        if is_training:
            # Calculate statistics from training data
            if 'target' in df.columns:
                # Driver acceptance rate
                driver_acceptance = df.groupby('driver_id')['target'].mean()
                self.train_stats['driver_acceptance'] = driver_acceptance.to_dict()
                df['driver_acceptance_rate'] = df['driver_id'].map(driver_acceptance)
            
            # Driver experience
            driver_counts = df['driver_id'].value_counts()
            self.train_stats['driver_experience'] = driver_counts.to_dict()
            df['driver_experience'] = df['driver_id'].map(driver_counts)
            
            # Customer frequency
            customer_counts = df['customer_id'].value_counts()
            self.train_stats['customer_frequency'] = customer_counts.to_dict()
            df['customer_frequency'] = df['customer_id'].map(customer_counts)
        else:
            # Use pre-computed statistics for test data
            if 'driver_acceptance' in self.train_stats:
                df['driver_acceptance_rate'] = df['driver_id'].map(
                    self.train_stats['driver_acceptance']
                ).fillna(0.5)
            
            if 'driver_experience' in self.train_stats:
                df['driver_experience'] = df['driver_id'].map(
                    self.train_stats['driver_experience']
                ).fillna(0)
            
            if 'customer_frequency' in self.train_stats:
                df['customer_frequency'] = df['customer_id'].map(
                    self.train_stats['customer_frequency']
                ).fillna(0)
        
        return df
    
    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        self.logger.info(f"Creating features (training={is_training})...")
        
        # Apply feature transformations
        df = self.extract_time_features(df)
        df = self.calculate_distances(df)
        df = self.extract_basic_features(df)
        df = self.extract_interaction_features(df)
        df = self.extract_location_features(df)
        
        # Only add historical features if we have the necessary columns
        if 'driver_id' in df.columns:
            df = self.extract_historical_features(df, is_training)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        # Columns to exclude
        exclude_cols = [
            'order_id', 'driver_id', 'customer_id', 'event_timestamp',
            'target', 'booking_status', 'participant_status', 'experiment_key'
        ]
        
        # Get all numeric columns that are not excluded
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        return feature_cols
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        self.logger.info("Fitting feature engineer...")
        
        # Add target to X for historical features
        if y is not None:
            X = X.copy()
            X['target'] = y.values if hasattr(y, 'values') else y
        
        # Create features
        X_processed = self.create_features(X, is_training=True)
        
        # Remove target if it was added
        if 'target' in X_processed.columns:
            X_processed = X_processed.drop('target', axis=1)
        
        # Get feature columns
        self.feature_columns = self.get_feature_columns(X_processed)
        self.logger.info(f"Identified {len(self.feature_columns)} feature columns")
        
        # Prepare feature matrix
        X_features = X_processed[self.feature_columns].fillna(0)
        
        # Fit scaler and imputer
        self.scaler.fit(X_features)
        self.imputer.fit(X_features)
        
        self.is_fitted = True
        self.logger.info("Feature engineer fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        self.logger.info("Transforming data...")
        
        # Create features
        X_processed = self.create_features(X, is_training=False)
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        # Prepare feature matrix
        X_features = X_processed[self.feature_columns].fillna(0)
        
        # Transform
        X_imputed = self.imputer.transform(X_features)
        X_scaled = self.scaler.transform(X_imputed)
        
        self.logger.info(f"Transformed data shape: {X_scaled.shape}")
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath: Path):
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'train_stats': self.train_stats,
            'is_fitted': self.is_fitted
        }, filepath)
        self.logger.info(f"Saved feature engineer to {filepath}")
    
    def load(self, filepath: Path):
        saved = joblib.load(filepath)
        self.scaler = saved['scaler']
        self.imputer = saved['imputer']
        self.label_encoders = saved['label_encoders']
        self.feature_columns = saved['feature_columns']
        self.train_stats = saved['train_stats']
        self.is_fitted = saved['is_fitted']
        self.logger.info(f"Loaded feature engineer from {filepath}")
        return self