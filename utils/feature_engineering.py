"""
Dubai Metro Prediction System - Feature Engineering Module
Centralized feature engineering to ensure consistency across training and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import DubaiMetroConfig, DubaiMetroUtils


class DubaiMetroFeatureEngineer:
    """Centralized feature engineering for Dubai Metro prediction system"""
    
    def __init__(self, station_encoder=None):
        """
        Initialize feature engineer
        
        Args:
            station_encoder: Pre-fitted LabelEncoder for stations (for test data)
        """
        self.station_encoder = station_encoder
        self.feature_columns = []
        
    def create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Time-based features
        df['weekday'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_friday'] = (df['weekday'] == 4).astype(int)  # Special day in UAE
        
        return df
    
    def create_hour_features(self, df):
        """Create hour-based features"""
        df = df.copy()
        
        # Hour-based features
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        df['is_morning'] = (df['hour'].between(6, 11)).astype(int)
        df['is_afternoon'] = (df['hour'].between(12, 17)).astype(int)
        df['is_evening'] = (df['hour'].between(18, 22)).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
        
        return df
    
    def create_lag_features(self, df):
        """Create lag features (uses only past data)"""
        df = df.copy()
        
        # Sort by station and date for proper lag calculation
        df = df.sort_values(['station', 'date', 'hour'])
        
        # Lag features
        for lag in DubaiMetroConfig.LAG_WINDOWS:
            df[f'checkin_lag_{lag}'] = df.groupby('station')['checkin_count'].shift(lag)
            df[f'checkout_lag_{lag}'] = df.groupby('station')['checkout_count'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df):
        """Create rolling window features"""
        df = df.copy()
        
        # Sort by station and date for proper rolling calculation
        df = df.sort_values(['station', 'date', 'hour'])
        
        # Rolling window features
        for window in DubaiMetroConfig.ROLLING_WINDOWS:
            df[f'checkin_rolling_mean_{window}'] = df.groupby('station')['checkin_count'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'checkout_rolling_mean_{window}'] = df.groupby('station')['checkout_count'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    
    def encode_stations(self, df, fit=True):
        """Encode station names"""
        df = df.copy()
        
        if fit and self.station_encoder is None:
            # Training phase - fit new encoder
            self.station_encoder = LabelEncoder()
            df['station_encoded'] = self.station_encoder.fit_transform(df['station'])
        elif self.station_encoder is not None:
            # Test phase - use existing encoder
            df['station_encoded'] = self.station_encoder.transform(df['station'])
        else:
            raise ValueError("No station encoder available")
        
        return df
    
    def create_station_statistics(self, df):
        """Create station-specific statistical features"""
        df = df.copy()
        
        # Station statistics (historical averages)
        station_stats = df.groupby('station')[['checkin_count', 'checkout_count']].agg(['mean', 'std']).round(2)
        station_stats.columns = ['_'.join(col) for col in station_stats.columns]
        
        for stat_col in station_stats.columns:
            df[f'station_{stat_col}'] = df['station'].map(station_stats[stat_col])
        
        return df
    
    def create_all_features(self, df, fit_encoders=True):
        """
        Create all features for training or testing
        
        Args:
            df: Input dataframe with basic columns
            fit_encoders: Whether to fit encoders (True for training, False for test)
            
        Returns:
            DataFrame with all engineered features
        """
        # Convert date column to datetime if needed
        if df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'])
        
        # Create features step by step
        df = self.create_time_features(df)
        df = self.create_hour_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.encode_stations(df, fit=fit_encoders)
        df = self.create_station_statistics(df)
        
        # Define feature columns in consistent order
        self.feature_columns = [
            'hour', 'weekday', 'month', 'day_of_month', 'is_weekend', 'is_friday',
            'is_rush_hour', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
            'station_encoded', 'is_operational'
        ]
        
        # Add lag and rolling features
        lag_features = [col for col in df.columns if 'lag_' in col]
        rolling_features = [col for col in df.columns if 'rolling_' in col]
        station_features = [col for col in df.columns if 'station_checkin' in col or 'station_checkout' in col]
        
        self.feature_columns.extend(sorted(lag_features))
        self.feature_columns.extend(sorted(rolling_features))
        self.feature_columns.extend(sorted(station_features))
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        return df
    
    def get_feature_columns(self):
        """Get list of feature columns in consistent order"""
        return self.feature_columns.copy()
    
    def get_station_encoder(self):
        """Get fitted station encoder"""
        return self.station_encoder


def create_features_for_datetime(station_name, target_datetime, station_encoder, use_averages=True):
    """
    Create feature vector for a single datetime (for model_predictor.py)
    
    Args:
        station_name: Name of the station
        target_datetime: Target datetime for prediction
        station_encoder: Fitted LabelEncoder for stations
        use_averages: Whether to use historical averages for lag/rolling features
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Time-based features
    features['hour'] = target_datetime.hour
    features['weekday'] = target_datetime.weekday()
    features['month'] = target_datetime.month
    features['day_of_month'] = target_datetime.day
    features['is_weekend'] = int(target_datetime.weekday() >= 5)
    features['is_friday'] = int(target_datetime.weekday() == 4)
    
    # Hour-based features
    features['is_rush_hour'] = int((7 <= target_datetime.hour <= 9) or (17 <= target_datetime.hour <= 19))
    features['is_morning'] = int(6 <= target_datetime.hour <= 11)
    features['is_afternoon'] = int(12 <= target_datetime.hour <= 17)
    features['is_evening'] = int(18 <= target_datetime.hour <= 22)
    features['is_night'] = int(target_datetime.hour >= 23 or target_datetime.hour <= 5)
    
    # Station encoding
    features['station_encoded'] = station_encoder.transform([station_name])[0]
    
    # Operational status
    features['is_operational'] = int(DubaiMetroUtils.is_operational_hour(target_datetime, target_datetime.hour))
    
    if use_averages:
        # Use historical averages for lag and rolling features
        avg_checkin = 587  # Overall average from training data
        avg_checkout = 587
        
        # Adjust by hour pattern
        hour_multiplier = get_hour_multiplier(target_datetime.hour)
        avg_checkin *= hour_multiplier
        avg_checkout *= hour_multiplier
        
        # Lag features
        for lag in DubaiMetroConfig.LAG_WINDOWS:
            features[f'checkin_lag_{lag}'] = avg_checkin
            features[f'checkout_lag_{lag}'] = avg_checkout
        
        # Rolling features
        for window in DubaiMetroConfig.ROLLING_WINDOWS:
            features[f'checkin_rolling_mean_{window}'] = avg_checkin
            features[f'checkout_rolling_mean_{window}'] = avg_checkout
        
        # Station statistics
        features['station_checkin_count_mean'] = avg_checkin
        features['station_checkin_count_std'] = 200  # Approximate
        features['station_checkout_count_mean'] = avg_checkout
        features['station_checkout_count_std'] = 200
    
    return features


def get_hour_multiplier(hour):
    """Get traffic multiplier based on hour of day"""
    # Based on observed patterns from EDA
    hour_patterns = {
        0: 0.025, 1: 0.005, 2: 0.004, 3: 0.002, 4: 0.053,
        5: 0.338, 6: 0.764, 7: 1.352, 8: 1.723, 9: 1.358,
        10: 1.026, 11: 0.973, 12: 0.985, 13: 1.020, 14: 1.073,
        15: 1.189, 16: 1.461, 17: 2.040, 18: 2.383, 19: 1.971,
        20: 1.529, 21: 1.228, 22: 1.073, 23: 0.423
    }
    return hour_patterns.get(hour, 1.0)
