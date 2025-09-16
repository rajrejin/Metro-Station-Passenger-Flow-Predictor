"""
Dubai Metro Prediction Interface

This module provides an easy-to-use interface for making predictions
using the trained Dubai Metro passenger flow models.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MetroPredictor:
    """
    Interface for making predictions using trained Dubai Metro models
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the predictor with trained models
        
        Args:
            models_dir (str): Directory containing saved models
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.station_list = []
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and preprocessors"""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found. Please train models first.")
        
        print("Loading trained models...")
        
        # Load models
        for target in ['checkin', 'checkout']:
            self.models[target] = {}
            for model_file in os.listdir(self.models_dir):
                if model_file.startswith(f'{target}_') and model_file.endswith('_model.joblib'):
                    model_name = model_file.replace(f'{target}_', '').replace('_model.joblib', '')
                    model_path = os.path.join(self.models_dir, model_file)
                    self.models[target][model_name] = joblib.load(model_path)
                    print(f"  Loaded {target} {model_name} model")
        
        # Load preprocessors
        for file in os.listdir(self.models_dir):
            if file.startswith('scaler_'):
                scaler_name = file.replace('scaler_', '').replace('.joblib', '')
                self.scalers[scaler_name] = joblib.load(os.path.join(self.models_dir, file))
            
            elif file.startswith('encoder_'):
                encoder_name = file.replace('encoder_', '').replace('.joblib', '')
                self.encoders[encoder_name] = joblib.load(os.path.join(self.models_dir, file))
        
        # Load feature columns
        self.feature_columns = joblib.load(os.path.join(self.models_dir, 'feature_columns.joblib'))
        
        # Get station list
        if 'station' in self.encoders:
            self.station_list = list(self.encoders['station'].classes_)
        
        print(f"Loaded models for {len(self.station_list)} stations")
        print(f"Available models: {list(self.models['checkin'].keys())}")
    
    def predict_single(self, station_name, target_datetime, model_name='random_forest', use_historical_data=True):
        """
        Predict passenger flow for a single station and time
        
        Args:
            station_name (str): Name of the metro station
            target_datetime (datetime): Target date and time for prediction
            model_name (str): Model to use ('linear_regression', 'random_forest', 'lightgbm', 'gradient_boosting')
            use_historical_data (bool): Whether to use historical patterns for lag features
            
        Returns:
            dict: Prediction results
        """
        if station_name not in self.station_list:
            raise ValueError(f"Station '{station_name}' not found. Available stations: {self.station_list[:5]}...")
        
        if model_name not in self.models['checkin']:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models['checkin'].keys())}")
        
        # Create feature vector
        features = self._create_features_for_datetime(station_name, target_datetime, use_historical_data)
        
        # Scale features
        features_scaled = self.scalers['features'].transform([features])
        
        # Make predictions
        checkin_pred = self.models['checkin'][model_name].predict(features_scaled)[0]
        checkout_pred = self.models['checkout'][model_name].predict(features_scaled)[0]
        
        # Ensure non-negative predictions
        checkin_pred = max(0, checkin_pred)
        checkout_pred = max(0, checkout_pred)
        
        return {
            'station': station_name,
            'datetime': target_datetime,
            'predicted_checkin': round(checkin_pred, 0),
            'predicted_checkout': round(checkout_pred, 0),
            'model_used': model_name,
            'is_operational': self._is_operational(target_datetime)
        }
    
    def predict_hourly_range(self, station_name, start_datetime, hours_ahead=24, model_name='random_forest'):
        """
        Predict passenger flow for a range of hours
        
        Args:
            station_name (str): Name of the metro station
            start_datetime (datetime): Starting date and time
            hours_ahead (int): Number of hours to predict
            model_name (str): Model to use
            
        Returns:
            pd.DataFrame: Predictions for the specified time range
        """
        predictions = []
        
        for hour_offset in range(hours_ahead):
            target_time = start_datetime + timedelta(hours=hour_offset)
            pred = self.predict_single(station_name, target_time, model_name)
            predictions.append(pred)
        
        return pd.DataFrame(predictions)
    
    def predict_multiple_stations(self, station_names, target_datetime, model_name='random_forest'):
        """
        Predict passenger flow for multiple stations at the same time
        
        Args:
            station_names (list): List of station names
            target_datetime (datetime): Target date and time
            model_name (str): Model to use
            
        Returns:
            pd.DataFrame: Predictions for all stations
        """
        predictions = []
        
        for station in station_names:
            if station in self.station_list:
                pred = self.predict_single(station, target_datetime, model_name)
                predictions.append(pred)
            else:
                print(f"Warning: Station '{station}' not found in training data")
        
        return pd.DataFrame(predictions)
    
    def _create_features_for_datetime(self, station_name, target_datetime, use_historical_data=True):
        """Create feature vector for a specific datetime"""
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
        features['station_encoded'] = self.encoders['station'].transform([station_name])[0]
        
        # Operational status
        features['is_operational'] = int(self._is_operational(target_datetime))
        
        # For lag and rolling features, use historical averages if historical data not available
        if use_historical_data:
            # In a real implementation, you would load historical data
            # For now, we'll use station averages
            pass
        
        # Fill lag features with station averages (simplified approach)
        avg_checkin = 587  # Overall average from training data
        avg_checkout = 587
        
        # Adjust by hour pattern
        hour_multiplier = self._get_hour_multiplier(target_datetime.hour)
        avg_checkin *= hour_multiplier
        avg_checkout *= hour_multiplier
        
        # Lag features
        for lag in [1, 2, 3, 24]:
            features[f'checkin_lag_{lag}'] = avg_checkin
            features[f'checkout_lag_{lag}'] = avg_checkout
        
        # Rolling features
        for window in [3, 6, 24]:
            features[f'checkin_rolling_mean_{window}'] = avg_checkin
            features[f'checkout_rolling_mean_{window}'] = avg_checkout
        
        # Station statistics (would need to be loaded from training data)
        features['station_checkin_count_mean'] = avg_checkin
        features['station_checkin_count_std'] = 200  # Approximate
        features['station_checkout_count_mean'] = avg_checkout
        features['station_checkout_count_std'] = 200
        
        # Return features in the correct order
        return [features[col] for col in self.feature_columns]
    
    def _get_hour_multiplier(self, hour):
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
    
    def _is_operational(self, target_datetime):
        """Check if metro is operational at given datetime"""
        weekday = target_datetime.weekday()
        hour = target_datetime.hour
        
        if weekday == 4:  # Friday
            return 5 <= hour <= 24 or hour == 0  # 5 AM to 1 AM next day
        elif weekday == 6:  # Sunday
            return 8 <= hour <= 23  # 8 AM to 12 AM
        else:  # Monday-Thursday, Saturday
            return 5 <= hour <= 23  # 5 AM to 12 AM
    
    def get_station_list(self):
        """Get list of available stations"""
        return self.station_list.copy()
    
    def get_available_models(self):
        """Get list of available model types"""
        return list(self.models['checkin'].keys())


def example_usage():
    """Example usage of the MetroPredictor"""
    print("🚇 DUBAI METRO PREDICTION INTERFACE")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = MetroPredictor()
        
        # Example 1: Single prediction
        print("\n📍 Single Station Prediction:")
        target_time = datetime(2025, 9, 3, 8, 0)  # Tomorrow 8 AM
        result = predictor.predict_single(
            station_name="BurJuman Metro Station",
            target_datetime=target_time,
            model_name="random_forest"
        )
        
        print(f"Station: {result['station']}")
        print(f"Time: {result['datetime'].strftime('%Y-%m-%d %H:%M')}")
        print(f"Predicted Check-ins: {result['predicted_checkin']}")
        print(f"Predicted Check-outs: {result['predicted_checkout']}")
        print(f"Operational: {result['is_operational']}")
        
        # Example 2: Hourly predictions for next 12 hours
        print(f"\n⏰ Next 12 Hours Prediction:")
        hourly_pred = predictor.predict_hourly_range(
            station_name="BurJuman Metro Station",
            start_datetime=datetime.now().replace(minute=0, second=0, microsecond=0),
            hours_ahead=12,
            model_name="random_forest"
        )
        
        print(hourly_pred[['datetime', 'predicted_checkin', 'predicted_checkout', 'is_operational']].to_string(index=False))
        
        # Example 3: Multiple stations at rush hour
        print(f"\n🚅 Rush Hour Predictions (6 PM):")
        top_stations = ["BurJuman Metro Station", "Al Rigga Metro Station", "Union Metro Station"]
        rush_hour = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
        
        multi_pred = predictor.predict_multiple_stations(
            station_names=top_stations,
            target_datetime=rush_hour,
            model_name="random_forest"
        )
        
        print(multi_pred[['station', 'predicted_checkin', 'predicted_checkout']].to_string(index=False))
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    example_usage()
