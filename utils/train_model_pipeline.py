"""
Dubai Metro Passenger Flow Prediction Pipeline

This module implements a comprehensive machine learning pipeline for predicting
Dubai Metro passenger flows using time series data.

Author: Dubai Metro Prediction System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Additional ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")


class MetroPredictionPipeline:
    """
    Complete pipeline for Dubai Metro passenger flow prediction
    """
    
    def __init__(self, dataset_path='data/csv_files/dubai_metro_hourly_dataset_train.csv'):
        """
        Initialize the prediction pipeline
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.data = None
        self.processed_data = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_columns = ['checkin_count', 'checkout_count']
        
    def load_data(self):
        """Load and initial data exploration"""
        print("Loading Dubai Metro dataset...")
        self.data = pd.read_csv(self.dataset_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Stations: {self.data['station'].nunique()}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        
        return self.data
    
    def exploratory_data_analysis(self, save_plots=True):
        """
        Perform comprehensive EDA
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.data[['checkin_count', 'checkout_count']].describe())
        
        # Top stations by traffic
        print("\nTop 10 Busiest Stations (by total checkins):")
        station_traffic = self.data.groupby('station')['checkin_count'].sum().sort_values(ascending=False)
        print(station_traffic.head(10))
        
        # Hourly patterns
        print("\nHourly Traffic Patterns:")
        hourly_avg = self.data.groupby('hour')[['checkin_count', 'checkout_count']].mean()
        print(hourly_avg)
        
        # Weekday patterns
        self.data['weekday'] = self.data['date'].dt.day_name()
        weekday_avg = self.data.groupby('weekday')[['checkin_count', 'checkout_count']].mean()
        print("\nWeekday Traffic Patterns:")
        print(weekday_avg)
        
        if save_plots:
            self._create_eda_plots()
    
    def _create_eda_plots(self):
        """Create and save EDA plots"""
        plt.style.use('default')
        
        # 1. Overall traffic trends
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Daily total traffic
        daily_traffic = self.data.groupby('date')[['checkin_count', 'checkout_count']].sum()
        axes[0, 0].plot(daily_traffic.index, daily_traffic['checkin_count'], label='Check-ins', alpha=0.7)
        axes[0, 0].plot(daily_traffic.index, daily_traffic['checkout_count'], label='Check-outs', alpha=0.7)
        axes[0, 0].set_title('Daily Total Traffic')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Hourly patterns
        hourly_avg = self.data.groupby('hour')[['checkin_count', 'checkout_count']].mean()
        axes[0, 1].bar(hourly_avg.index, hourly_avg['checkin_count'], alpha=0.7, label='Check-ins')
        axes[0, 1].bar(hourly_avg.index, hourly_avg['checkout_count'], alpha=0.7, label='Check-outs')
        axes[0, 1].set_title('Average Hourly Traffic')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].legend()
        
        # Top stations
        top_stations = self.data.groupby('station')['checkin_count'].sum().sort_values(ascending=False).head(10)
        axes[1, 0].barh(range(len(top_stations)), top_stations.values)
        axes[1, 0].set_yticks(range(len(top_stations)))
        axes[1, 0].set_yticklabels([s[:20] + '...' if len(s) > 20 else s for s in top_stations.index])
        axes[1, 0].set_title('Top 10 Busiest Stations')
        axes[1, 0].set_xlabel('Total Check-ins')
        
        # Weekday patterns
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_avg = self.data.groupby('weekday')[['checkin_count', 'checkout_count']].mean().reindex(weekday_order)
        axes[1, 1].bar(weekday_avg.index, weekday_avg['checkin_count'], alpha=0.7, label='Check-ins')
        axes[1, 1].bar(weekday_avg.index, weekday_avg['checkout_count'], alpha=0.7, label='Check-outs')
        axes[1, 1].set_title('Average Weekday Traffic')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Ensure outputs directory exists
        import os
        os.makedirs('outputs/training', exist_ok=True)
        
        plt.savefig('outputs/training/eda_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("EDA plots saved as 'outputs/training/eda_overview.png'")
    
    def feature_engineering(self):
        """
        Create features for machine learning models
        """
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        df = self.data.copy()
        
        # Time-based features
        df['weekday'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_friday'] = (df['weekday'] == 4).astype(int)  # Special day in UAE
        
        # Hour-based features
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        df['is_morning'] = (df['hour'].between(6, 11)).astype(int)
        df['is_afternoon'] = (df['hour'].between(12, 17)).astype(int)
        df['is_evening'] = (df['hour'].between(18, 22)).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
        
        # Lag features (previous hours)
        df = df.sort_values(['station', 'date', 'hour'])
        
        for lag in [1, 2, 3, 24]:  # 1-3 hours ago, same time yesterday
            df[f'checkin_lag_{lag}'] = df.groupby('station')['checkin_count'].shift(lag)
            df[f'checkout_lag_{lag}'] = df.groupby('station')['checkout_count'].shift(lag)
        
        # Rolling window features
        for window in [3, 6, 24]:  # 3-hour, 6-hour, daily windows
            df[f'checkin_rolling_mean_{window}'] = df.groupby('station')['checkin_count'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'checkout_rolling_mean_{window}'] = df.groupby('station')['checkout_count'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        
        # Station encoding
        le_station = LabelEncoder()
        df['station_encoded'] = le_station.fit_transform(df['station'])
        self.encoders['station'] = le_station
        
        # Station statistics (historical averages)
        station_stats = df.groupby('station')[['checkin_count', 'checkout_count']].agg(['mean', 'std']).round(2)
        station_stats.columns = ['_'.join(col) for col in station_stats.columns]
        station_mapping = dict(zip(le_station.classes_, le_station.transform(le_station.classes_)))
        
        for stat_col in station_stats.columns:
            df[f'station_{stat_col}'] = df['station'].map(station_stats[stat_col])
        
        # Select feature columns
        self.feature_columns = [
            'hour', 'weekday', 'month', 'day_of_month', 'is_weekend', 'is_friday',
            'is_rush_hour', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
            'station_encoded', 'is_operational'
        ]
        
        # Add lag and rolling features (excluding NaN rows for training)
        lag_features = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        station_features = [col for col in df.columns if 'station_checkin' in col or 'station_checkout' in col]
        
        self.feature_columns.extend(lag_features)
        self.feature_columns.extend(station_features)
        
        self.processed_data = df
        
        print(f"Feature engineering completed!")
        print(f"Total features: {len(self.feature_columns)}")
        print(f"Feature columns: {self.feature_columns}")
        
        return df
    
    def prepare_data_for_training(self, test_size=0.2, validation_size=0.1):
        """
        Prepare data for training with proper time series splits
        
        Args:
            test_size (float): Proportion of data for testing
            validation_size (float): Proportion of data for validation
        """
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Remove rows with NaN values (from lag features)
        df_clean = self.processed_data.dropna()
        
        # Sort by date for time series split
        df_clean = df_clean.sort_values(['date', 'hour', 'station'])
        
        # Features and targets
        X = df_clean[self.feature_columns]
        y_checkin = df_clean['checkin_count']
        y_checkout = df_clean['checkout_count']
        
        # Time-based split (avoid data leakage)
        n_samples = len(df_clean)
        train_end_idx = int(n_samples * (1 - test_size - validation_size))
        val_end_idx = int(n_samples * (1 - test_size))
        
        # Split data
        X_train = X.iloc[:train_end_idx]
        X_val = X.iloc[train_end_idx:val_end_idx]
        X_test = X.iloc[val_end_idx:]
        
        y_checkin_train = y_checkin.iloc[:train_end_idx]
        y_checkin_val = y_checkin.iloc[train_end_idx:val_end_idx]
        y_checkin_test = y_checkin.iloc[val_end_idx:]
        
        y_checkout_train = y_checkout.iloc[:train_end_idx]
        y_checkout_val = y_checkout.iloc[train_end_idx:val_end_idx]
        y_checkout_test = y_checkout.iloc[val_end_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['features'] = scaler
        
        print(f"Data preparation completed!")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        return {
            'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
            'y_checkin_train': y_checkin_train, 'y_checkin_val': y_checkin_val, 'y_checkin_test': y_checkin_test,
            'y_checkout_train': y_checkout_train, 'y_checkout_val': y_checkout_val, 'y_checkout_test': y_checkout_test,
            'train_dates': df_clean.iloc[:train_end_idx]['date'],
            'val_dates': df_clean.iloc[train_end_idx:val_end_idx]['date'],
            'test_dates': df_clean.iloc[val_end_idx:]['date']
        }
    
    def train_models(self, data_splits):
        """
        Train multiple models for comparison
        
        Args:
            data_splits (dict): Data splits from prepare_data_for_training
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        models_to_train = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if LIGHTGBM_AVAILABLE:
            models_to_train['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100, random_state=42, verbose=-1
            )
        
        # Train models for both checkin and checkout prediction
        for target in ['checkin', 'checkout']:
            print(f"\nTraining models for {target} prediction...")
            self.models[target] = {}
            
            y_train = data_splits[f'y_{target}_train']
            y_val = data_splits[f'y_{target}_val']
            
            for model_name, model in models_to_train.items():
                print(f"  Training {model_name}...")
                
                # Train model
                model.fit(data_splits['X_train'], y_train)
                
                # Validate
                val_pred = model.predict(data_splits['X_val'])
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_mae = mean_absolute_error(y_val, val_pred)
                val_r2 = r2_score(y_val, val_pred)
                
                print(f"    Validation RMSE: {val_rmse:.2f}")
                print(f"    Validation MAE: {val_mae:.2f}")
                print(f"    Validation R²: {val_r2:.3f}")
                
                # Store model
                self.models[target][model_name] = model
        
        print("Model training completed!")
        
        return self.models
    
    def evaluate_models(self, data_splits):
        """
        Evaluate all trained models on test set
        
        Args:
            data_splits (dict): Data splits from prepare_data_for_training
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        
        for target in ['checkin', 'checkout']:
            print(f"\n{target.upper()} PREDICTION RESULTS:")
            print("-" * 40)
            
            results[target] = {}
            y_test = data_splits[f'y_{target}_test']
            
            for model_name, model in self.models[target].items():
                # Predict
                test_pred = model.predict(data_splits['X_test'])
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                mae = mean_absolute_error(y_test, test_pred)
                r2 = r2_score(y_test, test_pred)
                
                # Store results
                results[target][model_name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'predictions': test_pred
                }
                
                print(f"{model_name:20s} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | R²: {r2:8.3f}")
        
        return results
    
    def save_models(self, save_dir='models'):
        """
        Save trained models and preprocessors
        
        Args:
            save_dir (str): Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for target in self.models:
            for model_name, model in self.models[target].items():
                filename = f"{save_dir}/{target}_{model_name}_model.joblib"
                joblib.dump(model, filename)
        
        # Save scalers and encoders
        for scaler_name, scaler in self.scalers.items():
            filename = f"{save_dir}/scaler_{scaler_name}.joblib"
            joblib.dump(scaler, filename)
        
        for encoder_name, encoder in self.encoders.items():
            filename = f"{save_dir}/encoder_{encoder_name}.joblib"
            joblib.dump(encoder, filename)
        
        # Save feature columns
        joblib.dump(self.feature_columns, f"{save_dir}/feature_columns.joblib")
        
        print(f"Models and preprocessors saved to '{save_dir}' directory")
    
    def run_complete_pipeline(self):
        """
        Run the complete ML pipeline
        """
        print("DUBAI METRO PREDICTION PIPELINE")
        print("=" * 50)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. Load data
            self.load_data()
            
            # 2. EDA
            self.exploratory_data_analysis()
            
            # 3. Feature engineering
            self.feature_engineering()
            
            # 4. Prepare data
            data_splits = self.prepare_data_for_training()
            
            # 5. Train models
            self.train_models(data_splits)
            
            # 6. Evaluate models
            results = self.evaluate_models(data_splits)
            
            # 7. Save models
            self.save_models()
            
            print("\n" + "="*50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return results
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = MetroPredictionPipeline()
    results = pipeline.run_complete_pipeline()
