"""
Model Evaluation Module for Dubai Metro Prediction System
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import project modules
from config import DubaiMetroConfig

class ModelEvaluator:
    """Evaluate trained models on test data"""
    
    def __init__(self):
        self.config = DubaiMetroConfig()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.PATHS['testing_outputs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    def evaluate_models(self):
        """Main evaluation function"""
        try:
            # Load test data - using the existing processed test dataset
            test_file = self.config.PATHS['test_output']
            if not os.path.exists(test_file):
                self.logger.error(f"Test dataset not found: {test_file}")
                return False
            
            # Load test data directly
            test_data = pd.read_csv(test_file)
            
            # Create timestamp column from date and hour if it doesn't exist
            if 'timestamp' not in test_data.columns:
                test_data['date'] = pd.to_datetime(test_data['date'])
                test_data['timestamp'] = test_data['date'] + pd.to_timedelta(test_data['hour'], unit='h')
            else:
                test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
            
            if test_data is None or test_data.empty:
                self.logger.error("No test data available")
                return False
            
            self.logger.info(f"Test data shape: {test_data.shape}")
            self.logger.info(f"Date range: {test_data['timestamp'].min().date()} to {test_data['timestamp'].max().date()}")
            
            # Load saved encoders and scalers from training
            try:
                station_encoder = joblib.load(Path(self.config.PATHS['models_dir']) / 'encoder_station.joblib')
                scaler = joblib.load(Path(self.config.PATHS['models_dir']) / 'scaler_features.joblib')
                self.logger.info("Loaded encoders and scalers from training")
            except FileNotFoundError:
                self.logger.error("Could not load encoders/scalers from training")
                return False
            
            # Process test data for features - simplified approach using saved feature columns
            df = test_data.copy()
            df = df.sort_values(['station', 'date', 'hour']).reset_index(drop=True)
            
            # Basic time features
            df['date'] = pd.to_datetime(df['date'])
            df['weekday'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month  
            df['day_of_month'] = df['date'].dt.day
            df['is_weekend'] = (df['weekday'] >= 5).astype(int)
            df['is_friday'] = (df['weekday'] == 4).astype(int)
            df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
            df['is_morning'] = (df['hour'].between(6, 11)).astype(int)
            df['is_afternoon'] = (df['hour'].between(12, 17)).astype(int)
            df['is_evening'] = (df['hour'].between(18, 22)).astype(int)
            df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
            
            # Encode stations using saved encoder
            df['station_encoded'] = station_encoder.transform(df['station'])
            
            # Lag features
            for lag in [1, 2, 3, 24]:
                df[f'checkin_lag_{lag}'] = df.groupby('station')['checkin_count'].shift(lag)
                df[f'checkout_lag_{lag}'] = df.groupby('station')['checkout_count'].shift(lag)
            
            # Rolling features
            for window in [3, 6, 24]:
                df[f'checkin_rolling_mean_{window}'] = df.groupby('station')['checkin_count'].rolling(window).mean().reset_index(0, drop=True)
                df[f'checkout_rolling_mean_{window}'] = df.groupby('station')['checkout_count'].rolling(window).mean().reset_index(0, drop=True)
            
            # Station statistics
            station_stats = df.groupby('station').agg({
                'checkin_count': ['mean', 'std'],
                'checkout_count': ['mean', 'std']
            }).round(2)
            station_stats.columns = ['station_checkin_count_mean', 'station_checkin_count_std',
                                   'station_checkout_count_mean', 'station_checkout_count_std']
            df = df.merge(station_stats, left_on='station', right_index=True, how='left')
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Extract targets
            y_test_checkin = df['checkin_count'].values
            y_test_checkout = df['checkout_count'].values
            
            # Load saved feature columns
            try:
                feature_cols = joblib.load(Path(self.config.PATHS['models_dir']) / 'feature_columns.joblib')
                self.logger.info(f"Loaded {len(feature_cols)} feature columns from training")
                
                # Ensure test features match training features
                X_test = df[feature_cols].values
                X_test = scaler.transform(X_test)  # Apply saved scaler
                self.logger.info(f"Features used: {X_test.shape[1]}")
                self.logger.info(f"Test samples: {X_test.shape[0]}")
                self.logger.info(f"Feature columns: {list(feature_cols)}")
                
            except FileNotFoundError:
                self.logger.warning("Feature columns file not found, using all features")
                return False
            
            # Evaluate models
            results = []
            model_files = list(Path(self.config.PATHS['models_dir']).glob('*_model.joblib'))
            
            if not model_files:
                self.logger.error("No model files found")
                return False
            
            self.logger.info("Starting model evaluation...")
            
            for model_file in model_files:
                try:
                    # Load model
                    model = joblib.load(model_file)
                    model_name = model_file.stem
                    
                    # Determine model type and target
                    if 'checkin' in model_name:
                        target_type = 'Checkin'
                        y_true = y_test_checkin
                    else:
                        target_type = 'Checkout'
                        y_true = y_test_checkout
                    
                    # Get base model name
                    base_name = model_name.replace('checkin_', '').replace('checkout_', '').replace('_model', '')
                    
                    self.logger.info(f"Evaluating {base_name} {target_type.lower()} model...")
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae = mean_absolute_error(y_true, y_pred)
                    mape = self.calculate_mape(y_true, y_pred)
                    
                    # Store results
                    results.append({
                        'Model': base_name.title().replace('_', ' '),
                        'Type': target_type,
                        'R²': r2,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {model_file.name}: {e}")
                    continue
            
            if not results:
                self.logger.error("No results to summarize")
                return False
            
            # Save results
            results_df = pd.DataFrame(results)
            output_dir = Path(self.config.PATHS['testing_outputs'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            results_file = output_dir / 'model_evaluation_results.csv'
            results_df.to_csv(results_file, index=False)
            
            # Create visualizations
            self.create_visualizations(results_df, output_dir)
            
            # Print summary
            self.print_results_summary(results_df)
            
            self.logger.info(f"Visualizations saved to {output_dir}/ directory")
            self.logger.info("Model evaluation completed successfully!")
            self.logger.info(f"Results saved to {output_dir}/ directory")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            return False
    
    def create_visualizations(self, results_df, output_dir):
        """Create evaluation visualizations"""
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
            
            # Metrics to plot
            metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                
                # Prepare data for plotting
                pivot_data = results_df.pivot(index='Model', columns='Type', values=metric)
                
                # Create bar plot
                pivot_data.plot(kind='bar', ax=ax, width=0.8)
                ax.set_title(f'{metric} by Model and Type', fontweight='bold')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric)
                ax.legend(title='Type')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.4f' if metric in ['R²', 'MAE'] else '%.2f', 
                               rotation=90, fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / 'model_evaluation_comparison.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
    
    def print_results_summary(self, results_df):
        """Print formatted results summary"""
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Find best models for each metric
        for target_type in ['Checkin', 'Checkout']:
            type_data = results_df[results_df['Type'] == target_type]
            
            print(f"\n{target_type.upper()} MODELS:")
            print("-" * 40)
            
            # Sort by R² (higher is better)
            best_r2 = type_data.loc[type_data['R²'].idxmax()]
            print(f"Best R² Score: {best_r2['Model']} ({best_r2['R²']:.6f})")
            
            # Sort by RMSE (lower is better)
            best_rmse = type_data.loc[type_data['RMSE'].idxmin()]
            print(f"Lowest RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
            
            # Sort by MAE (lower is better)
            best_mae = type_data.loc[type_data['MAE'].idxmin()]
            print(f"Lowest MAE: {best_mae['Model']} ({best_mae['MAE']:.4f})")
            
            # Sort by MAPE (lower is better)
            best_mape = type_data.loc[type_data['MAPE'].idxmin()]
            print(f"Lowest MAPE: {best_mape['Model']} ({best_mape['MAPE']:.2f}%)")
        
        print("\n" + "="*80)
        print("DETAILED RESULTS:")
        print("="*80)
        print(results_df.to_string(index=False, float_format='%.6f'))
        print("="*80)


def main():
    """Main function for standalone execution"""
    evaluator = ModelEvaluator()
    success = evaluator.evaluate_models()
    
    if success:
        print("\n✅ Model evaluation completed successfully!")
    else:
        print("\n❌ Model evaluation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
