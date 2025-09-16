#!/usr/bin/env python3
"""
Dubai Metro Prediction System - Unified Main Interface
Complete pipeline for data processin        # Run evaluation
        evaluator = ModelEvaluator()
        success = evaluator.evaluate_models()
        
        if success:
            print("\n🎉 SUCCESS! Model evaluation completed.")
            print("📁 Results saved to: outputs/testing/")
            return True
        else:
            print("\n❌ Model evaluation failed!")
            return Falsevaluation, and prediction

Usage:
    python dubai_metro_system.py --help
    python dubai_metro_system.py process-data --type train
    python dubai_metro_system.py process-data --type test
    python dubai_metro_system.py train-models
    python dubai_metro_system.py evaluate-models  
    python dubai_metro_system.py predict --station "BurJuman Metro Station" --datetime "2025-09-20 18:00"
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Add current directory and utils to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'utils'))

from config import DubaiMetroConfig
from utils.base_processor import BaseDubaiMetroProcessor
from utils.train_model_pipeline import MetroPredictionPipeline
from utils.model_evaluator import ModelEvaluator
from model_predictor import MetroPredictor


class DubaiMetroSystemInterface:
    """Unified interface for the complete Dubai Metro prediction system"""
    
    def __init__(self):
        """Initialize the system interface"""
        self.config = DubaiMetroConfig()
        
    def process_data(self, data_type='train'):
        """
        Process raw CSV data into clean datasets
        
        Args:
            data_type: 'train' or 'test'
        """
        print(f"🚇 DUBAI METRO DATA PROCESSING - {data_type.upper()}")
        print("="*60)
        
        if data_type == 'train':
            csv_folder = self.config.PATHS['train_csv_folder']
            output_file = self.config.PATHS['train_output']
        else:
            csv_folder = self.config.PATHS['test_csv_folder']
            output_file = self.config.PATHS['test_output']
        
        # Verify CSV folder exists
        if not os.path.exists(csv_folder):
            print(f"❌ CSV folder not found: {csv_folder}")
            return False
        
        # Create processor
        processor = BaseDubaiMetroProcessor(
            csv_folder_path=csv_folder,
            output_path=output_file,
            dataset_type=data_type
        )
        
        # Run processing
        success = processor.process_all_files()
        
        if success:
            print(f"\n🎉 SUCCESS! {data_type.title()} dataset created.")
            print(f"📁 File: {output_file}")
        else:
            print(f"\n❌ {data_type.title()} processing failed. Check logs for details.")
        
        return success
    
    def train_models(self):
        """Train all machine learning models"""
        print("🚇 DUBAI METRO MODEL TRAINING")
        print("="*60)
        
        # Check if training data exists
        train_file = self.config.PATHS['train_output']
        if not os.path.exists(train_file):
            print(f"❌ Training data not found: {train_file}")
            print("💡 Run data processing first: python dubai_metro_system.py process-data --type train")
            return False
        
        # Initialize and run training pipeline
        pipeline = MetroPredictionPipeline(dataset_path=train_file)
        results = pipeline.run_complete_pipeline()
        
        if results:
            print("\n🎉 SUCCESS! Models trained and saved.")
            print(f"📁 Models saved to: {self.config.PATHS['models_dir']}")
            return True
        else:
            print("\n❌ Model training failed.")
            return False
    
    def evaluate_models(self):
        """Evaluate trained models on test data"""
        print("🚇 DUBAI METRO MODEL EVALUATION")
        print("="*60)
        
        # Check if test data exists
        test_file = self.config.PATHS['test_output']
        if not os.path.exists(test_file):
            print(f"❌ Test data not found: {test_file}")
            print("💡 Run data processing first: python dubai_metro_system.py process-data --type test")
            return False
        
        # Check if models exist
        models_dir = Path(self.config.PATHS['models_dir'])
        if not models_dir.exists() or not any(models_dir.glob('*.joblib')):
            print(f"❌ No trained models found in: {models_dir}")
            print("💡 Train models first: python dubai_metro_system.py train-models")
            return False
        
        # Run evaluation
        evaluator = ModelEvaluator()
        success = evaluator.evaluate_models()
        
        if success:
            print("\n🎉 SUCCESS! Model evaluation completed.")
            print(f"📁 Results saved to: {self.config.PATHS['testing_outputs']}")
            return True
        else:
            print("\n❌ Model evaluation failed.")
            return False
    
    def make_prediction(self, station_name, target_datetime, model_name='random_forest'):
        """
        Make a single prediction
        
        Args:
            station_name: Name of the metro station
            target_datetime: Target datetime string (YYYY-MM-DD HH:MM)
            model_name: Model to use for prediction
        """
        print("🚇 DUBAI METRO PREDICTION")
        print("="*60)
        
        # Check if models exist
        models_dir = Path(self.config.PATHS['models_dir'])
        if not models_dir.exists() or not any(models_dir.glob('*.joblib')):
            print(f"❌ No trained models found in: {models_dir}")
            print("💡 Train models first: python dubai_metro_system.py train-models")
            return None
        
        try:
            # Parse datetime
            dt = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M')
            
            # Initialize predictor
            predictor = MetroPredictor()
            
            # Make prediction
            result = predictor.predict_single(
                station_name=station_name,
                target_datetime=dt,
                model_name=model_name
            )
            
            # Display results
            print(f"\n📍 PREDICTION RESULTS:")
            print(f"🚉 Station: {result['station']}")
            print(f"📅 Date/Time: {result['datetime'].strftime('%Y-%m-%d %H:%M')}")
            print(f"👥 Predicted Check-ins: {result['predicted_checkin']:.0f}")
            print(f"👥 Predicted Check-outs: {result['predicted_checkout']:.0f}")
            print(f"⚡ Model Used: {result['model_used']}")
            print(f"🕒 Operational: {'Yes' if result['is_operational'] else 'No'}")
            
            return result
            
        except ValueError as e:
            print(f"❌ Invalid datetime format. Use: YYYY-MM-DD HH:MM")
            return None
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete end-to-end pipeline"""
        print("🚇 DUBAI METRO COMPLETE PIPELINE")
        print("="*60)
        print("Running complete end-to-end pipeline...")
        
        # Step 1: Process training data
        print("\n📊 Step 1: Processing training data...")
        if not self.process_data('train'):
            return False
        
        # Step 2: Process test data
        print("\n📊 Step 2: Processing test data...")
        if not self.process_data('test'):
            return False
        
        # Step 3: Train models
        print("\n🤖 Step 3: Training models...")
        if not self.train_models():
            return False
        
        # Step 4: Evaluate models
        print("\n📈 Step 4: Evaluating models...")
        if not self.evaluate_models():
            return False
        
        print("\n🎉 COMPLETE PIPELINE SUCCESS!")
        print("="*60)
        print("✅ Training data processed")
        print("✅ Test data processed") 
        print("✅ Models trained and saved")
        print("✅ Models evaluated on test data")
        print("\n🚀 System ready for predictions!")
        print("📖 Example: python dubai_metro_system.py predict --station 'BurJuman Metro Station' --datetime '2025-09-20 18:00'")
        
        return True
    
    def show_status(self):
        """Show current system status"""
        print("🚇 DUBAI METRO SYSTEM STATUS")
        print("="*60)
        
        # Check data files
        train_file = Path(self.config.PATHS['train_output'])
        test_file = Path(self.config.PATHS['test_output'])
        models_dir = Path(self.config.PATHS['models_dir'])
        
        print("📊 DATA STATUS:")
        print(f"   Training Data: {'✅ Ready' if train_file.exists() else '❌ Missing'}")
        if train_file.exists():
            size_mb = train_file.stat().st_size / (1024*1024)
            print(f"                  {size_mb:.1f} MB")
        
        print(f"   Test Data:     {'✅ Ready' if test_file.exists() else '❌ Missing'}")
        if test_file.exists():
            size_mb = test_file.stat().st_size / (1024*1024)
            print(f"                  {size_mb:.1f} MB")
        
        print("\n🤖 MODELS STATUS:")
        if models_dir.exists():
            model_files = list(models_dir.glob('*.joblib'))
            if model_files:
                print(f"   Models: ✅ {len(model_files)} files ready")
                for model_file in sorted(model_files)[:5]:  # Show first 5
                    print(f"           - {model_file.name}")
                if len(model_files) > 5:
                    print(f"           ... and {len(model_files) - 5} more")
            else:
                print("   Models: ❌ No models found")
        else:
            print("   Models: ❌ Models directory missing")
        
        print("\n🎯 SYSTEM READINESS:")
        ready_for_training = train_file.exists()
        ready_for_evaluation = test_file.exists() and models_dir.exists()
        ready_for_prediction = models_dir.exists() and any(models_dir.glob('*.joblib'))
        
        print(f"   Training:   {'✅ Ready' if ready_for_training else '❌ Need training data'}")
        print(f"   Evaluation: {'✅ Ready' if ready_for_evaluation else '❌ Need test data & models'}")
        print(f"   Prediction: {'✅ Ready' if ready_for_prediction else '❌ Need trained models'}")


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Dubai Metro Prediction System - Complete ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s status                                    # Show system status
  %(prog)s process-data --type train                 # Process training data
  %(prog)s process-data --type test                  # Process test data  
  %(prog)s train-models                             # Train ML models
  %(prog)s evaluate-models                          # Evaluate on test data
  %(prog)s predict --station "BurJuman Metro Station" --datetime "2025-09-20 18:00"
  %(prog)s run-pipeline                             # Run complete pipeline
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Process data command
    process_parser = subparsers.add_parser('process-data', help='Process raw CSV data')
    process_parser.add_argument('--type', choices=['train', 'test'], required=True,
                              help='Type of data to process')
    
    # Train models command
    subparsers.add_parser('train-models', help='Train machine learning models')
    
    # Evaluate models command
    subparsers.add_parser('evaluate-models', help='Evaluate models on test data')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--station', required=True, help='Station name')
    predict_parser.add_argument('--datetime', required=True, help='Target datetime (YYYY-MM-DD HH:MM)')
    predict_parser.add_argument('--model', default='random_forest', 
                               choices=['linear_regression', 'random_forest', 'lightgbm', 'gradient_boosting'],
                               help='Model to use for prediction')
    
    # Complete pipeline command
    subparsers.add_parser('run-pipeline', help='Run complete end-to-end pipeline')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize system
    system = DubaiMetroSystemInterface()
    
    # Execute commands
    if args.command == 'status':
        system.show_status()
    
    elif args.command == 'process-data':
        system.process_data(args.type)
    
    elif args.command == 'train-models':
        system.train_models()
    
    elif args.command == 'evaluate-models':
        system.evaluate_models()
    
    elif args.command == 'predict':
        system.make_prediction(args.station, args.datetime, args.model)
    
    elif args.command == 'run-pipeline':
        system.run_complete_pipeline()


if __name__ == "__main__":
    main()
