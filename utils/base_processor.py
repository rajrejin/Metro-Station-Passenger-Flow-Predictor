"""
Dubai Metro Prediction System - Base Data Processor
Shared functionality for training and test dataset processors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import logging
import time
import gc
import sys
from pathlib import Path
import sqlite3
import tempfile
import csv
import psutil
from collections import defaultdict

from config import DubaiMetroConfig, DubaiMetroUtils


class BaseDubaiMetroProcessor:
    """
    Base class for Dubai Metro dataset processors
    Contains shared functionality for training and test processors
    """
    
    def __init__(self, csv_folder_path, output_path, dataset_type='train'):
        """
        Initialize the base processor
        
        Args:
            csv_folder_path: Path to CSV files
            output_path: Output file path
            dataset_type: 'train' or 'test'
        """
        self.csv_folder_path = csv_folder_path
        self.output_path = output_path
        self.dataset_type = dataset_type
        
        # Get configuration based on dataset type
        if dataset_type == 'train':
            config = DubaiMetroConfig.TRAINING_CONFIG
            log_file = f'{DubaiMetroConfig.PATHS["training_outputs"]}/train_metro_processing.log'
        else:
            config = DubaiMetroConfig.TEST_CONFIG
            log_file = f'{DubaiMetroConfig.PATHS["testing_outputs"]}/test_metro_processing.log'
        
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.missing_dates = config['missing_dates']
        
        # Setup logging
        self.setup_logging(log_file)
        
        # Database setup
        self.temp_db_path = tempfile.mktemp(suffix='.db')
        self.init_database()
        
        # Progress tracking
        self.processed_files = 0
        self.total_files = 0
        self.failed_files = []
        
    def setup_logging(self, log_file):
        """Setup logging configuration"""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True  # Override existing configuration
        )
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """Initialize SQLite database with performance optimizations"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Apply performance settings
        DubaiMetroUtils.setup_database_performance(cursor)
        
        # Create table for hourly aggregates
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_aggregates (
                station TEXT,
                date TEXT,
                hour INTEGER,
                checkin_count INTEGER DEFAULT 0,
                checkout_count INTEGER DEFAULT 0,
                PRIMARY KEY (station, date, hour)
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info(f"{self.dataset_type.title()} database initialized: {self.temp_db_path}")
    
    def check_memory_usage(self):
        """Monitor memory usage and perform cleanup if needed"""
        try:
            process = psutil.Process(os.getpid())
            memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            
            if memory_gb > DubaiMetroConfig.EMERGENCY_MEMORY_GB:
                self.logger.error(f"CRITICAL MEMORY: {memory_gb:.2f} GB - STOPPING")
                return False
            elif memory_gb > DubaiMetroConfig.MAX_MEMORY_GB:
                self.logger.warning(f"HIGH MEMORY: {memory_gb:.2f} GB - cleaning up")
                gc.collect()
            
            return True
        except:
            gc.collect()
            return True
    
    def process_csv_file(self, csv_file):
        """Process a single CSV file with enhanced logic"""
        filename = os.path.basename(csv_file)
        file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
        
        self.logger.info(f"Processing {filename} ({file_size:.1f} MB)")
        
        # Extract expected date from filename
        try:
            date_part = filename.replace("Metro_Ridership_", "").split("_")[0]
            expected_date = datetime.strptime(date_part, '%Y-%m-%d')
        except:
            self.logger.error(f"Could not parse date from filename: {filename}")
            return False
        
        # Skip if date is in missing dates
        date_str = expected_date.strftime('%Y-%m-%d')
        if date_str in self.missing_dates:
            self.logger.info(f"Skipping {filename} - known missing date")
            return True
        
        # In-memory aggregation
        aggregates = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'checkin': 0, 'checkout': 0})))
        
        processed_rows = 0
        valid_rows = 0
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(csv_file, 'r', encoding=encoding, errors='ignore') as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        
                        # Map column indices
                        col_map = {col.lower().strip(): i for i, col in enumerate(headers)}
                        
                        # Required columns
                        required_cols = ['txn_date', 'txn_time', 'start_location', 'end_location']
                        if not all(col in col_map for col in required_cols):
                            self.logger.warning(f"Missing required columns in {filename}")
                            continue
                        
                        # Process rows
                        for row_num, row in enumerate(reader):
                            processed_rows += 1
                            
                            # Memory check every 10000 rows
                            if processed_rows % 10000 == 0:
                                if not self.check_memory_usage():
                                    self.logger.error(f"Memory critical, stopping {filename}")
                                    break
                            
                            # Skip if row is too short
                            if len(row) <= max(col_map.values()):
                                continue
                            
                            try:
                                # Extract data
                                txn_date = row[col_map['txn_date']].strip()
                                txn_time = row[col_map['txn_time']].strip()
                                start_loc = row[col_map['start_location']].strip()
                                end_loc = row[col_map['end_location']].strip()
                                
                                # Basic validation
                                if not all([txn_date, txn_time, start_loc, end_loc]):
                                    continue
                                
                                if start_loc in ['nan', 'NaN', ''] or end_loc in ['nan', 'NaN', '']:
                                    continue
                                
                                # Parse date
                                try:
                                    date_obj = datetime.strptime(txn_date, '%Y-%m-%d')
                                except:
                                    try:
                                        date_obj = datetime.strptime(txn_date, '%d/%m/%Y')
                                    except:
                                        continue
                                
                                # Verify date matches filename
                                if date_obj.date() != expected_date.date():
                                    continue
                                
                                # Parse time and extract hour
                                try:
                                    time_obj = datetime.strptime(txn_time, '%H:%M:%S')
                                    hour = time_obj.hour
                                except:
                                    try:
                                        hour = int(txn_time.split(':')[0])
                                    except:
                                        continue
                                
                                # Note: We keep all data regardless of operational hours
                                # The is_operational flag will be set during dataset creation
                                
                                date_str = date_obj.strftime('%Y-%m-%d')
                                
                                # Normalize station names
                                start_loc = DubaiMetroUtils.normalize_station_name(start_loc)
                                end_loc = DubaiMetroUtils.normalize_station_name(end_loc)
                                
                                # Aggregate data
                                aggregates[start_loc][date_str][hour]['checkin'] += 1
                                aggregates[end_loc][date_str][hour]['checkout'] += 1
                                
                                valid_rows += 1
                                
                            except Exception as e:
                                # Skip problematic rows
                                continue
                        
                        break  # Successfully processed
                        
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error with encoding {encoding}: {e}")
                    continue
            
            if valid_rows == 0:
                self.logger.warning(f"No valid data in {filename}")
                return False
            
            # Write to database
            self.write_aggregates_to_database(aggregates)
            
            self.logger.info(f"SUCCESS: {filename} - {valid_rows:,} valid rows from {processed_rows:,} total")
            return True
            
        except Exception as e:
            self.logger.error(f"FAILED: {filename} - {str(e)}")
            self.failed_files.append(filename)
            return False
    
    def write_aggregates_to_database(self, aggregates):
        """Write aggregated data to SQLite database"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        batch_data = []
        for station, dates in aggregates.items():
            for date, hours in dates.items():
                for hour, counts in hours.items():
                    batch_data.append((
                        station, date, hour,
                        counts['checkin'], counts['checkout']
                    ))
        
        # Insert with conflict resolution
        cursor.executemany('''
            INSERT INTO hourly_aggregates (station, date, hour, checkin_count, checkout_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(station, date, hour) DO UPDATE SET
                checkin_count = checkin_count + excluded.checkin_count,
                checkout_count = checkout_count + excluded.checkout_count
        ''', batch_data)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Wrote {len(batch_data):,} aggregate records to database")
        
        # Cleanup memory
        del aggregates
        del batch_data
        gc.collect()
    
    def create_enhanced_dataset(self):
        """Create the final enhanced dataset with proper operating hours handling"""
        self.logger.info(f"Creating enhanced {self.dataset_type} dataset...")
        
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Get all unique stations
        cursor.execute("SELECT DISTINCT station FROM hourly_aggregates ORDER BY station")
        stations = [row[0] for row in cursor.fetchall()]
        self.logger.info(f"Found {len(stations)} unique stations")
        
        # Print all stations for verification
        self.logger.info("Station list:")
        for i, station in enumerate(stations, 1):
            self.logger.info(f"  {i:2d}. {station}")
        
        # Generate complete date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Calculate total records
        total_records = len(stations) * len(date_range) * 24
        self.logger.info(f"Creating {total_records:,} total records...")
        
        # Create the enhanced dataset
        with open(self.output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['station', 'date', 'hour', 'checkin_count', 'checkout_count', 'is_operational'])
            
            records_written = 0
            
            for station_idx, station in enumerate(stations):
                if station_idx % 10 == 0:
                    self.logger.info(f"Processing station {station_idx+1}/{len(stations)}: {station}")
                
                for date in date_range:
                    date_str = date.strftime('%Y-%m-%d')
                    date_obj = date
                    
                    if date_str in self.missing_dates:
                        # Missing dates: all hours get zero values with operational flag
                        for hour in range(24):
                            is_operational = DubaiMetroUtils.is_operational_hour(date_obj, hour)
                            writer.writerow([station, date_str, hour, 0, 0, is_operational])
                            records_written += 1
                    else:
                        # Get actual data for this station-date
                        cursor.execute('''
                            SELECT hour, checkin_count, checkout_count 
                            FROM hourly_aggregates 
                            WHERE station = ? AND date = ?
                        ''', (station, date_str))
                        
                        actual_data = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
                        
                        # Write all 24 hours with operational flag
                        for hour in range(24):
                            is_operational = DubaiMetroUtils.is_operational_hour(date_obj, hour)
                            
                            if hour in actual_data:
                                checkin, checkout = actual_data[hour]
                                writer.writerow([station, date_str, hour, checkin, checkout, is_operational])
                            else:
                                # No data for this hour
                                writer.writerow([station, date_str, hour, 0, 0, is_operational])
                            
                            records_written += 1
                
                # Progress update
                if station_idx % 10 == 0:
                    progress = (station_idx / len(stations)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({records_written:,} records)")
        
        conn.close()
        self.logger.info(f"Enhanced {self.dataset_type} dataset created: {records_written:,} records in {self.output_path}")
        
        return records_written
    
    def process_all_files(self):
        """Main processing pipeline"""
        start_time = time.time()
        self.logger.info(f"Starting {self.dataset_type} Dubai Metro data processing...")
        
        # Get CSV files
        csv_files = glob.glob(os.path.join(self.csv_folder_path, "*.csv"))
        self.total_files = len(csv_files)
        
        if not csv_files:
            self.logger.error("No CSV files found!")
            return False
        
        self.logger.info(f"Found {self.total_files} CSV files")
        
        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in csv_files) / (1024 * 1024 * 1024)
        self.logger.info(f"Total data size: {total_size:.2f} GB")
        
        # Sort files by size
        csv_files.sort(key=os.path.getsize)
        
        # Process files
        successful_files = 0
        
        for i, csv_file in enumerate(csv_files):
            self.logger.info(f"Processing file {i+1}/{self.total_files}")
            
            if not self.check_memory_usage():
                self.logger.error("Memory limit reached")
                break
            
            if self.process_csv_file(csv_file):
                successful_files += 1
            
            # Progress update (more frequent for smaller test dataset)
            update_freq = 5 if self.dataset_type == 'test' else 20
            if i % update_freq == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (self.total_files - i - 1) / rate if rate > 0 else 0
                
                progress = (i + 1) / self.total_files * 100
                self.logger.info(f"Overall Progress: {progress:.1f}% - ETA: {remaining/60:.1f} minutes")
        
        self.logger.info(f"File processing complete: {successful_files}/{self.total_files} successful")
        
        if successful_files == 0:
            self.logger.error("No files processed successfully!")
            return False
        
        # Create enhanced dataset
        records_written = self.create_enhanced_dataset()
        
        # Cleanup
        try:
            os.remove(self.temp_db_path)
        except:
            pass
        
        # Final summary
        total_time = time.time() - start_time
        self.logger.info("="*70)
        self.logger.info(f"{self.dataset_type.upper()} DUBAI METRO PROCESSING COMPLETE!")
        self.logger.info("="*70)
        self.logger.info(f"✅ Output file: {self.output_path}")
        self.logger.info(f"✅ Total records: {records_written:,}")
        self.logger.info(f"✅ Processing time: {total_time/60:.1f} minutes")
        self.logger.info(f"✅ Successful files: {successful_files}/{self.total_files}")
        self.logger.info(f"✅ Features:")
        self.logger.info(f"   • Respects Dubai Metro operating hours")
        self.logger.info(f"   • Consistent station naming")
        self.logger.info(f"   • Includes operational status flag")
        
        if self.failed_files:
            self.logger.warning(f"⚠️  Failed files: {len(self.failed_files)}")
            for failed_file in self.failed_files[:5]:  # Show first 5
                self.logger.warning(f"   - {failed_file}")
            if len(self.failed_files) > 5:
                self.logger.warning(f"   ... and {len(self.failed_files) - 5} more")
        
        self.logger.info("="*70)
        
        return True
