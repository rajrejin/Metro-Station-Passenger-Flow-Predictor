"""
Dubai Metro Prediction System - Configuration Module
Centralized configuration for consistent settings across all components
"""

from datetime import datetime

class DubaiMetroConfig:
    """Centralized configuration for Dubai Metro system"""
    
    # Dubai Metro Operating Hours
    OPERATING_HOURS = {
        'monday': {'start': 5, 'end': 24},      # 5 AM - 12 AM (midnight)
        'tuesday': {'start': 5, 'end': 24},     # 5 AM - 12 AM (midnight)
        'wednesday': {'start': 5, 'end': 24},   # 5 AM - 12 AM (midnight)  
        'thursday': {'start': 5, 'end': 24},    # 5 AM - 12 AM (midnight)
        'friday': {'start': 5, 'end': 25},      # 5 AM - 1 AM (next day)
        'saturday': {'start': 5, 'end': 24},    # 5 AM - 12 AM (midnight)
        'sunday': {'start': 8, 'end': 24}       # 8 AM - 12 AM (midnight)
    }
    
    # Station name mappings for consistency
    STATION_NAME_MAPPINGS = {
        # Sponsor rebranding (latest name gets priority, original in parentheses)
        'Al Khail Metro Station': 'Al Fardan Exchange (Al Khail) Metro Station',
        'Jabal Ali Metro Station': 'National Paints (Jabal Ali) Metro Station',
        'Mashreq Metro Station': 'InsuranceMarket.ae (Mashreq) Metro Station',
        'UAE Exchange Metro Station': 'Life Pharmacy (UAE Exchange) Metro Station',
        
        # Spacing fixes
        'Al Qusais  Metro Station': 'Al Qusais Metro Station',
        'Dubai Internet City  Metro Station': 'Dubai Internet City Metro Station',
        'Energy  Metro Station': 'Energy Metro Station', 
        'GGICO  Metro Station': 'GGICO Metro Station',
        'Union  Metro Station': 'Union Metro Station',
        
        # Etisalat consolidation
        'Etisalat Metro Station': 'etisalat by e& Metro Station',
        
        # Case corrections
        'centrepoint Metro Station': 'Centrepoint Metro Station',
        'max Metro Station': 'max Metro Station'  # Already correct
    }
    
    # Memory management settings
    MAX_MEMORY_GB = 3.0
    EMERGENCY_MEMORY_GB = 4.0
    
    # Database performance settings
    DB_PRAGMAS = {
        "synchronous": "OFF",
        "journal_mode": "OFF", 
        "cache_size": "200000",
        "temp_store": "MEMORY",
        "page_size": "65536",
        "mmap_size": "1073741824"
    }
    
    # Feature engineering settings
    LAG_WINDOWS = [1, 2, 3, 24]  # 1-3 hours ago, same time yesterday
    ROLLING_WINDOWS = [3, 6, 24]  # 3-hour, 6-hour, daily windows
    
    # Training data configuration
    TRAINING_CONFIG = {
        'start_date': datetime(2024, 7, 1),
        'end_date': datetime(2025, 6, 30),
        'missing_dates': {
            '2024-09-17', '2024-09-20', '2024-09-29', '2024-10-03', 
            '2024-09-06', '2025-05-19', '2025-05-21', '2025-06-20', 
            '2025-06-29'
        }
    }
    
    # Test data configuration  
    TEST_CONFIG = {
        'start_date': datetime(2025, 7, 3),
        'end_date': datetime(2025, 8, 22),
        'missing_dates': {
            # July 2025 missing dates
            '2025-07-04', '2025-07-06', '2025-07-07', '2025-07-08', '2025-07-09', 
            '2025-07-10', '2025-07-11', '2025-07-12', '2025-07-14', '2025-07-16', 
            '2025-07-18', '2025-07-23', '2025-07-24', '2025-07-26', 
            # August 2025 missing dates  
            '2025-08-05', '2025-08-19', '2025-08-21'
        }
    }
    
    # Paths
    PATHS = {
        'train_csv_folder': r"data\csv_files\Train",
        'test_csv_folder': r"data\csv_files\Test",
        'train_output': "data/csv_files/dubai_metro_hourly_dataset_train.csv",
        'test_output': "data/csv_files/dubai_metro_hourly_dataset_test.csv",
        'models_dir': "models",
        'training_outputs': "outputs/training",
        'testing_outputs': "outputs/testing"
    }


class DubaiMetroUtils:
    """Utility functions for Dubai Metro operations"""
    
    @staticmethod
    def is_operational_hour(date_obj, hour):
        """Check if metro is operational at given date and hour"""
        day_name = date_obj.strftime('%A').lower()
        
        if day_name not in DubaiMetroConfig.OPERATING_HOURS:
            return False
            
        start_hour = DubaiMetroConfig.OPERATING_HOURS[day_name]['start']
        end_hour = DubaiMetroConfig.OPERATING_HOURS[day_name]['end']
        
        # Handle end hour > 24 (Friday 1 AM next day)
        if end_hour > 24:
            return hour >= start_hour or hour < (end_hour - 24)
        else:
            return start_hour <= hour < end_hour
    
    @staticmethod
    def normalize_station_name(station_name):
        """Normalize station name using mappings"""
        station_name = station_name.strip()
        return DubaiMetroConfig.STATION_NAME_MAPPINGS.get(station_name, station_name)
    
    @staticmethod
    def setup_database_performance(cursor):
        """Apply performance optimizations to SQLite database"""
        for pragma, value in DubaiMetroConfig.DB_PRAGMAS.items():
            cursor.execute(f"PRAGMA {pragma} = {value}")
