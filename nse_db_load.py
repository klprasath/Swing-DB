import os
import time
import json
import pandas as pd
import psycopg2
from psycopg2 import Error
from datetime import datetime
import logging
from io import StringIO
import csv

class NSEDatabaseLoad:
    def __init__(self, download_path=r"C:\Python Code\Files"):
        self.download_path = download_path
        self.extracted_files = {"pd_file": None, "mcap_file": None, "highlow_file": None}
        self.parsed_date = None
        
        # Configure logging
        self.setup_logging()
        
        # Database configuration (fallback if db_connection module not available)
        self.db_config = {
            "user": "postgres",
            "password": "anirudh16",
            "host": "localhost",
            "port": "5432",
            "database": "swingdb"
        }
        
        # Initialize database connection method
        self.use_db_connection_module = False
        self.db_connection_func = None
        
        # Try to import and use db_connection module
        try:
            from db_connection import get_db_connection
            self.db_connection_func = get_db_connection
            self.use_db_connection_module = True
            print("✅ Using db_connection module for database connections")
            self.logger.info("Using db_connection module for database connections")
        except ImportError:
            print("⚠️ db_connection module not found, using direct database connection")
            self.logger.warning("db_connection module not found, using direct database connection")
            self.use_db_connection_module = False
    
    def setup_logging(self):
        """Setup consolidated logging for database script"""
        # Create consolidated logger
        self.logger = logging.getLogger('nse_database_load')
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler('nse_database_load.log')
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
        # Log start of new session
        self.logger.info("=" * 80)
        self.logger.info("NSE DATABASE LOAD SESSION STARTED")
        self.logger.info("=" * 80)
    
    def load_extraction_info(self):
        """Load extracted file information from JSON created by download script"""
        try:
            info_file = os.path.join(self.download_path, "extraction_info.json")
            
            if not os.path.exists(info_file):
                print(f"❌ extraction_info.json not found in {self.download_path}")
                print("   Make sure to run nse_download_extract.py first!")
                self.logger.error("extraction_info.json not found - run download script first")
                return False
            
            with open(info_file, 'r') as f:
                extraction_info = json.load(f)
            
            # Load extracted file information
            self.extracted_files = extraction_info.get("extracted_files", {})
            self.parsed_date = extraction_info.get("parsed_date")
            
            print(f"📋 Loaded extraction info:")
            print(f"   • PD file: {os.path.basename(self.extracted_files.get('pd_file', 'None'))}")
            print(f"   • MCAP file: {os.path.basename(self.extracted_files.get('mcap_file', 'None'))}")
            print(f"   • 52WK file: {os.path.basename(self.extracted_files.get('highlow_file', 'None'))}")
            print(f"   • Parsed date: {self.parsed_date}")
            
            self.logger.info(f"Loaded extraction info - PD: {self.extracted_files.get('pd_file')}")
            self.logger.info(f"Loaded extraction info - MCAP: {self.extracted_files.get('mcap_file')}")
            self.logger.info(f"Loaded extraction info - 52WK: {self.extracted_files.get('highlow_file')}")
            self.logger.info(f"Loaded extraction info - Date: {self.parsed_date}")
            
            # Verify files exist
            missing_files = []
            for file_type, file_path in self.extracted_files.items():
                if file_path and not os.path.exists(file_path):
                    missing_files.append(f"{file_type}: {file_path}")
            
            if missing_files:
                print(f"⚠️ Some extracted files are missing:")
                for missing in missing_files:
                    print(f"   • {missing}")
                self.logger.warning(f"Missing extracted files: {missing_files}")
                # Don't fail completely, just log warning
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading extraction info: {e}")
            self.logger.error(f"Error loading extraction info: {e}")
            return False
    
    def update_batch_status(self, step_name, status):
        """Update single row in batch process tracking table"""
        try:
            # Use parsed_date from file if available, otherwise use current date as fallback
            tracking_date = self.parsed_date if self.parsed_date else datetime.now().date()
            
            if self.use_db_connection_module and self.db_connection_func:
                with self.db_connection_func() as db:
                    # First, get the existing trd_dt and prev_trd_dt before deleting
                    existing_trd_dt = None
                    existing_prev_trd_dt = None
                    try:
                        select_query = "SELECT trd_dt, prev_trd_dt FROM stg_batch_process LIMIT 1"
                        db.execute(select_query)
                        result = db.fetchone()
                        if result:
                            existing_trd_dt = result[0]
                            existing_prev_trd_dt = result[1]
                    except:
                        existing_trd_dt = None
                        existing_prev_trd_dt = None
                    
                    # Determine prev_trd_dt value
                    if existing_trd_dt is None:
                        # First time running - no previous date
                        prev_trd_dt = None
                    elif existing_trd_dt == tracking_date:
                        # Same date: preserve existing prev_trd_dt (prior day's trd_dt)
                        prev_trd_dt = existing_prev_trd_dt
                    else:
                        # Same date as existing - keep the existing prev_trd_dt
                        prev_trd_dt = existing_trd_dt
                    
                    # Always update/insert the single row - delete existing and insert new
                    delete_query = "DELETE FROM stg_batch_process"
                    db.execute(delete_query)
                    
                    # Insert new record with prev_trd_dt
                    insert_query = """
                    INSERT INTO stg_batch_process (step, status, trd_dt, updated, prev_trd_dt)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s)
                    """
                    db.execute(insert_query, (step_name, status, tracking_date, prev_trd_dt))
                    
                    db.commit()
                    self.logger.info(f"BATCH_TRACKING: Updated single row - {step_name} - {status} for date {tracking_date}, prev_trd_dt: {prev_trd_dt}")
            else:
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                cursor = connection.cursor()
                try:
                    # First, get the existing trd_dt and prev_trd_dt before deleting
                    existing_trd_dt = None
                    existing_prev_trd_dt = None
                    try:
                        select_query = "SELECT trd_dt, prev_trd_dt FROM stg_batch_process LIMIT 1"
                        cursor.execute(select_query)
                        result = cursor.fetchone()
                        if result:
                            existing_trd_dt = result[0]
                            existing_prev_trd_dt = result[1]
                    except:
                        existing_trd_dt = None
                        existing_prev_trd_dt = None
                    
                    # Determine prev_trd_dt value
                    if existing_trd_dt is None:
                        # First time running - no previous date
                        prev_trd_dt = None
                    elif existing_trd_dt == tracking_date:
                        # Same date: preserve existing prev_trd_dt (prior day's trd_dt)
                        prev_trd_dt = existing_prev_trd_dt    
                    else:
                        # Same date as existing - keep the existing prev_trd_dt
                        prev_trd_dt = existing_trd_dt
                    
                    # Always update/insert the single row - delete existing and insert new
                    delete_query = "DELETE FROM stg_batch_process"
                    cursor.execute(delete_query)
                    
                    # Insert new record with prev_trd_dt
                    insert_query = """
                    INSERT INTO stg_batch_process (step, status, trd_dt, updated, prev_trd_dt)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s)
                    """
                    cursor.execute(insert_query, (step_name, status, tracking_date, prev_trd_dt))
                    
                    connection.commit()
                    self.logger.info(f"BATCH_TRACKING: Updated single row - {step_name} - {status} for date {tracking_date}, prev_trd_dt: {prev_trd_dt}")
                finally:
                    cursor.close()
                    connection.close()
                    
        except Exception as e:
            self.logger.error(f"BATCH_TRACKING ERROR: Failed to update {step_name} - {status}: {e}")
            # Don't fail the entire pipeline due to batch tracking errors
    
    def check_data_already_in_staging(self):
        """Check if the file date data already exists in stg_nse_data table"""
        if not self.parsed_date:
            return False, "No parsed date available for validation"
            
        try:
            if self.use_db_connection_module and self.db_connection_func:
                with self.db_connection_func() as db:
                    check_query = "SELECT COUNT(*) FROM stg_nse_data WHERE trd_dt = %s"
                    db.execute(check_query, (self.parsed_date,))
                    result = db.fetchone()[0] > 0
                    
                    if result:
                        self.logger.info(f"DATA_VALIDATION: Data for {self.parsed_date} already exists in stg_nse_data")
                        return True, f"Data for {self.parsed_date} already available in staging tables"
                    else:
                        self.logger.info(f"DATA_VALIDATION: Data for {self.parsed_date} not found in stg_nse_data - proceeding with load")
                        return False, f"Data for {self.parsed_date} not found in staging tables"
                        
            else:
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                try:
                    cursor = connection.cursor()
                    check_query = "SELECT COUNT(*) FROM stg_nse_data WHERE trd_dt = %s"
                    cursor.execute(check_query, (self.parsed_date,))
                    result = cursor.fetchone()[0] > 0
                    
                    if result:
                        self.logger.info(f"DATA_VALIDATION: Data for {self.parsed_date} already exists in stg_nse_data")
                        return True, f"Data for {self.parsed_date} already available in staging tables"
                    else:
                        self.logger.info(f"DATA_VALIDATION: Data for {self.parsed_date} not found in stg_nse_data - proceeding with load")
                        return False, f"Data for {self.parsed_date} not found in staging tables"
                        
                finally:
                    cursor.close()
                    connection.close()
                    
        except Exception as e:
            error_msg = f"Error checking if data already in staging: {e}"
            self.logger.error(f"DATA_VALIDATION ERROR: {error_msg}")
            # In case of error, assume not in staging to avoid blocking
            return False, "Validation error - assuming data not in staging"
    
    def check_if_already_processed(self):
        """Check if today's data is already completed by checking Step 5 status"""
        try:
            current_date = datetime.now().date()
            
            if self.use_db_connection_module and self.db_connection_func:
                with self.db_connection_func() as db:
                    check_query = """
                    SELECT COUNT(*) FROM stg_batch_process 
                    WHERE step = 'STEP 5 - PROCESS STOCK_DATA OPERATIONS' 
                    AND status = 'Completed' 
                    AND trd_dt = %s
                    """
                    db.execute(check_query, (current_date,))
                    result = db.fetchone()[0] > 0
                    
                    if result:
                        self.logger.info(f"VALIDATION: Data for {current_date} already processed completely")
                        return True, f"Data for {current_date} already processed completely"
                    else:
                        return False, f"Data for {current_date} not yet processed"
                        
            else:
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                try:
                    cursor = connection.cursor()
                    check_query = """
                    SELECT COUNT(*) FROM stg_batch_process 
                    WHERE step = 'STEP 5 - PROCESS STOCK_DATA OPERATIONS' 
                    AND status = 'Completed' 
                    AND trd_dt = %s
                    """
                    cursor.execute(check_query, (current_date,))
                    result = cursor.fetchone()[0] > 0
                    
                    if result:
                        self.logger.info(f"VALIDATION: Data for {current_date} already processed completely")
                        return True, f"Data for {current_date} already processed completely"
                    else:
                        return False, f"Data for {current_date} not yet processed"
                        
                finally:
                    cursor.close()
                    connection.close()
                    
        except Exception as e:
            error_msg = f"Error checking if data already processed: {e}"
            self.logger.error(f"VALIDATION ERROR: {error_msg}")
            # In case of error, assume not processed to avoid blocking
            return False, "Validation error - assuming not processed"
    
    def load_pd_data_to_database(self):
        """Load Pd data to database (from data_load.py logic) - WITH COMMITS"""
        if not self.extracted_files["pd_file"]:
            print("❌ No Pd file found to load")
            self.logger.error("STEP 2 ERROR: No Pd file found to load")
            return False
        
        file_path = self.extracted_files["pd_file"]
        
        try:
            print(f"\n📊 LOADING PD DATA TO DATABASE")
            print(f"📄 File: {os.path.basename(file_path)}")
            self.logger.info(f"STEP 2: Starting Pd data load from: {os.path.basename(file_path)}")
            
            # Verify file accessibility
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                self.logger.error(f"STEP 2 ERROR: File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File accessible: {file_size:.2f} MB")
            self.logger.info(f"STEP 2: File accessible: {file_size:.2f} MB")
            
            # Handle database connection based on available method
            if self.use_db_connection_module and self.db_connection_func:
                # Use db_connection context manager pattern
                with self.db_connection_func() as db:
                    print("🔌 Connected to PostgreSQL via db_connection module")
                    self.logger.info("STEP 2: Connected to PostgreSQL via db_connection module")
                    
                    # Step 1: Delete from nse_raw table
                    print("🗑️ Deleting records from nse_raw table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM nse_raw;"
                    db.execute(delete_query)
                    db.commit()  # EXPLICIT COMMIT
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted records from nse_raw in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 2: Deleted records from nse_raw in {delete_duration:.2f} seconds")
                    
                    # Step 2: Load CSV file into nse_raw table
                    print("📥 Loading CSV into nse_raw table...")
                    start_time = time.time()
                    
                    # Read CSV and insert row by row (db_connection doesn't support copy_expert)
                    with open(file_path, 'r') as f:
                        # Skip the header row
                        next(f)
                        
                        # Prepare insert query for individual rows
                        insert_row_query = """
                        INSERT INTO nse_raw VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        rows_inserted = 0
                        for line in f:
                            try:
                                # Parse CSV line (handle potential quotes and commas)
                                values = line.strip().split(',')
                                if len(values) >= 16:  # Ensure we have enough columns
                                    # Convert empty strings to None for NULL values
                                    cleaned_values = [val.strip() if val.strip() != '' else None for val in values[:16]]
                                    db.execute(insert_row_query, cleaned_values)
                                    rows_inserted += 1
                                    
                                    # Commit in batches for performance
                                    if rows_inserted % 1000 == 0:
                                        print(f"   📊 Inserted {rows_inserted} rows...")
                                        db.commit()  # BATCH COMMIT
                            except Exception as row_error:
                                print(f"   ⚠️ Skipped row due to error: {row_error}")
                                continue
                        
                        # Final commit for remaining rows
                        db.commit()  # EXPLICIT COMMIT
                        load_duration = time.time() - start_time
                        print(f"✅ Loaded {rows_inserted} rows into nse_raw table in {load_duration:.2f} seconds")
                        self.logger.info(f"STEP 2: Loaded {rows_inserted} rows into nse_raw table in {load_duration:.2f} seconds")
                    
                    # Step 3: Extract date from filename (e.g., Pd290825.csv -> 290825)
                    filename = os.path.basename(file_path)
                    date_str = filename[2:8]  # Extract '290825' from 'Pd290825.csv'
                    parsed_date = datetime.strptime(date_str, '%d%m%y').strftime('%Y-%m-%d')
                    self.parsed_date = parsed_date  # Store for final operations
                    print(f"📅 Parsed date: {parsed_date}")
                    self.logger.info(f"STEP 2: Parsed date: {parsed_date}")
                    
                    # Step 4: Insert data into stg_nse_data
                    print("📊 Inserting into stg_nse_data...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stg_nse_data 
                    SELECT SYMBOL, SECURITY, %s::DATE, PREV_CL_PR, OPEN_PRICE, HIGH_PRICE, 
                           LOW_PRICE, CLOSE_PRICE, NET_TRDQTY, HI_52_WK, LO_52_WK, 
                           NET_TRDVAL, MKT, SERIES, IND_SEC, CORP_IND, TRADES
                    FROM nse_raw
                    WHERE SERIES IN ('BE', 'BZ', 'EQ', 'SM', 'ST', 'SZ')
                    AND symbol IS NOT NULL;
                    """
                    db.execute(insert_query, (parsed_date,))
                    db.commit()  # EXPLICIT COMMIT
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted rows into stg_nse_data in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 2: Inserted rows into stg_nse_data in {insert_duration:.2f} seconds")
                    
                    # Step 5: Insert data into stg_nse_data_hist
                    print("📊 Inserting into stg_nse_data_hist...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stg_nse_data_hist 
                    SELECT SYMBOL, SECURITY, %s::DATE, PREV_CL_PR, OPEN_PRICE, HIGH_PRICE, 
                           LOW_PRICE, CLOSE_PRICE, NET_TRDQTY, HI_52_WK, LO_52_WK, 
                           NET_TRDVAL, MKT, SERIES, IND_SEC, CORP_IND, TRADES
                    FROM nse_raw
                    WHERE SERIES IN ('BE', 'BZ', 'EQ', 'SM', 'ST', 'SZ')
                    AND symbol IS NOT NULL;
                    """
                    db.execute(insert_query, (parsed_date,))
                    db.commit()  # EXPLICIT COMMIT
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted rows into stg_nse_data_hist in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 2: Inserted rows into stg_nse_data_hist in {insert_duration:.2f} seconds")
                    
                print("✅ Pd data load completed successfully")
                self.logger.info("STEP 2: Pd data load completed successfully")
                return True
            
            else:
                # Use traditional psycopg2 connection pattern
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                cursor = connection.cursor()
                print("🔌 Connected to PostgreSQL via direct connection")
                self.logger.info("STEP 2: Connected to PostgreSQL via direct connection")
                
                try:
                    # Step 1: Delete from nse_raw table
                    print("🗑️ Deleting records from nse_raw table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM nse_raw;"
                    cursor.execute(delete_query)
                    row_count = cursor.rowcount
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted {row_count} records in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 2: Deleted {row_count} records from nse_raw in {delete_duration:.2f} seconds")
                    
                    # Step 2: Load CSV file into nse_raw table
                    print("📥 Loading CSV into nse_raw table...")
                    with open(file_path, 'r') as f:
                        # Skip the header row
                        next(f)
                        # Create a StringIO buffer to hold CSV data
                        buffer = StringIO()
                        for line in f:
                            buffer.write(line)
                        buffer.seek(0)
                        
                        start_time = time.time()
                        copy_query = """
                        COPY nse_raw FROM STDIN
                        DELIMITER ',' 
                        CSV
                        NULL ' ';
                        """
                        cursor.copy_expert(copy_query, buffer)
                        load_duration = time.time() - start_time
                        print(f"✅ Loaded CSV into nse_raw table in {load_duration:.2f} seconds")
                        self.logger.info(f"STEP 2: Loaded CSV into nse_raw table in {load_duration:.2f} seconds")
                    
                    # Step 3: Extract date from filename (e.g., Pd290825.csv -> 290825)
                    filename = os.path.basename(file_path)
                    date_str = filename[2:8]  # Extract '290825' from 'Pd290825.csv'
                    parsed_date = datetime.strptime(date_str, '%d%m%y').strftime('%Y-%m-%d')
                    self.parsed_date = parsed_date  # Store for final operations
                    print(f"📅 Parsed date: {parsed_date}")
                    self.logger.info(f"STEP 2: Parsed date: {parsed_date}")
                    
                    # Step 4: Insert data into stg_nse_data
                    print("📊 Inserting into stg_nse_data...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stg_nse_data 
                    SELECT SYMBOL, SECURITY, %s::DATE, PREV_CL_PR, OPEN_PRICE, HIGH_PRICE, 
                           LOW_PRICE, CLOSE_PRICE, NET_TRDQTY, HI_52_WK, LO_52_WK, 
                           NET_TRDVAL, MKT, SERIES, IND_SEC, CORP_IND, TRADES
                    FROM nse_raw
                    WHERE SERIES IN ('BE', 'BZ', 'EQ', 'SM', 'ST', 'SZ')
                    AND symbol IS NOT NULL;
                    """
                    cursor.execute(insert_query, (parsed_date,))
                    row_count = cursor.rowcount
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted {row_count} rows into stg_nse_data in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 2: Inserted {row_count} rows into stg_nse_data in {insert_duration:.2f} seconds")
                    
                    # Step 5: Insert data into stg_nse_data_hist
                    print("📊 Inserting into stg_nse_data_hist...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stg_nse_data_hist 
                    SELECT SYMBOL, SECURITY, %s::DATE, PREV_CL_PR, OPEN_PRICE, HIGH_PRICE, 
                           LOW_PRICE, CLOSE_PRICE, NET_TRDQTY, HI_52_WK, LO_52_WK, 
                           NET_TRDVAL, MKT, SERIES, IND_SEC, CORP_IND, TRADES
                    FROM nse_raw
                    WHERE SERIES IN ('BE', 'BZ', 'EQ', 'SM', 'ST', 'SZ')
                    AND symbol IS NOT NULL;
                    """
                    cursor.execute(insert_query, (parsed_date,))
                    row_count = cursor.rowcount
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted {row_count} rows into stg_nse_data_hist in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 2: Inserted {row_count} rows into stg_nse_data_hist in {insert_duration:.2f} seconds")
                    
                    # Commit transaction
                    connection.commit()
                    print("✅ Pd data load completed successfully")
                    self.logger.info("STEP 2: Pd data load completed successfully")
                    return True
                    
                except Exception as e:
                    connection.rollback()
                    print("🔄 Transaction rolled back")
                    raise e
                finally:
                    cursor.close()
                    connection.close()
                    print("🔌 PostgreSQL connection closed")
                    self.logger.info("STEP 2: PostgreSQL connection closed")
                
        except Exception as e:
            print(f"❌ Error loading Pd data: {e}")
            self.logger.error(f"STEP 2 ERROR: Error loading Pd data: {e}")
            return False
    
    def load_mcap_data_to_database(self):
        """Load MCAP data to database (from mcap_load.py logic) - WITH COMMITS"""
        if not self.extracted_files["mcap_file"]:
            print("❌ No MCAP file found to load")
            self.logger.error("STEP 3 ERROR: No MCAP file found to load")
            return False
        
        file_path = self.extracted_files["mcap_file"]
        
        try:
            print(f"\n💰 LOADING MCAP DATA TO DATABASE")
            print(f"📄 File: {os.path.basename(file_path)}")
            self.logger.info(f"STEP 3: Starting MCAP data load from: {os.path.basename(file_path)}")
            
            # Verify file accessibility
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                self.logger.error(f"STEP 3 ERROR: File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File accessible: {file_size:.2f} MB")
            self.logger.info(f"STEP 3: File accessible: {file_size:.2f} MB")
            
            # Handle database connection based on available method
            if self.use_db_connection_module and self.db_connection_func:
                # Use db_connection context manager pattern
                with self.db_connection_func() as db:
                    print("🔌 Connected to PostgreSQL via db_connection module")
                    self.logger.info("STEP 3: Connected to PostgreSQL via db_connection module")
                    
                    # Step 1: Delete from mcap_raw table
                    print("🗑️ Deleting records from mcap_raw table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM mcap_raw;"
                    db.execute(delete_query)
                    db.commit()  # EXPLICIT COMMIT
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted records from mcap_raw in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 3: Deleted records from mcap_raw in {delete_duration:.2f} seconds")
                    
                    # Step 2: Load CSV file into mcap_raw table (row by row for db_connection)
                    print("📥 Loading CSV into mcap_raw table...")
                    start_time = time.time()
                    
                    # Read CSV and insert row by row (db_connection doesn't support copy_expert)
                    with open(file_path, 'r') as f:
                        # Skip the header row
                        next(f)
                        
                        # Prepare insert query for individual rows
                        insert_row_query = """
                        INSERT INTO mcap_raw VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        rows_inserted = 0
                        for line in f:
                            try:
                                # Parse CSV line (handle potential quotes and commas)
                                values = line.strip().split(',')
                                if len(values) >= 10:  # Ensure we have enough columns
                                    # Convert empty strings to None for NULL values
                                    cleaned_values = [val.strip().strip('"') if val.strip() != '' else None for val in values[:10]]
                                    db.execute(insert_row_query, cleaned_values)
                                    rows_inserted += 1
                                    
                                    # Print progress and commit in batches for performance
                                    if rows_inserted % 1000 == 0:
                                        print(f"   📊 Inserted {rows_inserted} rows...")
                                        db.commit()  # BATCH COMMIT
                            except Exception as row_error:
                                print(f"   ⚠️ Skipped row due to error: {row_error}")
                                continue
                        
                        # Final commit for remaining rows
                        db.commit()  # EXPLICIT COMMIT
                        load_duration = time.time() - start_time
                        print(f"✅ Loaded {rows_inserted} rows into mcap_raw table in {load_duration:.2f} seconds")
                        self.logger.info(f"STEP 3: Loaded {rows_inserted} rows into mcap_raw table in {load_duration:.2f} seconds")
                    
                    # Step 3: Extract date from filename (e.g., MCAP29082025.csv -> 29082025)
                    filename = os.path.basename(file_path)
                    date_str = filename[4:12]  # Extract '29082025' from 'MCAP29082025.csv'
                    parsed_date = datetime.strptime(date_str, '%d%m%Y').strftime('%Y-%m-%d')
                    print(f"📅 Parsed date: {parsed_date}")
                    self.logger.info(f"STEP 3: Parsed date: {parsed_date}")
                    
                    # Step 4: Delete from stg_mcap_data table
                    print("🗑️ Deleting records from stg_mcap_data table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM stg_mcap_data;"
                    db.execute(delete_query)
                    db.commit()  # EXPLICIT COMMIT
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted records from stg_mcap_data in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 3: Deleted records from stg_mcap_data in {delete_duration:.2f} seconds")
                    
                    # Step 5: Insert data from mcap_raw to stg_mcap_data
                    print("📊 Inserting into stg_mcap_data...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stg_mcap_data 
                    SELECT symbol,"Security Name","Market Cap(Rs.)",TO_DATE("Trade Date", 'DD Mon YYYY'),TO_DATE("Last Trade Date", 'DD Mon YYYY'),"Face Value(Rs.)","Issue Size","Close Price/Paid up value(Rs.)",series,category
                    FROM mcap_raw
                    WHERE SERIES IN ('BE', 'BZ', 'EQ', 'SM', 'ST', 'SZ')
                    AND SYMBOL IS NOT NULL;
                    """
                    db.execute(insert_query)
                    db.commit()  # EXPLICIT COMMIT
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted rows into stg_mcap_data in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 3: Inserted rows into stg_mcap_data in {insert_duration:.2f} seconds")
                    
                print("✅ MCAP data load completed successfully")
                self.logger.info("STEP 3: MCAP data load completed successfully")
                return True
            
            else:
                # Use traditional psycopg2 connection pattern
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                cursor = connection.cursor()
                print("🔌 Connected to PostgreSQL via direct connection")
                self.logger.info("STEP 3: Connected to PostgreSQL via direct connection")
                
                try:
                    # Check for table locks on mcap_raw
                    print("🔍 Checking for locks on mcap_raw table...")
                    lock_query = """
                    SELECT pg_class.relname, pg_locks.mode, pg_locks.locktype
                    FROM pg_locks
                    JOIN pg_class ON pg_locks.relation = pg_class.oid
                    WHERE pg_class.relname = 'mcap_raw';
                    """
                    cursor.execute(lock_query)
                    locks = cursor.fetchall()
                    if locks:
                        print(f"⚠️ Locks detected on mcap_raw: {locks}")
                        self.logger.warning(f"STEP 3: Locks detected on mcap_raw: {locks}")
                    else:
                        print("✅ No locks detected on mcap_raw")
                        self.logger.info("STEP 3: No locks detected on mcap_raw")
                    
                    # Step 1: Delete from mcap_raw table
                    print("🗑️ Deleting records from mcap_raw table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM mcap_raw;"
                    cursor.execute(delete_query)
                    row_count = cursor.rowcount
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted {row_count} records in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 3: Deleted {row_count} records from mcap_raw in {delete_duration:.2f} seconds")
                    
                    # Commit after DELETE
                    connection.commit()
                    print("✅ Committed DELETE transaction")
                    
                    # Step 2: Load CSV file into mcap_raw table
                    print("📥 Loading CSV into mcap_raw table...")
                    with open(file_path, 'r') as f:
                        # Skip the header row
                        next(f)
                        # Create a StringIO buffer to hold CSV data
                        buffer = StringIO()
                        for line in f:
                            buffer.write(line)
                        buffer.seek(0)
                        
                        start_time = time.time()
                        copy_query = """
                        COPY mcap_raw
                        FROM STDIN
                        DELIMITER ',' 
                        CSV
                        NULL ' ';
                        """
                        cursor.copy_expert(copy_query, buffer)
                        connection.commit()
                        load_duration = time.time() - start_time
                        print(f"✅ Loaded CSV into mcap_raw table in {load_duration:.2f} seconds")
                        self.logger.info(f"STEP 3: Loaded CSV into mcap_raw table in {load_duration:.2f} seconds")
                    
                    # Step 3: Extract date from filename (e.g., MCAP29082025.csv -> 29082025)
                    filename = os.path.basename(file_path)
                    date_str = filename[4:12]  # Extract '29082025' from 'MCAP29082025.csv'
                    parsed_date = datetime.strptime(date_str, '%d%m%Y').strftime('%Y-%m-%d')
                    print(f"📅 Parsed date: {parsed_date}")
                    self.logger.info(f"STEP 3: Parsed date: {parsed_date}")
                    
                    # Step 4: Delete from stg_mcap_data table
                    print("🗑️ Deleting records from stg_mcap_data table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM stg_mcap_data;"
                    cursor.execute(delete_query)
                    row_count = cursor.rowcount
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted {row_count} records in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 3: Deleted {row_count} records from stg_mcap_data in {delete_duration:.2f} seconds")
                    
                    # Step 5: Insert data from mcap_raw to stg_mcap_data
                    print("📊 Inserting into stg_mcap_data...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stg_mcap_data 
                    SELECT symbol,"Security Name","Market Cap(Rs.)",TO_DATE("Trade Date", 'DD Mon YYYY'),TO_DATE("Last Trade Date", 'DD Mon YYYY'),"Face Value(Rs.)","Issue Size","Close Price/Paid up value(Rs.)",series,category
                    FROM mcap_raw
                    WHERE SERIES IN ('BE', 'BZ', 'EQ', 'SM', 'ST', 'SZ')
                    AND SYMBOL IS NOT NULL;
                    """
                    cursor.execute(insert_query)
                    row_count = cursor.rowcount
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted {row_count} rows into stg_mcap_data in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 3: Inserted {row_count} rows into stg_mcap_data in {insert_duration:.2f} seconds")
                    
                    # Commit transaction
                    connection.commit()
                    print("✅ MCAP data load completed successfully")
                    self.logger.info("STEP 3: MCAP data load completed successfully")
                    return True
                    
                except Exception as e:
                    connection.rollback()
                    print("🔄 Transaction rolled back")
                    raise e
                finally:
                    cursor.close()
                    connection.close()
                    print("🔌 PostgreSQL connection closed")
                    self.logger.info("STEP 3: PostgreSQL connection closed")
                
        except Exception as e:
            print(f"❌ Error loading MCAP data: {e}")
            self.logger.error(f"STEP 3 ERROR: Error loading MCAP data: {e}")
            return False
    
    def load_52wk_data_to_database(self):
        """Load 52 Week High Low data to database - WITH COMMITS"""
        if not self.extracted_files["highlow_file"]:
            print("❌ No 52 Week High Low file found to load")
            self.logger.error("STEP 4 ERROR: No 52 Week High Low file found to load")
            return False
        
        file_path = self.extracted_files["highlow_file"]
        
        try:
            print(f"\n📈 LOADING 52 WEEK HIGH LOW DATA TO DATABASE")
            print(f"📄 File: {os.path.basename(file_path)}")
            self.logger.info(f"STEP 4: Starting 52 Week High Low data load from: {os.path.basename(file_path)}")
            
            # Verify file accessibility
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                self.logger.error(f"STEP 4 ERROR: File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File accessible: {file_size:.2f} MB")
            self.logger.info(f"STEP 4: File accessible: {file_size:.2f} MB")
            
            # Handle database connection based on available method
            if self.use_db_connection_module and self.db_connection_func:
                # Use db_connection context manager pattern
                with self.db_connection_func() as db:
                    print("🔌 Connected to PostgreSQL via db_connection module")
                    self.logger.info("STEP 4: Connected to PostgreSQL via db_connection module")
                    
                    # Step 1: Delete from stg_52wk_highlow_raw table
                    print("🗑️ Deleting records from stg_52wk_highlow_raw table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM stg_52wk_highlow_raw;"
                    db.execute(delete_query)
                    db.commit()  # EXPLICIT COMMIT
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted records from stg_52wk_highlow_raw in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 4: Deleted records from stg_52wk_highlow_raw in {delete_duration:.2f} seconds")
                    
                    # Step 2: Load CSV file into stg_52wk_highlow_raw table
                    print("📥 Loading CSV into stg_52wk_highlow_raw table...")
                    start_time = time.time()
                    
                    # Read CSV and insert row by row (db_connection doesn't support copy_expert)
                    with open(file_path, 'r') as f:
                        # Skip first 2 rows (disclaimer and effective date)
                        next(f)  # Skip first row
                        next(f)  # Skip second row
                        
                        # Third row is header - we can skip it since we're defining our own column structure
                        header_row = next(f)
                        print(f"📋 Header row: {header_row.strip()}")
                        
                        # Prepare insert query for individual rows
                        insert_row_query = """
                        INSERT INTO stg_52wk_highlow_raw (SYMBOL, SERIES, Adjusted_52_Week_High, "52_Week_High_Date", Adjusted_52_Week_Low, "52_Week_Low_DT")
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        
                        rows_inserted = 0
                        for line in f:
                            try:
                                # Parse CSV line using csv.reader for better handling
                                csv_reader = csv.reader(StringIO(line.strip()))
                                values = next(csv_reader)
                                
                                if len(values) >= 6:  # Ensure we have enough columns
                                    # Convert empty strings to None for NULL values
                                    cleaned_values = []
                                    for i, val in enumerate(values[:6]):
                                        cleaned_val = val.strip() if val.strip() != '' else None
                                        # Handle numeric conversions for high/low values
                                        if i in [2, 4] and cleaned_val:  # High and Low price columns
                                            try:
                                                cleaned_val = float(cleaned_val)
                                            except:
                                                cleaned_val = None
                                        cleaned_values.append(cleaned_val)
                                    
                                    db.execute(insert_row_query, cleaned_values)
                                    rows_inserted += 1
                                    
                                    # Print progress and commit in batches for performance
                                    if rows_inserted % 1000 == 0:
                                        print(f"   📊 Inserted {rows_inserted} rows...")
                                        db.commit()  # BATCH COMMIT
                                        
                            except Exception as row_error:
                                print(f"   ⚠️ Skipped row due to error: {row_error}")
                                print(f"   🔍 Problematic line: {line.strip()[:100]}...")
                                continue
                        
                        # Final commit for remaining rows
                        db.commit()  # EXPLICIT COMMIT
                        load_duration = time.time() - start_time
                        print(f"✅ Loaded {rows_inserted} rows into stg_52wk_highlow_raw table in {load_duration:.2f} seconds")
                        self.logger.info(f"STEP 4: Loaded {rows_inserted} rows into stg_52wk_highlow_raw table in {load_duration:.2f} seconds")
                    
                print("✅ 52 Week High Low data load completed successfully")
                self.logger.info("STEP 4: 52 Week High Low data load completed successfully")
                return True
            
            else:
                # Use traditional psycopg2 connection pattern
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                cursor = connection.cursor()
                print("🔌 Connected to PostgreSQL via direct connection")
                self.logger.info("STEP 4: Connected to PostgreSQL via direct connection")
                
                try:
                    # Step 1: Delete from stg_52wk_highlow_raw table
                    print("🗑️ Deleting records from stg_52wk_highlow_raw table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM stg_52wk_highlow_raw;"
                    cursor.execute(delete_query)
                    row_count = cursor.rowcount
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted {row_count} records in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 4: Deleted {row_count} records from stg_52wk_highlow_raw in {delete_duration:.2f} seconds")
                    
                    # Step 2: Load CSV file into stg_52wk_highlow_raw table
                    print("📥 Loading CSV into stg_52wk_highlow_raw table...")
                    start_time = time.time()
                    
                    # Read the file and process it row by row for better error handling
                    with open(file_path, 'r') as f:
                        # Skip first 2 rows (disclaimer and effective date)
                        next(f)  # Skip first row
                        next(f)  # Skip second row
                        next(f)  # Skip third row (header)
                        
                        rows_inserted = 0
                        for line in f:
                            try:
                                # Parse CSV line properly using csv.reader
                                csv_reader = csv.reader(StringIO(line.strip()))
                                values = next(csv_reader)
                                
                                if len(values) >= 6:
                                    # Clean values
                                    cleaned_values = []
                                    for i, val in enumerate(values[:6]):
                                        cleaned_val = val.strip() if val.strip() != '' else None
                                        # Handle numeric conversions
                                        if i in [2, 4] and cleaned_val:  # High and Low price columns
                                            try:
                                                cleaned_val = float(cleaned_val)
                                            except:
                                                cleaned_val = None
                                        cleaned_values.append(cleaned_val)
                                    
                                    insert_query = """
                                    INSERT INTO stg_52wk_highlow_raw (SYMBOL, SERIES, Adjusted_52_Week_High, "52_Week_High_Date", Adjusted_52_Week_Low, "52_Week_Low_DT")
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    """
                                    cursor.execute(insert_query, cleaned_values)
                                    rows_inserted += 1
                                    
                            except Exception as row_error:
                                print(f"   ⚠️ Skipped row due to error: {row_error}")
                                continue
                    
                    load_duration = time.time() - start_time
                    print(f"✅ Loaded {rows_inserted} rows into stg_52wk_highlow_raw table in {load_duration:.2f} seconds")
                    self.logger.info(f"STEP 4: Loaded {rows_inserted} rows into stg_52wk_highlow_raw table in {load_duration:.2f} seconds")
                    
                    # Commit transaction
                    connection.commit()
                    print("✅ 52 Week High Low data load completed successfully")
                    self.logger.info("STEP 4: 52 Week High Low data load completed successfully")
                    return True
                    
                except Exception as e:
                    connection.rollback()
                    print("🔄 Transaction rolled back")
                    raise e
                finally:
                    cursor.close()
                    connection.close()
                    print("🔌 PostgreSQL connection closed")
                    self.logger.info("STEP 4: PostgreSQL connection closed")
                
        except Exception as e:
            print(f"❌ Error loading 52 Week High Low data: {e}")
            self.logger.error(f"STEP 4 ERROR: Error loading 52 Week High Low data: {e}")
            return False

    def process_stock_data_operations(self):
        """Final stock_data delete/insert/update operations - WITH COMMITS"""
        if not self.parsed_date:
            print("❌ No parsed date available for stock_data operations")
            self.logger.error("STEP 5 ERROR: No parsed date available for stock_data operations")
            return False
        
        try:
            print(f"\n📈 PROCESSING STOCK_DATA OPERATIONS")
            print(f"📅 Using date: {self.parsed_date}")
            self.logger.info(f"STEP 5: Starting stock_data operations for date: {self.parsed_date}")
            
            # Handle database connection based on available method
            if self.use_db_connection_module and self.db_connection_func:
                # Use db_connection context manager pattern
                with self.db_connection_func() as db:
                    print("🔌 Connected to PostgreSQL via db_connection module")
                    self.logger.info("STEP 5: Connected to PostgreSQL via db_connection module")
                    
                    # Step 1: Delete from stock_data
                    print("🗑️ Deleting all records from stock_data table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM stock_data;"
                    db.execute(delete_query)
                    db.commit()  # EXPLICIT COMMIT
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted records from stock_data in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 5: Deleted records from stock_data in {delete_duration:.2f} seconds")
                    
                    # Step 2: Insert data into stock_data from stg_nse_data
                    print(f"📊 Inserting data into stock_data for date {self.parsed_date}...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stock_data 
                    (SELECT symbol, name, trd_dt, close_pr, prev_cl_pr, volume, NULL, hi_52_wk, lo_52_wk, NULL, NULL, NULL, NULL, NULL
                     FROM stg_nse_data
                     WHERE trd_dt = %s);
                    """
                    db.execute(insert_query, (self.parsed_date,))
                    db.commit()  # EXPLICIT COMMIT
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted rows into stock_data in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 5: Inserted rows into stock_data for date {self.parsed_date} in {insert_duration:.2f} seconds")
                    
                    # Step 3: Update stock_data with mcap data
                    print("💰 Updating stock_data with market cap data...")
                    start_time = time.time()
                    update_query = """
                    UPDATE stock_data s
                    SET mcap = (SELECT mcap/10000000 FROM stg_mcap_data m WHERE m.symbol = s.symbol);
                    """
                    db.execute(update_query)
                    db.commit()  # EXPLICIT COMMIT
                    update_duration = time.time() - start_time
                    print(f"✅ Updated rows with market cap data in {update_duration:.2f} seconds")
                    self.logger.info(f"STEP 5: Updated rows with market cap data in {update_duration:.2f} seconds")
                    
                print("✅ Stock data operations completed successfully")
                self.logger.info("STEP 5: Stock data operations completed successfully")
                return True
            
            else:
                # Use traditional psycopg2 connection pattern
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                cursor = connection.cursor()
                print("🔌 Connected to PostgreSQL via direct connection")
                self.logger.info("STEP 5: Connected to PostgreSQL via direct connection")
                
                try:
                    # Step 1: Delete from stock_data
                    print("🗑️ Deleting all records from stock_data table...")
                    start_time = time.time()
                    delete_query = "DELETE FROM stock_data;"
                    cursor.execute(delete_query)
                    row_count = cursor.rowcount
                    delete_duration = time.time() - start_time
                    print(f"✅ Deleted {row_count} records from stock_data in {delete_duration:.2f} seconds")
                    self.logger.info(f"STEP 5: Deleted {row_count} records from stock_data in {delete_duration:.2f} seconds")
                    
                    # Step 2: Insert data into stock_data from stg_nse_data
                    print(f"📊 Inserting data into stock_data for date {self.parsed_date}...")
                    start_time = time.time()
                    insert_query = """
                    INSERT INTO stock_data 
                    (SELECT symbol, name, trd_dt, close_pr, prev_cl_pr, volume, NULL, hi_52_wk, lo_52_wk, NULL, NULL, NULL, NULL, NULL
                     FROM stg_nse_data
                     WHERE trd_dt = %s);
                    """
                    cursor.execute(insert_query, (self.parsed_date,))
                    row_count = cursor.rowcount
                    insert_duration = time.time() - start_time
                    print(f"✅ Inserted {row_count} rows into stock_data in {insert_duration:.2f} seconds")
                    self.logger.info(f"STEP 5: Inserted {row_count} rows into stock_data for date {self.parsed_date} in {insert_duration:.2f} seconds")
                    
                    # Step 3: Update stock_data with mcap data
                    print("💰 Updating stock_data with market cap data...")
                    start_time = time.time()
                    update_query = """
                    UPDATE stock_data s
                    SET mcap = (SELECT mcap/10000000 FROM stg_mcap_data m WHERE m.symbol = s.symbol);
                    """
                    cursor.execute(update_query)
                    row_count = cursor.rowcount
                    update_duration = time.time() - start_time
                    print(f"✅ Updated {row_count} rows with market cap data in {update_duration:.2f} seconds")
                    self.logger.info(f"STEP 5: Updated {row_count} rows with market cap data in {update_duration:.2f} seconds")
                    
                    # Step 4: Commit all stock_data operations
                    connection.commit()
                    print("✅ Stock data operations completed successfully")
                    self.logger.info("STEP 5: Stock data operations completed successfully")
                    return True
                    
                except Exception as e:
                    connection.rollback()
                    print("🔄 Transaction rolled back")
                    raise e
                finally:
                    cursor.close()
                    connection.close()
                    print("🔌 PostgreSQL connection closed")
                    self.logger.info("STEP 5: PostgreSQL connection closed")
                
        except Exception as e:
            print(f"❌ Error in stock_data operations: {e}")
            self.logger.error(f"STEP 5 ERROR: Error in stock_data operations: {e}")
            return False
    
    def run_database_load_pipeline(self):
        """Run the database loading pipeline (Steps 2-5)"""
        print("=" * 90)
        print("                    NSE DATABASE LOAD PIPELINE")
        print("              Load PD → Load MCAP → Load 52WK → Process Stock Data")
        print("=" * 90)
        
        pipeline_start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("DATABASE LOAD PIPELINE START")
        self.logger.info("=" * 80)
        
        # Load extraction information from download script
        print("\n🔍 LOADING EXTRACTION INFO FROM DOWNLOAD SCRIPT")
        print("-" * 50)
        
        if not self.load_extraction_info():
            print("❌ Failed to load extraction info - cannot proceed")
            self.logger.error("Failed to load extraction info - cannot proceed")
            return False
        
        print("✅ Extraction info loaded successfully")
        
        # Define step names for batch tracking
        step_names = [
            "STEP 2 - LOAD PD DATA TO DATABASE", 
            "STEP 3 - LOAD MCAP DATA TO DATABASE",
            "STEP 4 - LOAD 52WK HIGH LOW DATA TO DATABASE",
            "STEP 5 - PROCESS STOCK_DATA OPERATIONS"
        ]
        
        # VALIDATION: Check if data is already processed
        print("\n🔍 CHECKING IF DATA ALREADY PROCESSED")
        print("-" * 50)
        
        already_processed, process_reason = self.check_if_already_processed()
        if already_processed:
            print(f"✅ PIPELINE SKIPPED: {process_reason}")
            self.logger.info(f"PIPELINE SKIPPED: {process_reason}")
            return True  # Return True since no processing was needed
        
        print(f"➡️ {process_reason} - Proceeding with database pipeline")
        
        # Step 2: Load PD data
        print("\n🚀 STEP 2: LOAD PD DATA TO DATABASE")
        print("-" * 50)
        self.update_batch_status(step_names[0], "Started")
        self.logger.info("STEP 2: Starting PD data loading process")
        
        try:
            if not self.load_pd_data_to_database():
                print("❌ PIPELINE FAILED: Could not load PD data")
                self.logger.error("PIPELINE FAILED: Could not load PD data")
                self.update_batch_status(step_names[0], "Failed")
                return False
            
            print("✅ STEP 2 COMPLETED: PD data loaded successfully")
            self.logger.info("STEP 2 COMPLETED: PD data loaded successfully")
            self.update_batch_status(step_names[0], "Completed")
            
        except Exception as e:
            print(f"❌ STEP 2 FAILED: {e}")
            self.logger.error(f"STEP 2 FAILED: {e}")
            self.update_batch_status(step_names[0], "Failed")
            return False
        
        # Step 3: Load MCAP data
        print("\n🚀 STEP 3: LOAD MCAP DATA TO DATABASE")
        print("-" * 50)
        self.update_batch_status(step_names[1], "Started")
        self.logger.info("STEP 3: Starting MCAP data loading process")
        
        try:
            if not self.load_mcap_data_to_database():
                print("❌ PIPELINE FAILED: Could not load MCAP data")
                self.logger.error("PIPELINE FAILED: Could not load MCAP data")
                self.update_batch_status(step_names[1], "Failed")
                return False
            
            print("✅ STEP 3 COMPLETED: MCAP data loaded successfully")
            self.logger.info("STEP 3 COMPLETED: MCAP data loaded successfully")
            self.update_batch_status(step_names[1], "Completed")
            
        except Exception as e:
            print(f"❌ STEP 3 FAILED: {e}")
            self.logger.error(f"STEP 3 FAILED: {e}")
            self.update_batch_status(step_names[1], "Failed")
            return False
        
        # Step 4: Load 52 Week High Low data
        print("\n🚀 STEP 4: LOAD 52WK HIGH LOW DATA TO DATABASE")
        print("-" * 50)
        self.update_batch_status(step_names[2], "Started")
        self.logger.info("STEP 4: Starting 52 Week High Low data loading process")
        
        try:
            if self.extracted_files["highlow_file"]:
                if not self.load_52wk_data_to_database():
                    print("❌ PIPELINE FAILED: Could not load 52 Week High Low data")
                    self.logger.error("PIPELINE FAILED: Could not load 52 Week High Low data")
                    self.update_batch_status(step_names[2], "Failed")
                    return False
                
                print("✅ STEP 4 COMPLETED: 52 Week High Low data loaded successfully")
                self.logger.info("STEP 4 COMPLETED: 52 Week High Low data loaded successfully")
            else:
                print("⚠️ STEP 4 SKIPPED: No 52 Week High Low file available")
                self.logger.warning("STEP 4 SKIPPED: No 52 Week High Low file available")
            
            self.update_batch_status(step_names[2], "Completed")
            
        except Exception as e:
            print(f"❌ STEP 4 FAILED: {e}")
            self.logger.error(f"STEP 4 FAILED: {e}")
            self.update_batch_status(step_names[2], "Failed")
            return False
        
        # Step 5: Process stock_data operations
        print("\n🚀 STEP 5: PROCESS STOCK_DATA OPERATIONS")
        print("-" * 50)
        self.update_batch_status(step_names[3], "Started")
        self.logger.info("STEP 5: Starting stock_data operations process")
        
        try:
            if not self.process_stock_data_operations():
                print("❌ PIPELINE FAILED: Could not process stock_data operations")
                self.logger.error("PIPELINE FAILED: Could not process stock_data operations")
                self.update_batch_status(step_names[3], "Failed")
                return False
            
            print("✅ STEP 5 COMPLETED: Stock data operations completed successfully")
            self.logger.info("STEP 5 COMPLETED: Stock data operations completed successfully")
            self.update_batch_status(step_names[3], "Completed")
            
        except Exception as e:
            print(f"❌ STEP 5 FAILED: {e}")
            self.logger.error(f"STEP 5 FAILED: {e}")
            self.update_batch_status(step_names[3], "Failed")
            return False
        
        # Pipeline completion
        pipeline_duration = time.time() - pipeline_start_time
        
        print("\n" + "=" * 90)
        print("🎉 DATABASE LOAD PIPELINE FINISHED SUCCESSFULLY! 🎉")
        print("=" * 90)
        print(f"⏱️ Total pipeline duration: {pipeline_duration:.2f} seconds")
        print(f"📊 Database: Updated with latest stock data")
        print(f"📅 Process date: {self.parsed_date}")
        print("\n📋 COMPLETED OPERATIONS:")
        print("   ✅ Loaded PD data into stg_nse_data & stg_nse_data_hist")
        print("   ✅ Loaded MCAP data into stg_mcap_data")
        print("   ✅ Loaded 52 Week High Low data into stg_52wk_highlow_raw")
        print("   ✅ Refreshed stock_data table with latest data")
        print("   ✅ Updated market cap values")
        print("   ✅ All batch process statuses updated")
        print("=" * 90)
        
        self.logger.info("=" * 80)
        self.logger.info(f"DATABASE LOAD PIPELINE COMPLETED SUCCESSFULLY in {pipeline_duration:.2f} seconds")
        self.logger.info(f"Process date: {self.parsed_date}")
        self.logger.info("All operations completed: Load PD → Load MCAP → Load 52WK → Process Stock Data")
        self.logger.info("All batch process statuses updated in stg_batch_process table")
        self.logger.info("=" * 80)
        
        return True

def main():
    print("📋 NSE Database Load Script")
    print("=" * 50)
    print("Requirements:")
    print("   • pip install pandas psycopg2")
    print("   • PostgreSQL database with required tables")
    print("   • extraction_info.json from download script")
    print("   • Database connection configured via db_config.py and db_connection.py")
    print("\n📄 Log File Generated:")
    print("   • nse_database_load.log")
    print("\n⚠️ Prerequisites:")
    print("   • Run nse_download_extract.py first to download files")
    print("")
    
    # Initialize database load pipeline
    pipeline = NSEDatabaseLoad(
        download_path=r"C:\Python Code\Files"
    )
    
    try:
        # Run database load pipeline (Steps 2-5)
        success = pipeline.run_database_load_pipeline()
        
        if success:
            print("\n✅ Database loading completed successfully!")
            print("📊 Check nse_database_load.log for detailed operation records")
            print("📊 Check stg_batch_process table for step-by-step status tracking")
            print("📈 Stock data is now updated in the database")
        else:
            print("\n❌ Database loading pipeline failed.")
            print("📊 Check nse_database_load.log for details")
            print("📊 Check stg_batch_process table to see which step failed")
            
    except Exception as e:
        print(f"\n💥 Unexpected pipeline error: {e}")
        pipeline.logger.error(f"UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()