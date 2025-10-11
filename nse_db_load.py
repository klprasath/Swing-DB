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
        self.extracted_files = {"pd_file": None, "mcap_file": None, "bc_file": None, "highlow_file": None}
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
            
            # If BC file is not in extraction_info, try to find it manually
            if not self.extracted_files.get("bc_file"):
                self.find_bc_file()
            
            print(f"📋 Loaded extraction info:")
            print(f"   • PD file: {os.path.basename(self.extracted_files.get('pd_file', 'None'))}")
            print(f"   • MCAP file: {os.path.basename(self.extracted_files.get('mcap_file', 'None'))}")
            print(f"   • BC file: {os.path.basename(self.extracted_files.get('bc_file', 'None'))}")
            print(f"   • 52WK file: {os.path.basename(self.extracted_files.get('highlow_file', 'None'))}")
            print(f"   • Parsed date: {self.parsed_date}")
            
            self.logger.info(f"Loaded extraction info - PD: {self.extracted_files.get('pd_file')}")
            self.logger.info(f"Loaded extraction info - MCAP: {self.extracted_files.get('mcap_file')}")
            self.logger.info(f"Loaded extraction info - BC: {self.extracted_files.get('bc_file')}")
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

    def find_bc_file(self):
        """Find BC file in the download directory"""
        try:
            # Look for BC files matching pattern Bc<ddmmyy>.csv
            for filename in os.listdir(self.download_path):
                if filename.startswith("Bc") and filename.endswith(".csv"):
                    full_path = os.path.join(self.download_path, filename)
                    if os.path.isfile(full_path):
                        self.extracted_files["bc_file"] = full_path
                        self.logger.info(f"Found BC file: {filename}")
                        break
        except Exception as e:
            self.logger.warning(f"Error finding BC file: {e}")

class DatabaseConnectionWrapper:
    """Wrapper class to provide unified interface for both db_connection module and direct psycopg2"""
    
    def __init__(self, connection_obj, is_db_connection_module=False):
        self.connection_obj = connection_obj
        self.is_db_connection_module = is_db_connection_module
        if not is_db_connection_module:
            self.cursor, self.connection = connection_obj
    
    def execute(self, query, params=None):
        """Execute a query with optional parameters"""
        if self.is_db_connection_module:
            return self.connection_obj.execute(query, params) if params else self.connection_obj.execute(query)
        else:
            return self.cursor.execute(query, params) if params else self.cursor.execute(query)
    
    def fetchone(self):
        """Fetch one row from the result"""
        if self.is_db_connection_module:
            return self.connection_obj.fetchone()
        else:
            return self.cursor.fetchone()
    
    def fetchall(self):
        """Fetch all rows from the result"""
        if self.is_db_connection_module:
            return self.connection_obj.fetchall()
        else:
            return self.cursor.fetchall()
    
    def commit(self):
        """Commit the current transaction"""
        if self.is_db_connection_module:
            return self.connection_obj.commit()
        else:
            return self.connection.commit()
    
    def rollback(self):
        """Rollback the current transaction"""
        if self.is_db_connection_module:
            return self.connection_obj.rollback()
        else:
            return self.connection.rollback()
    
    def copy_expert(self, sql, file):
        """Copy expert method (only available for direct psycopg2)"""
        if not self.is_db_connection_module:
            return self.cursor.copy_expert(sql, file)
        else:
            raise NotImplementedError("copy_expert not available in db_connection module")
    
    @property
    def rowcount(self):
        """Get the number of rows affected by the last statement"""
        if self.is_db_connection_module:
            # db_connection module may not have rowcount, return None
            return getattr(self.connection_obj, 'rowcount', None)
        else:
            return self.cursor.rowcount

# Continue with the main NSEDatabaseLoad class methods
class NSEDatabaseLoad(NSEDatabaseLoad):
    
    def update_batch_status(self, step_name, status, db_wrapper):
        """Update single row in batch process tracking table"""
        try:
            # Use parsed_date from file if available, otherwise use current date as fallback
            tracking_date = self.parsed_date if self.parsed_date else datetime.now().date()
            
            # First, get the existing trd_dt and prev_trd_dt before deleting
            existing_trd_dt = None
            existing_prev_trd_dt = None
            try:
                select_query = "SELECT trd_dt, prev_trd_dt FROM stg_batch_process LIMIT 1"
                db_wrapper.execute(select_query)
                result = db_wrapper.fetchone()
                if result:
                    existing_trd_dt = result[0]
                    existing_prev_trd_dt = result[1]
            except:
                existing_trd_dt = None
                existing_prev_trd_dt = None
            
            # Determine prev_trd_dt value
            if existing_trd_dt is None:
                prev_trd_dt = None
            elif existing_trd_dt == tracking_date:
                prev_trd_dt = existing_prev_trd_dt
            else:
                prev_trd_dt = existing_trd_dt
            
            # Always update/insert the single row - delete existing and insert new
            delete_query = "DELETE FROM stg_batch_process"
            db_wrapper.execute(delete_query)
            
            # Insert new record with prev_trd_dt
            insert_query = """
            INSERT INTO stg_batch_process (step, status, trd_dt, updated, prev_trd_dt)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s)
            """
            db_wrapper.execute(insert_query, (step_name, status, tracking_date, prev_trd_dt))
            db_wrapper.commit()
            
            self.logger.info(f"BATCH_TRACKING: Updated single row - {step_name} - {status} for date {tracking_date}")
                    
        except Exception as e:
            self.logger.error(f"BATCH_TRACKING ERROR: Failed to update {step_name} - {status}: {e}")
            # Don't fail the entire pipeline due to batch tracking errors

    def check_data_already_in_staging(self, db_wrapper):
        """Check if the file date data already exists in stg_nse_data table"""
        if not self.parsed_date:
            return False, "No parsed date available for validation"
            
        try:
            check_query = "SELECT COUNT(*) FROM stg_nse_data WHERE trd_dt = %s"
            db_wrapper.execute(check_query, (self.parsed_date,))
            result = db_wrapper.fetchone()[0] > 0
            
            if result:
                self.logger.info(f"DATA_VALIDATION: Data for {self.parsed_date} already exists in stg_nse_data")
                return True, f"Data for {self.parsed_date} already available in staging tables"
            else:
                self.logger.info(f"DATA_VALIDATION: Data for {self.parsed_date} not found in stg_nse_data - proceeding with load")
                return False, f"Data for {self.parsed_date} not found in staging tables"
                        
        except Exception as e:
            error_msg = f"Error checking if data already in staging: {e}"
            self.logger.error(f"DATA_VALIDATION ERROR: {error_msg}")
            # In case of error, assume not in staging to avoid blocking
            return False, "Validation error - assuming data not in staging"

    def check_if_already_processed(self, db_wrapper):
        """Check if today's data is already completed by checking STAGE 6 status"""
        try:
            current_date = datetime.now().date()
            
            check_query = """
            SELECT COUNT(*) FROM stg_batch_process 
            WHERE step = 'STAGE 6 - PROCESS STOCK_DATA OPERATIONS' 
            AND status = 'Completed' 
            AND trd_dt = %s
            """
            db_wrapper.execute(check_query, (current_date,))
            result = db_wrapper.fetchone()[0] > 0
            
            if result:
                self.logger.info(f"VALIDATION: Data for {current_date} already processed completely")
                return True, f"Data for {current_date} already processed completely"
            else:
                return False, f"Data for {current_date} not yet processed"
                        
        except Exception as e:
            error_msg = f"Error checking if data already processed: {e}"
            self.logger.error(f"VALIDATION ERROR: {error_msg}")
            # In case of error, assume not processed to avoid blocking
            return False, "Validation error - assuming not processed"

    def load_pd_data_to_database(self, db_wrapper):
        """Load Pd data to database - simplified with single connection interface"""
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
            
            # Step 1: Delete from nse_raw table
            print("🗑️ Deleting records from nse_raw table...")
            start_time = time.time()
            delete_query = "DELETE FROM nse_raw;"
            db_wrapper.execute(delete_query)
            delete_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "all"
            print(f"✅ Deleted {row_count} records from nse_raw in {delete_duration:.2f} seconds")
            self.logger.info(f"STEP 2: Deleted {row_count} records from nse_raw in {delete_duration:.2f} seconds")
            
            # Step 2: Load CSV file into nse_raw table
            print("📥 Loading CSV into nse_raw table...")
            start_time = time.time()
            
            # Use different loading methods based on connection type
            if db_wrapper.is_db_connection_module:
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
                                db_wrapper.execute(insert_row_query, cleaned_values)
                                rows_inserted += 1
                                
                                # Commit in batches for performance
                                if rows_inserted % 1000 == 0:
                                    print(f"   📊 Inserted {rows_inserted} rows...")
                                    db_wrapper.commit()
                        except Exception as row_error:
                            print(f"   ⚠️ Skipped row due to error: {row_error}")
                            continue
                    
                    # Final commit for remaining rows
                    db_wrapper.commit()
                    print(f"✅ Loaded {rows_inserted} rows into nse_raw table in {time.time() - start_time:.2f} seconds")
                    self.logger.info(f"STEP 2: Loaded {rows_inserted} rows into nse_raw table in {time.time() - start_time:.2f} seconds")
            else:
                # Use copy_expert for direct psycopg2 connection
                with open(file_path, 'r') as f:
                    # Skip the header row
                    next(f)
                    # Create a StringIO buffer to hold CSV data
                    buffer = StringIO()
                    for line in f:
                        buffer.write(line)
                    buffer.seek(0)
                    
                    copy_query = """
                    COPY nse_raw FROM STDIN
                    DELIMITER ',' 
                    CSV
                    NULL ' ';
                    """
                    db_wrapper.copy_expert(copy_query, buffer)
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
            db_wrapper.execute(insert_query, (parsed_date,))
            db_wrapper.commit()
            insert_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "multiple"
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
            db_wrapper.execute(insert_query, (parsed_date,))
            db_wrapper.commit()
            insert_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "multiple"
            print(f"✅ Inserted {row_count} rows into stg_nse_data_hist in {insert_duration:.2f} seconds")
            self.logger.info(f"STEP 2: Inserted {row_count} rows into stg_nse_data_hist in {insert_duration:.2f} seconds")
            
            print("✅ Pd data load completed successfully")
            self.logger.info("STEP 2: Pd data load completed successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error loading Pd data: {e}")
            self.logger.error(f"STEP 2 ERROR: Error loading Pd data: {e}")
            return False

    def load_mcap_data_to_database(self, db_wrapper):
        """Load MCAP data to database - simplified with single connection interface"""
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
            
            # Step 1: Delete from mcap_raw table
            print("🗑️ Deleting records from mcap_raw table...")
            start_time = time.time()
            delete_query = "DELETE FROM mcap_raw;"
            db_wrapper.execute(delete_query)
            db_wrapper.commit()
            delete_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "all"
            print(f"✅ Deleted {row_count} records from mcap_raw in {delete_duration:.2f} seconds")
            self.logger.info(f"STEP 3: Deleted {row_count} records from mcap_raw in {delete_duration:.2f} seconds")
            
            # Step 2: Load CSV file into mcap_raw table
            print("📥 Loading CSV into mcap_raw table...")
            start_time = time.time()
            
            # Use different loading methods based on connection type
            if db_wrapper.is_db_connection_module:
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
                                db_wrapper.execute(insert_row_query, cleaned_values)
                                rows_inserted += 1
                                
                                # Print progress and commit in batches for performance
                                if rows_inserted % 1000 == 0:
                                    print(f"   📊 Inserted {rows_inserted} rows...")
                                    db_wrapper.commit()
                        except Exception as row_error:
                            print(f"   ⚠️ Skipped row due to error: {row_error}")
                            continue
                    
                    # Final commit for remaining rows
                    db_wrapper.commit()
                    load_duration = time.time() - start_time
                    print(f"✅ Loaded {rows_inserted} rows into mcap_raw table in {load_duration:.2f} seconds")
                    self.logger.info(f"STEP 3: Loaded {rows_inserted} rows into mcap_raw table in {load_duration:.2f} seconds")
            else:
                # Use copy_expert for direct psycopg2 connection
                with open(file_path, 'r') as f:
                    # Skip the header row
                    next(f)
                    # Create a StringIO buffer to hold CSV data
                    buffer = StringIO()
                    for line in f:
                        buffer.write(line)
                    buffer.seek(0)
                    
                    copy_query = """
                    COPY mcap_raw
                    FROM STDIN
                    DELIMITER ',' 
                    CSV
                    NULL ' ';
                    """
                    db_wrapper.copy_expert(copy_query, buffer)
                    db_wrapper.commit()
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
            db_wrapper.execute(delete_query)
            db_wrapper.commit()
            delete_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "all"
            print(f"✅ Deleted {row_count} records from stg_mcap_data in {delete_duration:.2f} seconds")
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
            db_wrapper.execute(insert_query)
            db_wrapper.commit()
            insert_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "multiple"
            print(f"✅ Inserted {row_count} rows into stg_mcap_data in {insert_duration:.2f} seconds")
            self.logger.info(f"STEP 3: Inserted {row_count} rows into stg_mcap_data in {insert_duration:.2f} seconds")
            
            print("✅ MCAP data load completed successfully")
            self.logger.info("STEP 3: MCAP data load completed successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error loading MCAP data: {e}")
            self.logger.error(f"STEP 3 ERROR: Error loading MCAP data: {e}")
            return False

    def load_bc_data_to_database(self, db_wrapper):
        """Load BC data to database - simplified with single connection interface"""
        if not self.extracted_files["bc_file"]:
            print("❌ No BC file found to load")
            self.logger.error("STEP 4 ERROR: No BC file found to load")
            return False
        
        file_path = self.extracted_files["bc_file"]
        
        try:
            print(f"\n🢢 LOADING BC DATA TO DATABASE")
            print(f"📄 File: {os.path.basename(file_path)}")
            self.logger.info(f"STEP 4: Starting BC data load from: {os.path.basename(file_path)}")
            
            # Verify file accessibility
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                self.logger.error(f"STEP 4 ERROR: File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File accessible: {file_size:.2f} MB")
            self.logger.info(f"STEP 4: File accessible: {file_size:.2f} MB")
            
            def parse_date(date_str):
                """Parse date string in DD/MM/YYYY format, return None if empty or invalid"""
                if not date_str or date_str.strip() == '':
                    return None
                try:
                    return datetime.strptime(date_str.strip(), '%d/%m/%Y').date()
                except:
                    return None
            
            # Step 1: Delete from corp_raw table
            print("🗑️ Deleting records from corp_raw table...")
            start_time = time.time()
            delete_query = "DELETE FROM corp_raw;"
            db_wrapper.execute(delete_query)
            db_wrapper.commit()
            delete_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "all"
            print(f"✅ Deleted {row_count} records from corp_raw in {delete_duration:.2f} seconds")
            self.logger.info(f"STEP 4: Deleted {row_count} records from corp_raw in {delete_duration:.2f} seconds")
            
            # Step 2: Load CSV file into corp_raw table
            print("📥 Loading CSV into corp_raw table...")
            start_time = time.time()
            
            # Read CSV and insert row by row for better error handling
            with open(file_path, 'r') as f:
                # Skip the header row
                next(f)
                
                # Prepare insert query for individual rows
                insert_row_query = """
                INSERT INTO corp_raw (series, symbol, security, record_dt, bc_strt_dt, bc_end_dt, ex_dt, nd_strt_dt, nd_end_dt, purpose)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                rows_inserted = 0
                for line in f:
                    try:
                        # Parse CSV line using csv.reader for better handling
                        csv_reader = csv.reader(StringIO(line.strip()), delimiter=',')
                        values = next(csv_reader)
                        
                        if len(values) >= 10:  # Ensure we have enough columns
                            # Clean and convert values
                            series = values[0].strip() if values[0].strip() else None
                            symbol = values[1].strip() if values[1].strip() else None
                            security = values[2].strip() if values[2].strip() else None
                            record_dt = parse_date(values[3])
                            bc_strt_dt = parse_date(values[4])
                            bc_end_dt = parse_date(values[5])
                            ex_dt = parse_date(values[6])
                            nd_strt_dt = parse_date(values[7])
                            nd_end_dt = parse_date(values[8])
                            purpose = values[9].strip() if values[9].strip() else None
                            
                            cleaned_values = [series, symbol, security, record_dt, bc_strt_dt, bc_end_dt, ex_dt, nd_strt_dt, nd_end_dt, purpose]
                            
                            db_wrapper.execute(insert_row_query, cleaned_values)
                            rows_inserted += 1
                            
                            # Print progress and commit in batches for performance
                            if rows_inserted % 1000 == 0:
                                print(f"   📊 Inserted {rows_inserted} rows...")
                                db_wrapper.commit()
                                
                    except Exception as row_error:
                        print(f"   ⚠️ Skipped row due to error: {row_error}")
                        print(f"   🔍 Problematic line: {line.strip()[:100]}...")
                        continue
                
                # Final commit for remaining rows
                db_wrapper.commit()
                load_duration = time.time() - start_time
                print(f"✅ Loaded {rows_inserted} rows into corp_raw table in {load_duration:.2f} seconds")
                self.logger.info(f"STEP 4: Loaded {rows_inserted} rows into corp_raw table in {load_duration:.2f} seconds")
            
            # Step 3: Extract date from filename (e.g., Bc220925.csv -> 220925)
            filename = os.path.basename(file_path)
            date_str = filename[2:8]  # Extract '220925' from 'Bc220925.csv'
            parsed_date = datetime.strptime(date_str, '%d%m%y').strftime('%Y-%m-%d')
            print(f"📅 Parsed date: {parsed_date}")
            self.logger.info(f"STEP 4: Parsed date: {parsed_date}")
            
            # Step 4: Insert SPLIT data into stg_corp_ancmnts
            print("📊 Inserting SPLIT data into stg_corp_ancmnts...")
            start_time = time.time()
            split_query = """
            INSERT INTO stg_corp_ancmnts
            SELECT DISTINCT symbol,TO_DATE(ex_dt,'DD/MM/YYYY'),'SPLIT',NULL,
             CAST((regexp_matches(purpose, 'FRM[^0-9]+([0-9]+) TO[^0-9]+([0-9]+)'))[1] AS INTEGER),
              CAST((regexp_matches(purpose, 'FRM[^0-9]+([0-9]+) TO[^0-9]+([0-9]+)'))[2] AS INTEGER)
            FROM corp_raw r
            WHERE purpose LIKE '%SPLT%'
            AND NOT EXISTS(SELECT 1 FROM stg_corp_ancmnts c WHERE c.symbol = r.symbol AND c.ex_dt = TO_DATE(r.ex_dt,'DD/MM/YYYY'))
            """
            db_wrapper.execute(split_query)
            db_wrapper.commit()
            insert_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "multiple"
            print(f"✅ Inserted {row_count} SPLIT rows into stg_corp_ancmnts in {insert_duration:.2f} seconds")
            self.logger.info(f"STEP 4: Inserted {row_count} SPLIT rows into stg_corp_ancmnts in {insert_duration:.2f} seconds")
            
            # Step 5: Insert BONUS data into stg_corp_ancmnts
            print("📊 Inserting BONUS data into stg_corp_ancmnts...")
            start_time = time.time()
            bonus_query = """
            INSERT INTO stg_corp_ancmnts
            SELECT DISTINCT symbol,TO_DATE(ex_dt,'DD/MM/YYYY'),'BONUS',
            SUBSTRING(TRIM(purpose) FROM 'BONUS\\s+(.*)')
            FROM corp_raw r
            WHERE purpose LIKE '%BONUS%'
            AND NOT EXISTS(SELECT 1 FROM stg_corp_ancmnts c WHERE c.symbol = r.symbol AND c.ex_dt = TO_DATE(r.ex_dt,'DD/MM/YYYY'))
            """
            db_wrapper.execute(bonus_query)
            db_wrapper.commit()
            insert_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "multiple"
            print(f"✅ Inserted {row_count} BONUS rows into stg_corp_ancmnts in {insert_duration:.2f} seconds")
            self.logger.info(f"STEP 4: Inserted {row_count} BONUS rows into stg_corp_ancmnts in {insert_duration:.2f} seconds")
            
            print("✅ BC data load completed successfully")
            self.logger.info("STEP 4: BC data load completed successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error loading BC data: {e}")
            self.logger.error(f"STEP 4 ERROR: Error loading BC data: {e}")
            return False

    def load_52wk_data_to_database(self, db_wrapper):
        """Load 52 Week High Low data to database - simplified with single connection interface"""
        if not self.extracted_files["highlow_file"]:
            print("❌ No 52 Week High Low file found to load")
            self.logger.error("STEP 5 ERROR: No 52 Week High Low file found to load")
            return False
        
        file_path = self.extracted_files["highlow_file"]
        
        try:
            print(f"\n📈 LOADING 52 WEEK HIGH LOW DATA TO DATABASE")
            print(f"📄 File: {os.path.basename(file_path)}")
            self.logger.info(f"STEP 5: Starting 52 Week High Low data load from: {os.path.basename(file_path)}")
            
            # Verify file accessibility
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                self.logger.error(f"STEP 5 ERROR: File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File accessible: {file_size:.2f} MB")
            self.logger.info(f"STEP 5: File accessible: {file_size:.2f} MB")
            
            # Step 1: Delete from stg_52wk_highlow_raw table
            print("🗑️ Deleting records from stg_52wk_highlow_raw table...")
            start_time = time.time()
            delete_query = "DELETE FROM stg_52wk_highlow_raw;"
            db_wrapper.execute(delete_query)
            db_wrapper.commit()
            delete_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "all"
            print(f"✅ Deleted {row_count} records from stg_52wk_highlow_raw in {delete_duration:.2f} seconds")
            self.logger.info(f"STEP 5: Deleted {row_count} records from stg_52wk_highlow_raw in {delete_duration:.2f} seconds")
            
            # Step 2: Load CSV file into stg_52wk_highlow_raw table
            print("📥 Loading CSV into stg_52wk_highlow_raw table...")
            start_time = time.time()
            
            # Read CSV and insert row by row for better error handling
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
                            
                            db_wrapper.execute(insert_row_query, cleaned_values)
                            rows_inserted += 1
                            
                            # Print progress and commit in batches for performance
                            if rows_inserted % 1000 == 0:
                                print(f"   📊 Inserted {rows_inserted} rows...")
                                db_wrapper.commit()
                                
                    except Exception as row_error:
                        print(f"   ⚠️ Skipped row due to error: {row_error}")
                        print(f"   🔍 Problematic line: {line.strip()[:100]}...")
                        continue
                
                # Final commit for remaining rows
                db_wrapper.commit()
                load_duration = time.time() - start_time
                print(f"✅ Loaded {rows_inserted} rows into stg_52wk_highlow_raw table in {load_duration:.2f} seconds")
                self.logger.info(f"STEP 5: Loaded {rows_inserted} rows into stg_52wk_highlow_raw table in {load_duration:.2f} seconds")
            
            print("✅ 52 Week High Low data load completed successfully")
            self.logger.info("STEP 5: 52 Week High Low data load completed successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error loading 52 Week High Low data: {e}")
            self.logger.error(f"STEP 5 ERROR: Error loading 52 Week High Low data: {e}")
            return False

    def process_stock_data_operations(self, db_wrapper):
        """Final stock_data delete/insert/update operations - simplified with single connection interface"""
        if not self.parsed_date:
            print("❌ No parsed date available for stock_data operations")
            self.logger.error("STEP 6 ERROR: No parsed date available for stock_data operations")
            return False
        
        try:
            print(f"\n📈 PROCESSING STOCK_DATA OPERATIONS")
            print(f"📅 Using date: {self.parsed_date}")
            self.logger.info(f"STEP 6: Starting stock_data operations for date: {self.parsed_date}")
            
            # Step 1: Delete from stock_data
            print("🗑️ Deleting all records from stock_data table...")
            start_time = time.time()
            delete_query = "DELETE FROM stock_data;"
            db_wrapper.execute(delete_query)
            db_wrapper.commit()
            delete_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "all"
            print(f"✅ Deleted {row_count} records from stock_data in {delete_duration:.2f} seconds")
            self.logger.info(f"STEP 6: Deleted {row_count} records from stock_data in {delete_duration:.2f} seconds")
            
            # Step 2: Insert data into stock_data from stg_nse_data
            print(f"📊 Inserting data into stock_data for date {self.parsed_date}...")
            start_time = time.time()
            insert_query = """
            INSERT INTO stock_data 
            (SELECT symbol, name, trd_dt, close_pr, prev_cl_pr, volume, NULL, hi_52_wk, lo_52_wk, NULL, NULL, NULL, NULL, NULL
             FROM stg_nse_data
             WHERE trd_dt = %s);
            """
            db_wrapper.execute(insert_query, (self.parsed_date,))
            db_wrapper.commit()
            insert_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "multiple"
            print(f"✅ Inserted {row_count} rows into stock_data in {insert_duration:.2f} seconds")
            self.logger.info(f"STEP 6: Inserted {row_count} rows into stock_data for date {self.parsed_date} in {insert_duration:.2f} seconds")
            
            # Step 3: Update stock_data with mcap data
            print("💰 Updating stock_data with market cap data...")
            start_time = time.time()
            update_query = """
            UPDATE stock_data s
            SET mcap = (SELECT mcap/10000000 FROM stg_mcap_data m WHERE m.symbol = s.symbol);
            """
            db_wrapper.execute(update_query)
            db_wrapper.commit()
            update_duration = time.time() - start_time
            row_count = db_wrapper.rowcount if db_wrapper.rowcount else "multiple"
            print(f"✅ Updated {row_count} rows with market cap data in {update_duration:.2f} seconds")
            self.logger.info(f"STEP 6: Updated {row_count} rows with market cap data in {update_duration:.2f} seconds")
            
            print("✅ Stock data operations completed successfully")
            self.logger.info("STEP 6: Stock data operations completed successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error in stock_data operations: {e}")
            self.logger.error(f"STEP 6 ERROR: Error in stock_data operations: {e}")
            return False

    def run_database_load_pipeline(self):
        """Run the database loading pipeline (Stages 2-6) with single database connection"""
        print("=" * 90)
        print("                    NSE DATABASE LOAD PIPELINE")
        print("         Load PD → Load MCAP → Load BC → Load 52WK → Process Stock Data")
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
        
        # Define stage names for batch tracking (changed from STEP to STAGE)
        stage_names = [
            "STAGE 2 - LOAD PD DATA TO DATABASE", 
            "STAGE 3 - LOAD MCAP DATA TO DATABASE",
            "STAGE 4 - LOAD BC DATA TO DATABASE",
            "STAGE 5 - LOAD 52WK HIGH LOW DATA TO DATABASE",
            "STAGE 6 - PROCESS STOCK_DATA OPERATIONS"
        ]
        
        # Open database connection once for the entire pipeline
        print("\n🔌 ESTABLISHING DATABASE CONNECTION FOR PIPELINE")
        print("-" * 50)
        
        try:
            if self.use_db_connection_module and self.db_connection_func:
                # Use db_connection context manager pattern
                print("✅ Opening connection via db_connection module")
                self.logger.info("PIPELINE: Opening connection via db_connection module")
                
                with self.db_connection_func() as db:
                    print("🔌 Connected to PostgreSQL via db_connection module")
                    self.logger.info("PIPELINE: Connected to PostgreSQL via db_connection module")
                    
                    # Create wrapper for db_connection module
                    db_wrapper = DatabaseConnectionWrapper(db, is_db_connection_module=True)
                    
                    # VALIDATION: Check if data is already processed (using the single connection)
                    print("\n🔍 CHECKING IF DATA ALREADY PROCESSED")
                    print("-" * 50)
                    
                    already_processed, process_reason = self.check_if_already_processed(db_wrapper)
                    if already_processed:
                        print(f"✅ PIPELINE SKIPPED: {process_reason}")
                        self.logger.info(f"PIPELINE SKIPPED: {process_reason}")
                        return True  # Return True since no processing was needed
                    
                    print(f"➡️ {process_reason} - Proceeding with database pipeline")
                    
                    # Execute all stages with the single connection
                    success = self._execute_pipeline_stages(db_wrapper, stage_names, pipeline_start_time)
                    return success
            
            else:
                # Use traditional psycopg2 connection pattern
                print("✅ Opening connection via direct PostgreSQL connection")
                self.logger.info("PIPELINE: Opening connection via direct PostgreSQL connection")
                
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                cursor = connection.cursor()
                
                try:
                    print("🔌 Connected to PostgreSQL via direct connection")
                    self.logger.info("PIPELINE: Connected to PostgreSQL via direct connection")
                    
                    # Create wrapper for direct psycopg2 connection
                    db_wrapper = DatabaseConnectionWrapper((cursor, connection), is_db_connection_module=False)
                    
                    # VALIDATION: Check if data is already processed (using the single connection)
                    print("\n🔍 CHECKING IF DATA ALREADY PROCESSED")
                    print("-" * 50)
                    
                    already_processed, process_reason = self.check_if_already_processed(db_wrapper)
                    if already_processed:
                        print(f"✅ PIPELINE SKIPPED: {process_reason}")
                        self.logger.info(f"PIPELINE SKIPPED: {process_reason}")
                        return True  # Return True since no processing was needed
                    
                    print(f"➡️ {process_reason} - Proceeding with database pipeline")
                    
                    # Execute all stages with the single connection
                    success = self._execute_pipeline_stages(db_wrapper, stage_names, pipeline_start_time)
                    return success
                    
                finally:
                    cursor.close()
                    connection.close()
                    print("🔌 PostgreSQL connection closed")
                    self.logger.info("PIPELINE: PostgreSQL connection closed")
                    
        except Exception as e:
            print(f"❌ Failed to establish database connection: {e}")
            self.logger.error(f"PIPELINE ERROR: Failed to establish database connection: {e}")
            return False

    def _execute_pipeline_stages(self, db_wrapper, stage_names, pipeline_start_time):
        """Execute all pipeline stages with the provided database connection wrapper"""
        
        # Stage 2: Load PD data
        print("\n🚀 STAGE 2: LOAD PD DATA TO DATABASE")
        print("-" * 50)
        self.update_batch_status(stage_names[0], "Started", db_wrapper)
        self.logger.info("STAGE 2: Starting PD data loading process")
        
        try:
            if not self.load_pd_data_to_database(db_wrapper):
                print("❌ PIPELINE FAILED: Could not load PD data")
                self.logger.error("PIPELINE FAILED: Could not load PD data")
                self.update_batch_status(stage_names[0], "Failed", db_wrapper)
                return False
            
            print("✅ STAGE 2 COMPLETED: PD data loaded successfully")
            self.logger.info("STAGE 2 COMPLETED: PD data loaded successfully")
            self.update_batch_status(stage_names[0], "Completed", db_wrapper)
            
        except Exception as e:
            print(f"❌ STAGE 2 FAILED: {e}")
            self.logger.error(f"STAGE 2 FAILED: {e}")
            self.update_batch_status(stage_names[0], "Failed", db_wrapper)
            return False
        
        # Stage 3: Load MCAP data
        print("\n🚀 STAGE 3: LOAD MCAP DATA TO DATABASE")
        print("-" * 50)
        self.update_batch_status(stage_names[1], "Started", db_wrapper)
        self.logger.info("STAGE 3: Starting MCAP data loading process")
        
        try:
            if not self.load_mcap_data_to_database(db_wrapper):
                print("❌ PIPELINE FAILED: Could not load MCAP data")
                self.logger.error("PIPELINE FAILED: Could not load MCAP data")
                self.update_batch_status(stage_names[1], "Failed", db_wrapper)
                return False
            
            print("✅ STAGE 3 COMPLETED: MCAP data loaded successfully")
            self.logger.info("STAGE 3 COMPLETED: MCAP data loaded successfully")
            self.update_batch_status(stage_names[1], "Completed", db_wrapper)
            
        except Exception as e:
            print(f"❌ STAGE 3 FAILED: {e}")
            self.logger.error(f"STAGE 3 FAILED: {e}")
            self.update_batch_status(stage_names[1], "Failed", db_wrapper)
            return False
        
        # Stage 4: Load BC data
        print("\n🚀 STAGE 4: LOAD BC DATA TO DATABASE")
        print("-" * 50)
        self.update_batch_status(stage_names[2], "Started", db_wrapper)
        self.logger.info("STAGE 4: Starting BC data loading process")
        
        try:
            if self.extracted_files["bc_file"]:
                if not self.load_bc_data_to_database(db_wrapper):
                    print("❌ PIPELINE FAILED: Could not load BC data")
                    self.logger.error("PIPELINE FAILED: Could not load BC data")
                    self.update_batch_status(stage_names[2], "Failed", db_wrapper)
                    return False
                
                print("✅ STAGE 4 COMPLETED: BC data loaded successfully")
                self.logger.info("STAGE 4 COMPLETED: BC data loaded successfully")
            else:
                print("⚠️ STAGE 4 SKIPPED: No BC file available")
                self.logger.warning("STAGE 4 SKIPPED: No BC file available")
            
            self.update_batch_status(stage_names[2], "Completed", db_wrapper)
            
        except Exception as e:
            print(f"❌ STAGE 4 FAILED: {e}")
            self.logger.error(f"STAGE 4 FAILED: {e}")
            self.update_batch_status(stage_names[2], "Failed", db_wrapper)
            return False
        
        # Stage 5: Load 52 Week High Low data
        print("\n🚀 STAGE 5: LOAD 52WK HIGH LOW DATA TO DATABASE")
        print("-" * 50)
        self.update_batch_status(stage_names[3], "Started", db_wrapper)
        self.logger.info("STAGE 5: Starting 52 Week High Low data loading process")
        
        try:
            if self.extracted_files["highlow_file"]:
                if not self.load_52wk_data_to_database(db_wrapper):
                    print("❌ PIPELINE FAILED: Could not load 52 Week High Low data")
                    self.logger.error("PIPELINE FAILED: Could not load 52 Week High Low data")
                    self.update_batch_status(stage_names[3], "Failed", db_wrapper)
                    return False
                
                print("✅ STAGE 5 COMPLETED: 52 Week High Low data loaded successfully")
                self.logger.info("STAGE 5 COMPLETED: 52 Week High Low data loaded successfully")
            else:
                print("⚠️ STAGE 5 SKIPPED: No 52 Week High Low file available")
                self.logger.warning("STAGE 5 SKIPPED: No 52 Week High Low file available")
            
            self.update_batch_status(stage_names[3], "Completed", db_wrapper)
            
        except Exception as e:
            print(f"❌ STAGE 5 FAILED: {e}")
            self.logger.error(f"STAGE 5 FAILED: {e}")
            self.update_batch_status(stage_names[3], "Failed", db_wrapper)
            return False
        
        # Stage 6: Process stock_data operations
        print("\n🚀 STAGE 6: PROCESS STOCK_DATA OPERATIONS")
        print("-" * 50)
        self.update_batch_status(stage_names[4], "Started", db_wrapper)
        self.logger.info("STAGE 6: Starting stock_data operations process")
        
        try:
            if not self.process_stock_data_operations(db_wrapper):
                print("❌ PIPELINE FAILED: Could not process stock_data operations")
                self.logger.error("PIPELINE FAILED: Could not process stock_data operations")
                self.update_batch_status(stage_names[4], "Failed", db_wrapper)
                return False
            
            print("✅ STAGE 6 COMPLETED: Stock data operations completed successfully")
            self.logger.info("STAGE 6 COMPLETED: Stock data operations completed successfully")
            self.update_batch_status(stage_names[4], "Completed", db_wrapper)
            
        except Exception as e:
            print(f"❌ STAGE 6 FAILED: {e}")
            self.logger.error(f"STAGE 6 FAILED: {e}")
            self.update_batch_status(stage_names[4], "Failed", db_wrapper)
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
        print("   ✅ Loaded BC data into corp_raw")
        print("   ✅ Loaded 52 Week High Low data into stg_52wk_highlow_raw")
        print("   ✅ Refreshed stock_data table with latest data")
        print("   ✅ Updated market cap values")
        print("   ✅ All batch process statuses updated")
        print("   🔌 Single database connection used throughout pipeline")
        print("=" * 90)
        
        self.logger.info("=" * 80)
        self.logger.info(f"DATABASE LOAD PIPELINE COMPLETED SUCCESSFULLY in {pipeline_duration:.2f} seconds")
        self.logger.info(f"Process date: {self.parsed_date}")
        self.logger.info("All operations completed: Load PD → Load MCAP → Load BC → Load 52WK → Process Stock Data")
        self.logger.info("All batch process statuses updated in stg_batch_process table")
        self.logger.info("Single database connection used throughout pipeline for improved performance")
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
    print("\n🔧 Performance Improvement:")
    print("   • Single database connection used throughout pipeline")
    print("   • Unified connection interface eliminates duplicate connection logic")
    print("   • Reduced connection overhead and improved transaction management")
    print("")
    
    # Initialize database load pipeline
    pipeline = NSEDatabaseLoad(
        download_path=r"C:\Python Code\Files"
    )
    
    try:
        # Run database load pipeline (Stages 2-6) with single connection
        success = pipeline.run_database_load_pipeline()
        
        if success:
            print("\n✅ Database loading completed successfully!")
            print("📊 Check nse_database_load.log for detailed operation records")
            print("📊 Check stg_batch_process table for stage-by-stage status tracking")
            print("📈 Stock data is now updated in the database")
            print("🔌 Pipeline used single database connection with unified interface")
        else:
            print("\n❌ Database loading pipeline failed.")
            print("📊 Check nse_database_load.log for details")
            print("📊 Check stg_batch_process table to see which stage failed")
            
    except Exception as e:
        print(f"\n💥 Unexpected pipeline error: {e}")
        pipeline.logger.error(f"UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()