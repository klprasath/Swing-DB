import os
import time
import zipfile
import pandas as pd
import shutil
import psycopg2
from psycopg2 import Error
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
import logging
from io import StringIO
import glob

class NSEDataPipeline:
    def __init__(self, download_path=r"C:\Python Code\Files", headless=False):
        self.download_path = download_path
        self.headless = headless
        self.driver = None
        self.extracted_files = {"pd_file": None, "mcap_file": None}
        self.parsed_date = None  # Store the parsed date for final operations
        
        # Configure logging with multiple sections
        self.setup_logging()
        
        # Create download directory if it doesn't exist
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
            print(f"✅ Created directory: {self.download_path}")
    
    def validate_trading_day(self, check_date=None):
        """
        Validate if the given date is a trading day (not weekend or holiday)
        Returns: (is_trading_day: bool, reason: str)
        """
        if check_date is None:
            check_date = datetime.now().date()
        
        # Convert to datetime.date if it's a datetime object
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        
        try:
            # Check if it's weekend (Saturday = 5, Sunday = 6)
            weekday = check_date.weekday()
            if weekday >= 5:  # Saturday or Sunday
                weekend_name = "Saturday" if weekday == 5 else "Sunday"
                reason = f"Today is {weekend_name} (weekend)"
                self.main_logger.info(reason)
                return False, reason
            
            # Check if it's a holiday using database
            with get_db_connection() as db:
                holiday_query = "SELECT COUNT(*) FROM stg_nse_holidays WHERE trd_dt = %s"
                db.execute(holiday_query, (check_date,))
                result = db.fetchone()
                
                if result and result[0] > 0:
                    reason = f"Today ({check_date}) is a NSE holiday"
                    self.main_logger.info(reason)
                    return False, reason
            
            # If we reach here, it's a valid trading day
            self.main_logger.info(f"Date {check_date} is a valid trading day")
            return True, "Valid trading day"
            
        except Exception as e:
            error_msg = f"Error validating trading day: {e}"
            self.main_logger.error(error_msg)
            # In case of error, assume it's a trading day to avoid blocking
            return True, "Validation error - assuming trading day"
    
    def setup_logging(self):
        """Setup logging with three different sections"""
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Section 1: Selenium/Extraction Logger
        self.selenium_logger = logging.getLogger('selenium_extraction')
        selenium_handler = logging.FileHandler('nse_selenium_extraction.log')
        selenium_handler.setFormatter(formatter)
        self.selenium_logger.addHandler(selenium_handler)
        self.selenium_logger.setLevel(logging.INFO)
        
        # Section 2: Database Loading Logger  
        self.db_load_logger = logging.getLogger('database_loading')
        db_handler = logging.FileHandler('nse_database_loading.log')
        db_handler.setFormatter(formatter)
        self.db_load_logger.addHandler(db_handler)
        self.db_load_logger.setLevel(logging.INFO)
        
        # Section 3: Stock Data Operations Logger
        self.stock_data_logger = logging.getLogger('stock_data_operations')
        stock_handler = logging.FileHandler('nse_stock_data_operations.log')
        stock_handler.setFormatter(formatter)
        self.stock_data_logger.addHandler(stock_handler)
        self.stock_data_logger.setLevel(logging.INFO)
        
        # Main pipeline logger
        self.main_logger = logging.getLogger('main_pipeline')
        main_handler = logging.FileHandler('nse_main_pipeline.log')
        main_handler.setFormatter(formatter)
        self.main_logger.addHandler(main_handler)
        self.main_logger.setLevel(logging.INFO)
    
    def setup_driver(self):
        """Setup Chrome driver with download preferences"""
        chrome_options = Options()
        
        # Set download directory
        prefs = {
            "download.default_directory": self.download_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Additional options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Initialize driver
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✅ Chrome driver initialized successfully")
            self.selenium_logger.info("Chrome driver initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing Chrome driver: {e}")
            self.selenium_logger.error(f"Error initializing Chrome driver: {e}")
            return False
    
    def download_bhavcopy(self, target_date="28-Aug-2025"):
        """Download Bhavcopy (PR) zip file for the specified date"""
        if not self.setup_driver():
            return None
        
        try:
            print(f"🌐 Navigating to NSE all-reports page...")
            self.selenium_logger.info("Navigating to NSE all-reports page")
            self.driver.get("https://www.nseindia.com/all-reports")
            
            # Wait for page to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            print("⏳ Waiting for dynamic content to load...")
            time.sleep(8)
            
            # Find date element containing target date
            print(f"🔍 Looking for date element: '{target_date}'")
            date_selectors = [
                f"//*[contains(text(), '{target_date}')]",
                f"//*[contains(text(), '28-Aug-2025')]"
            ]
            
            date_element = None
            for selector in date_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        date_element = elements[0]
                        print(f"✅ Found date element")
                        break
                except:
                    continue
            
            if not date_element:
                print("❌ Could not find date element")
                return None
            
            # Find parent container with checkboxes
            print("🔍 Looking for checkboxes container...")
            parent_container = date_element
            for _ in range(5):
                try:
                    parent_container = parent_container.find_element(By.XPATH, "..")
                    checkboxes_in_container = parent_container.find_elements(By.XPATH, ".//input[@type='checkbox']")
                    if len(checkboxes_in_container) > 5:
                        print(f"✅ Found container with {len(checkboxes_in_container)} checkboxes")
                        break
                except:
                    break
            
            # Find Bhavcopy checkbox
            print("🔍 Looking for 'Bhavcopy (PR)' checkbox...")
            all_checkboxes = parent_container.find_elements(By.XPATH, ".//input[@type='checkbox']")
            bhavcopy_checkbox = self.find_bhavcopy_checkbox(all_checkboxes)
            
            if not bhavcopy_checkbox:
                print("❌ Could not find Bhavcopy (PR) checkbox")
                return None
            
            # Uncheck all other checkboxes first
            print("🔄 Unchecking all other checkboxes...")
            unchecked_count = 0
            for checkbox in all_checkboxes:
                if checkbox != bhavcopy_checkbox and checkbox.is_selected():
                    try:
                        self.driver.execute_script("arguments[0].click();", checkbox)
                        unchecked_count += 1
                    except:
                        pass
            print(f"✅ Unchecked {unchecked_count} other checkboxes")
            
            # Check only Bhavcopy checkbox
            print("☑️ Checking Bhavcopy (PR) checkbox...")
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", bhavcopy_checkbox)
                time.sleep(1)
                
                if not bhavcopy_checkbox.is_selected():
                    self.driver.execute_script("arguments[0].click();", bhavcopy_checkbox)
                    time.sleep(2)
                    print("✅ Bhavcopy checkbox checked")
                
                # Verify selection
                selected_count = len([cb for cb in all_checkboxes if cb.is_selected()])
                print(f"📊 Total selected checkboxes: {selected_count}")
                
            except Exception as e:
                print(f"❌ Error checking Bhavcopy checkbox: {e}")
                return None
            
            # Find and click download button
            print("🔍 Looking for download button...")
            download_selectors = [
                "//button[contains(text(), 'Download')]",
                "//a[contains(text(), 'Download')]",
                "//input[@type='submit' and @value='Download']",
                "//button[@type='submit']"
            ]
            
            download_button = None
            for selector in download_selectors:
                try:
                    download_button = self.driver.find_element(By.XPATH, selector)
                    if download_button.is_displayed() and download_button.is_enabled():
                        print(f"✅ Found download button")
                        break
                except:
                    continue
            
            if download_button:
                print("⬇️ Clicking download button...")
                self.driver.execute_script("arguments[0].click();", download_button)
                
                # Wait for download
                possible_filenames = ["Reports-Daily-Multiple.zip", "PR280825.zip", "Reports-Daily.zip"]
                downloaded_file = None
                
                for filename in possible_filenames:
                    print(f"⏳ Checking for download: {filename}")
                    downloaded_file = self.wait_for_download(filename, timeout=30)
                    if downloaded_file:
                        break
                
                if not downloaded_file:
                    downloaded_file = self.find_latest_zip_file()
                
                return downloaded_file
            else:
                print("❌ Could not find download button")
                return None
                
        except Exception as e:
            print(f"❌ Error during download: {e}")
            self.selenium_logger.error(f"Error during download: {e}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
    
    def find_bhavcopy_checkbox(self, checkboxes):
        """Find the Bhavcopy (PR) checkbox by inspecting surrounding text"""
        print(f"🔍 Inspecting {len(checkboxes)} checkboxes for Bhavcopy...")
        
        for i, checkbox in enumerate(checkboxes):
            try:
                parent = checkbox.find_element(By.XPATH, "..")
                parent_text = parent.text.lower()
                
                # Get broader context
                try:
                    grandparent = parent.find_element(By.XPATH, "..")
                    context_text = grandparent.text.lower()
                except:
                    context_text = parent_text
                
                combined_text = f"{parent_text} {context_text}"
                
                # Look for Bhavcopy and PR keywords
                if ('bhavcopy' in combined_text or 'bhav' in combined_text) and 'pr' in combined_text:
                    print(f"✅ Found Bhavcopy checkbox #{i+1}")
                    return checkbox
                elif 'pr280825' in combined_text:
                    print(f"✅ Found PR280825 checkbox #{i+1}")
                    return checkbox
                    
            except Exception as e:
                continue
        
        print("❌ Could not identify Bhavcopy checkbox")
        return None
    
    def wait_for_download(self, expected_filename, timeout=60):
        """Wait for file download to complete"""
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            filepath = os.path.join(self.download_path, expected_filename)
            
            if os.path.exists(filepath):
                # Check if download is complete
                time.sleep(2)
                size1 = os.path.getsize(filepath)
                time.sleep(2)
                size2 = os.path.getsize(filepath)
                
                if size1 == size2 and size1 > 0:
                    print(f"✅ Download completed: {expected_filename} ({size1:,} bytes)")
                    self.selenium_logger.info(f"Download completed: {expected_filename} ({size1:,} bytes)")
                    return filepath
            
            time.sleep(1)
        
        return None
    
    def find_latest_zip_file(self):
        """Find the most recently created zip file"""
        try:
            zip_files = []
            for filename in os.listdir(self.download_path):
                if filename.lower().endswith('.zip'):
                    filepath = os.path.join(self.download_path, filename)
                    creation_time = os.path.getctime(filepath)
                    zip_files.append((filepath, creation_time, filename))
            
            if zip_files:
                zip_files.sort(key=lambda x: x[1], reverse=True)
                latest_file = zip_files[0][0]
                latest_name = zip_files[0][2]
                print(f"✅ Found latest zip file: {latest_name}")
                return latest_file
            else:
                print("❌ No zip files found")
                return None
                
        except Exception as e:
            print(f"❌ Error finding latest zip: {e}")
            return None
    
    def extract_specific_files(self, zip_filepath):
        """Extract only specific files and handle archiving"""
        if not zip_filepath or not os.path.exists(zip_filepath):
            print("❌ ZIP file not found")
            return []
        
        try:
            # Create Archives folder
            archives_path = os.path.join(self.download_path, "Archives")
            if not os.path.exists(archives_path):
                os.makedirs(archives_path)
                print(f"✅ Created Archives folder")
            
            print(f"\n📦 Processing: {os.path.basename(zip_filepath)}")
            
            # Extract main zip
            with zipfile.ZipFile(zip_filepath, 'r') as main_zip:
                file_list = main_zip.namelist()
                print(f"📂 Files in main ZIP: {file_list}")
                
                # Find PR280825.zip inside
                pr_zip_name = None
                for filename in file_list:
                    if 'PR280825.zip' in filename or ('pr' in filename.lower() and filename.lower().endswith('.zip')):
                        pr_zip_name = filename
                        break
                
                if not pr_zip_name:
                    print("❌ PR280825.zip not found in main ZIP")
                    return []
                
                # Extract PR280825.zip temporarily
                temp_pr_path = os.path.join(self.download_path, "temp_pr.zip")
                main_zip.extract(pr_zip_name, self.download_path)
                
                # Rename to standard name
                extracted_pr_path = os.path.join(self.download_path, pr_zip_name)
                if extracted_pr_path != temp_pr_path:
                    shutil.move(extracted_pr_path, temp_pr_path)
                
                print(f"✅ Extracted: {pr_zip_name}")
            
            # Copy PR280825.zip to Archives
            archive_dest = os.path.join(archives_path, "PR280825.zip")
            shutil.copy2(temp_pr_path, archive_dest)
            print(f"📚 Archived: PR280825.zip")
            
            # Extract specific files from PR zip
            target_files = ["MCAP28082025.csv", "Pd280825.csv"]
            extracted_csvs = []
            
            with zipfile.ZipFile(temp_pr_path, 'r') as pr_zip:
                pr_file_list = pr_zip.namelist()
                print(f"📂 Files in PR ZIP: {pr_file_list}")
                
                for target_file in target_files:
                    if target_file in pr_file_list:
                        # Extract with original name
                        pr_zip.extract(target_file, self.download_path)
                        extracted_path = os.path.join(self.download_path, target_file)
                        extracted_csvs.append(extracted_path)
                        print(f"✅ Extracted: {target_file}")
                        
                        # Store file references for database loading
                        if target_file.startswith("Pd"):
                            self.extracted_files["pd_file"] = extracted_path
                        elif target_file.startswith("MCAP"):
                            self.extracted_files["mcap_file"] = extracted_path
                        
                    else:
                        print(f"⚠️ File not found: {target_file}")
            
            # Clean up temporary files
            self.cleanup_except_targets(extracted_csvs)
            
            return extracted_csvs
            
        except Exception as e:
            print(f"❌ Error extracting files: {e}")
            self.selenium_logger.error(f"Error extracting files: {e}")
            return []
    
    def cleanup_except_targets(self, csv_files_to_keep):
        """Clean up all files except our target CSV files and Archives folder"""
        try:
            print("\n🧹 Cleaning up temporary files...")
            
            # Files and folders to keep
            keep_items = {"Archives"}  # Always keep Archives folder
            
            # Add our target CSV filenames to keep
            for csv_path in csv_files_to_keep:
                keep_items.add(os.path.basename(csv_path))
            
            deleted_count = 0
            for item in os.listdir(self.download_path):
                if item in keep_items:
                    print(f"   💾 Keeping: {item}")
                    continue
                
                item_path = os.path.join(self.download_path, item)
                
                if os.path.isdir(item_path):
                    print(f"   📁 Keeping folder: {item}")
                    continue
                
                if os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                        print(f"   🗑️ Deleted: {item}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"   ❌ Could not delete {item}: {e}")
            
            print(f"✅ Cleaned up {deleted_count} temporary files")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            self.selenium_logger.error(f"Error during cleanup: {e}")
    
    def load_pd_data_to_database(self):
        """Load Pd data to database (from data_load.py logic)"""
        if not self.extracted_files["pd_file"]:
            print("❌ No Pd file found to load")
            return False
        
        file_path = self.extracted_files["pd_file"]
        
        try:
            print(f"\n📊 LOADING PD DATA TO DATABASE")
            print(f"📁 File: {os.path.basename(file_path)}")
            self.db_load_logger.info(f"Starting Pd data load from: {file_path}")
            
            # Verify file accessibility
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                self.db_load_logger.error(f"File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File accessible: {file_size:.2f} MB")
            
            # Connect to database
            print("🔌 Connecting to PostgreSQL...")
            with get_db_connection() as connection:
                cursor = connection.cursor()
                print("✅ Connected to PostgreSQL")
                self.db_load_logger.info("Connected to PostgreSQL for Pd data load")
            
            # Step 1: Delete from nse_raw table
            print("🗑️ Deleting records from nse_raw table...")
            start_time = time.time()
            delete_query = "DELETE FROM nse_raw;"
            cursor.execute(delete_query)
            row_count = cursor.rowcount
            delete_duration = time.time() - start_time
            print(f"✅ Deleted {row_count} records in {delete_duration:.2f} seconds")
            self.db_load_logger.info(f"Deleted {row_count} records from nse_raw in {delete_duration:.2f} seconds")
            
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
                self.db_load_logger.info(f"Loaded CSV into nse_raw table in {load_duration:.2f} seconds")
            
            # Step 3: Extract date from filename (e.g., Pd280825.csv -> 280825)
            filename = os.path.basename(file_path)
            date_str = filename[2:8]  # Extract '280825' from 'Pd280825.csv'
            parsed_date = datetime.strptime(date_str, '%d%m%y').strftime('%Y-%m-%d')
            self.parsed_date = parsed_date  # Store for final operations
            print(f"📅 Parsed date: {parsed_date}")
            self.db_load_logger.info(f"Parsed date: {parsed_date}")
            
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
            self.db_load_logger.info(f"Inserted {row_count} rows into stg_nse_data in {insert_duration:.2f} seconds")
            
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
            self.db_load_logger.info(f"Inserted {row_count} rows into stg_nse_data_hist in {insert_duration:.2f} seconds")
            
                # Commit transaction
                connection.commit()
                print("✅ Pd data load completed successfully")
                self.db_load_logger.info("Pd data load transaction committed successfully")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading Pd data: {e}")
            self.db_load_logger.error(f"Error loading Pd data: {e}")
            return False
    
    def load_mcap_data_to_database(self):
        """Load MCAP data to database (from mcap_load.py logic)"""
        if not self.extracted_files["mcap_file"]:
            print("❌ No MCAP file found to load")
            return False
        
        file_path = self.extracted_files["mcap_file"]
        
        try:
            print(f"\n💰 LOADING MCAP DATA TO DATABASE")
            print(f"📁 File: {os.path.basename(file_path)}")
            self.db_load_logger.info(f"Starting MCAP data load from: {file_path}")
            
            # Verify file accessibility
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                self.db_load_logger.error(f"File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✅ File accessible: {file_size:.2f} MB")
            
            # Connect to database
            print("🔌 Connecting to PostgreSQL...")
            with get_db_connection() as connection:
                cursor = connection.cursor()
                print("✅ Connected to PostgreSQL")
                self.db_load_logger.info("Connected to PostgreSQL for MCAP data load")
            
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
                self.db_load_logger.warning(f"Locks detected on mcap_raw: {locks}")
            else:
                print("✅ No locks detected on mcap_raw")
                self.db_load_logger.info("No locks detected on mcap_raw")
            
            # Step 1: Delete from mcap_raw table
            print("🗑️ Deleting records from mcap_raw table...")
            start_time = time.time()
            delete_query = "DELETE FROM mcap_raw;"
            cursor.execute(delete_query)
            row_count = cursor.rowcount
            delete_duration = time.time() - start_time
            print(f"✅ Deleted {row_count} records in {delete_duration:.2f} seconds")
            self.db_load_logger.info(f"Deleted {row_count} records from mcap_raw in {delete_duration:.2f} seconds")
            
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
                self.db_load_logger.info(f"Loaded CSV into mcap_raw table in {load_duration:.2f} seconds")
            
            # Step 3: Extract date from filename (e.g., MCAP28082025.csv -> 28082025)
            filename = os.path.basename(file_path)
            date_str = filename[4:12]  # Extract '28082025' from 'MCAP28082025.csv'
            parsed_date = datetime.strptime(date_str, '%d%m%Y').strftime('%Y-%m-%d')
            print(f"📅 Parsed date: {parsed_date}")
            self.db_load_logger.info(f"Parsed date: {parsed_date}")
            
            # Step 4: Delete from stg_mcap_data table
            print("🗑️ Deleting records from stg_mcap_data table...")
            start_time = time.time()
            delete_query = "DELETE FROM stg_mcap_data;"
            cursor.execute(delete_query)
            row_count = cursor.rowcount
            delete_duration = time.time() - start_time
            print(f"✅ Deleted {row_count} records in {delete_duration:.2f} seconds")
            self.db_load_logger.info(f"Deleted {row_count} records from stg_mcap_data in {delete_duration:.2f} seconds")
            
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
            self.db_load_logger.info(f"Inserted {row_count} rows into stg_mcap_data in {insert_duration:.2f} seconds")
            
                # Commit transaction
                connection.commit()
                print("✅ MCAP data load completed successfully")
                self.db_load_logger.info("MCAP data load transaction committed successfully")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading MCAP data: {e}")
            self.db_load_logger.error(f"Error loading MCAP data: {e}")
            return False
    
    def process_stock_data_operations(self):
        """Final stock_data delete/insert/update operations"""
        if not self.parsed_date:
            print("❌ No parsed date available for stock_data operations")
            return False
        
        try:
            print(f"\n📈 PROCESSING STOCK_DATA OPERATIONS")
            print(f"📅 Using date: {self.parsed_date}")
            self.stock_data_logger.info(f"Starting stock_data operations for date: {self.parsed_date}")
            
            # Connect to database
            print("🔌 Connecting to PostgreSQL...")
            with get_db_connection() as connection:
                cursor = connection.cursor()
                print("✅ Connected to PostgreSQL")
                self.stock_data_logger.info("Connected to PostgreSQL for stock_data operations")
            
            # Step 1: Delete from stock_data
            print("🗑️ Deleting all records from stock_data table...")
            start_time = time.time()
            delete_query = "DELETE FROM stock_data;"
            cursor.execute(delete_query)
            row_count = cursor.rowcount
            delete_duration = time.time() - start_time
            print(f"✅ Deleted {row_count} records from stock_data in {delete_duration:.2f} seconds")
            self.stock_data_logger.info(f"Deleted {row_count} records from stock_data in {delete_duration:.2f} seconds")
            
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
            self.stock_data_logger.info(f"Inserted {row_count} rows into stock_data for date {self.parsed_date} in {insert_duration:.2f} seconds")
            
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
            self.stock_data_logger.info(f"Updated {row_count} rows with market cap data in {update_duration:.2f} seconds")
            
                # Step 4: Commit all stock_data operations
                connection.commit()
                print("✅ Stock data operations completed successfully")
                self.stock_data_logger.info("Stock data operations transaction committed successfully")
                
                return True
                
        except Exception as e:
            print(f"❌ Error in stock_data operations: {e}")
            self.stock_data_logger.error(f"Error in stock_data operations: {e}")
            return False
    
    def run_complete_pipeline(self, target_date="28-Aug-2025"):
        """Run the complete NSE data pipeline"""
        print("=" * 90)
        print("                    NSE COMPLETE DATA PIPELINE")
        print("      Download → Extract → Load PD → Load MCAP → Process Stock Data")
        print("=" * 90)
        
        pipeline_start_time = time.time()
        self.main_logger.info("Starting complete NSE data pipeline")
        
        # VALIDATION: Check if today is a trading day (not weekend or holiday)
        print("\n🔍 VALIDATING TRADING DAY")
        print("-" * 50)
        
        is_trading_day, reason = self.validate_trading_day()
        if not is_trading_day:
            print(f"❌ PIPELINE STOPPED: {reason}")
            self.main_logger.info(f"Pipeline stopped: {reason}")
            return False
        
        print(f"✅ {reason} - Proceeding with pipeline")
        
        # Step 1: Download and extract files
        print("\n🚀 STEP 1: DOWNLOAD & EXTRACT NSE FILES")
        print("-" * 50)
        self.selenium_logger.info("=== SELENIUM EXTRACTION PROCESS STARTED ===")
        
        zip_file = self.download_bhavcopy(target_date)
        if not zip_file:
            print("❌ PIPELINE FAILED: Could not download files")
            self.main_logger.error("Pipeline failed at download step")
            return False
        
        csv_files = self.extract_specific_files(zip_file)
        if not csv_files:
            print("❌ PIPELINE FAILED: Could not extract CSV files")
            self.main_logger.error("Pipeline failed at extraction step")
            return False
        
        print(f"✅ STEP 1 COMPLETED: Extracted {len(csv_files)} files")
        for csv_file in csv_files:
            print(f"   • {os.path.basename(csv_file)}")
        self.selenium_logger.info("=== SELENIUM EXTRACTION PROCESS COMPLETED ===")
        self.main_logger.info(f"Step 1 completed: Extracted {len(csv_files)} files")
        
        # Step 2: Load PD data
        print("\n🚀 STEP 2: LOAD PD DATA TO DATABASE")
        print("-" * 50)
        self.db_load_logger.info("=== DATABASE LOADING PROCESS STARTED ===")
        
        if not self.load_pd_data_to_database():
            print("❌ PIPELINE FAILED: Could not load PD data")
            self.main_logger.error("Pipeline failed at PD data loading step")
            return False
        
        print("✅ STEP 2 COMPLETED: PD data loaded successfully")
        self.main_logger.info("Step 2 completed: PD data loaded successfully")
        
        # Step 3: Load MCAP data
        print("\n🚀 STEP 3: LOAD MCAP DATA TO DATABASE")
        print("-" * 50)
        
        if not self.load_mcap_data_to_database():
            print("❌ PIPELINE FAILED: Could not load MCAP data")
            self.main_logger.error("Pipeline failed at MCAP data loading step")
            return False
        
        print("✅ STEP 3 COMPLETED: MCAP data loaded successfully")
        self.db_load_logger.info("=== DATABASE LOADING PROCESS COMPLETED ===")
        self.main_logger.info("Step 3 completed: MCAP data loaded successfully")
        
        # Step 4: Process stock_data operations
        print("\n🚀 STEP 4: PROCESS STOCK_DATA OPERATIONS")
        print("-" * 50)
        self.stock_data_logger.info("=== STOCK DATA OPERATIONS PROCESS STARTED ===")
        
        if not self.process_stock_data_operations():
            print("❌ PIPELINE FAILED: Could not process stock_data operations")
            self.main_logger.error("Pipeline failed at stock_data operations step")
            return False
        
        print("✅ STEP 4 COMPLETED: Stock data operations completed successfully")
        self.stock_data_logger.info("=== STOCK DATA OPERATIONS PROCESS COMPLETED ===")
        self.main_logger.info("Step 4 completed: Stock data operations completed successfully")
        
        # Pipeline completion
        pipeline_duration = time.time() - pipeline_start_time
        
        print("\n" + "=" * 90)
        print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY! 🎉")
        print("=" * 90)
        print(f"⏱️  Total pipeline duration: {pipeline_duration:.2f} seconds")
        print(f"📁 Files location: {self.download_path}")
        print(f"📚 Archive location: {os.path.join(self.download_path, 'Archives')}")
        print(f"📊 Database: Updated with latest stock data")
        print(f"📅 Process date: {self.parsed_date}")
        print("\n📋 COMPLETED OPERATIONS:")
        print("   ✅ Downloaded and extracted NSE files")
        print("   ✅ Loaded PD data into stg_nse_data & stg_nse_data_hist")
        print("   ✅ Loaded MCAP data into stg_mcap_data")
        print("   ✅ Refreshed stock_data table with latest data")
        print("   ✅ Updated market cap values")
        print("=" * 90)
        
        self.main_logger.info(f"Complete pipeline finished successfully in {pipeline_duration:.2f} seconds")
        self.main_logger.info("=== COMPLETE NSE DATA PIPELINE FINISHED ===")
        
        return True

def main():
    print("📋 Requirements:")
    print("   • pip install selenium pandas requests psycopg2")
    print("   • ChromeDriver: https://chromedriver.chromium.org/")
    print("   • PostgreSQL database with required tables")
    print("   • Database connection configured via db_config.py and db_connection.py")
    print("\n📄 Log Files Generated:")
    print("   • nse_selenium_extraction.log - Download & extraction process")
    print("   • nse_database_loading.log - Database loading operations")  
    print("   • nse_stock_data_operations.log - Final stock_data processing")
    print("   • nse_main_pipeline.log - Overall pipeline status")
    print("\n🔍 Validations:")
    print("   • Holiday validation using stg_nse_holidays table")
    print("   • Weekend validation (Saturday/Sunday)")
    print("   • Database connectivity validation")
    print("")
    
    # Initialize pipeline
    pipeline = NSEDataPipeline(
        download_path=r"C:\Python Code\Files",
        headless=False  # Set to True to hide browser
    )
    
    try:
        # Run complete pipeline (includes trading day validation)
        success = pipeline.run_complete_pipeline("28-Aug-2025")
        
        if success:
            print("\n✅ All operations completed successfully!")
            print("📊 Check log files for detailed operation records")
        else:
            print("\n❌ Pipeline failed or stopped due to validation.")
            print("📊 Check log files for details and validation results")
            
    except Exception as e:
        print(f"\n💥 Unexpected pipeline error: {e}")
        pipeline.main_logger.error(f"Unexpected pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()