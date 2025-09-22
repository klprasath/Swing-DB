import os
import time
import zipfile
import shutil
import json
import psycopg2
from psycopg2 import Error
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import logging
import csv
from io import StringIO

class NSEDownloadExtract:
    def __init__(self, download_path=r"C:\Python Code\Files", headless=False):
        self.download_path = download_path
        self.headless = headless
        self.driver = None
        self.extracted_files = {"pd_file": None, "mcap_file": None, "highlow_file": None}
        self.parsed_date = None
        
        # Configure logging
        self.setup_logging()
        
        # Database configuration (for batch tracking only)
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
            print("✅ Using db_connection module for batch tracking")
            self.logger.info("Using db_connection module for batch tracking")
        except ImportError:
            print("⚠️ db_connection module not found, using direct connection for batch tracking")
            self.logger.warning("db_connection module not found, using direct connection for batch tracking")
            self.use_db_connection_module = False
        
        # Create download directory if it doesn't exist
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
            print(f"✅ Created directory: {self.download_path}")
            self.logger.info(f"Created directory: {self.download_path}")
    
    def get_current_date_string(self):
        """Get current date in DD-MMM-YYYY format for NSE reports"""
        current_date = datetime.now()
        return current_date.strftime("%d-%b-%Y")
    
    def get_current_date_ddmmyyyy(self):
        """Get current date in DDMMYYYY format for 52 Week High Low Report"""
        current_date = datetime.now()
        return current_date.strftime("%d%m%Y")
    
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
                self.logger.info(f"VALIDATION: {reason}")
                return False, reason
            
            # Check if it's a holiday using database
            if self.use_db_connection_module and self.db_connection_func:
                with self.db_connection_func() as db:
                    holiday_query = "SELECT COUNT(*) FROM stg_nse_holidays WHERE trd_dt = %s"
                    db.execute(holiday_query, (check_date,))
                    result = db.fetchone()
            else:
                connection = psycopg2.connect(**self.db_config, connect_timeout=10)
                try:
                    cursor = connection.cursor()
                    holiday_query = "SELECT COUNT(*) FROM stg_nse_holidays WHERE trd_dt = %s"
                    cursor.execute(holiday_query, (check_date,))
                    result = cursor.fetchone()
                finally:
                    cursor.close()
                    connection.close()
                
            if result and result[0] > 0:
                reason = f"Today ({check_date}) is a NSE holiday"
                self.logger.info(f"VALIDATION: {reason}")
                return False, reason
            
            # If we reach here, it's a valid trading day
            self.logger.info(f"VALIDATION: Date {check_date} is a valid trading day")
            return True, "Valid trading day"
            
        except Exception as e:
            error_msg = f"Error validating trading day: {e}"
            self.logger.error(f"VALIDATION ERROR: {error_msg}")
            # In case of error, assume it's a trading day to avoid blocking
            return True, "Validation error - assuming trading day"
    
    def update_batch_status(self, step_name, status):
        """Update single row in batch process tracking table for download steps only"""
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
    
    def setup_logging(self):
        """Setup consolidated logging for download script"""
        # Create consolidated logger
        self.logger = logging.getLogger('nse_download_extract')
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler('nse_download_extract.log')
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
        # Log start of new session
        self.logger.info("=" * 80)
        self.logger.info("NSE DOWNLOAD & EXTRACT SESSION STARTED")
        self.logger.info("=" * 80)
    
    def setup_driver(self):
        """Setup Chrome driver with download preferences - ENHANCED FOR CSV DOWNLOADS"""
        chrome_options = Options()
        
        # Enhanced download directory settings
        prefs = {
            "download.default_directory": self.download_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "safebrowsing.disable_download_protection": True,
            "profile.default_content_settings.popups": 0,
            "profile.content_settings.pattern_pairs.*.multiple-automatic-downloads": 1,
            "profile.default_content_setting_values.automatic_downloads": 1,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Additional options for download handling
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Initialize driver
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✅ Chrome driver initialized successfully")
            self.logger.info("STEP 1: Chrome driver initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing Chrome driver: {e}")
            self.logger.error(f"STEP 1 ERROR: Error initializing Chrome driver: {e}")
            return False
    
    def check_browser_downloads(self):
        """Check if there are any pending downloads in the browser"""
        try:
            # Open downloads page
            self.driver.get("chrome://downloads/")
            time.sleep(3)
            
            # Get download items
            download_items = self.driver.execute_script("""
                return document.querySelector('downloads-manager')
                    .shadowRoot.querySelector('#downloadsList')
                    .items.length;
            """)
            
            print(f"📊 Browser downloads count: {download_items}")
            
            # Get download details if any
            if download_items > 0:
                download_info = self.driver.execute_script("""
                    const downloads = document.querySelector('downloads-manager')
                        .shadowRoot.querySelector('#downloadsList').items;
                    return Array.from(downloads).slice(0, 3).map(item => ({
                        fileName: item.fileName,
                        state: item.state,
                        filePath: item.filePath
                    }));
                """)
                print(f"🔍 Recent downloads: {download_info}")
                return download_info
                
        except Exception as e:
            print(f"⚠️ Could not check browser downloads: {e}")
        
        return []

    def download_bhavcopy(self, target_date=None):
        """Download Bhavcopy (PR) zip file for the specified date"""
        if target_date is None:
            target_date = self.get_current_date_string()
        
        if not self.setup_driver():
            return None
        
        try:
            print(f"🌐 Navigating to NSE all-reports page...")
            print(f"🗓️ Looking for data for date: {target_date}")
            self.logger.info(f"STEP 1: Navigating to NSE all-reports page for date: {target_date}")
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
                f"//*[contains(text(), '{self.get_current_date_string()}')]"
            ]
            
            date_element = None
            for selector in date_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        date_element = elements[0]
                        print(f"✅ Found date element")
                        self.logger.info(f"STEP 1: Found date element for {target_date}")
                        break
                except:
                    continue
            
            if not date_element:
                print("❌ Could not find date element")
                self.logger.error(f"STEP 1 ERROR: Could not find date element for {target_date}")
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
                        self.logger.info(f"STEP 1: Found container with {len(checkboxes_in_container)} checkboxes")
                        break
                except:
                    break
            
            # Find Bhavcopy checkbox
            print("🔍 Looking for 'Bhavcopy (PR)' checkbox...")
            all_checkboxes = parent_container.find_elements(By.XPATH, ".//input[@type='checkbox']")
            bhavcopy_checkbox = self.find_bhavcopy_checkbox(all_checkboxes)

            if not bhavcopy_checkbox:
                print("❌ Could not find Bhavcopy (PR) checkbox")
                self.logger.error("STEP 1 ERROR: Could not find Bhavcopy (PR) checkbox")
                return None

            # Also find 52 Week High Low checkbox
            print("🔍 Looking for '52 Week High Low Report' checkbox...")
            highlow_checkbox = self.find_52wk_checkbox(all_checkboxes)

            if not highlow_checkbox:
                print("⚠️ Could not find 52 Week High Low checkbox - proceeding with bhavcopy only")
                self.logger.warning("STEP 1: Could not find 52 Week High Low checkbox")
            
            # Uncheck ALL checkboxes on the entire page first
            print("🔄 Clearing ALL checkboxes on the page...")
            try:
                all_page_checkboxes = self.driver.find_elements(By.XPATH, "//input[@type='checkbox']")
                print(f"📊 Found {len(all_page_checkboxes)} total checkboxes on page")
                
                unchecked_count = 0
                for checkbox in all_page_checkboxes:
                    try:
                        if checkbox != bhavcopy_checkbox and checkbox.is_selected():
                            self.driver.execute_script("arguments[0].click();", checkbox)
                            unchecked_count += 1
                            time.sleep(0.2)
                    except:
                        pass
                
                print(f"✅ Cleared {unchecked_count} checkboxes")
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Error clearing checkboxes: {e}")
            print(f"✅ Unchecked {unchecked_count} other checkboxes")
            self.logger.info(f"STEP 1: Unchecked {unchecked_count} other checkboxes")
            
            # Check only Bhavcopy checkbox
            print("☑️ Checking Bhavcopy (PR) checkbox...")
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", bhavcopy_checkbox)
                time.sleep(1)
                
                if not bhavcopy_checkbox.is_selected():
                    self.driver.execute_script("arguments[0].click();", bhavcopy_checkbox)
                    time.sleep(2)
                    print("✅ Bhavcopy checkbox checked")
                    self.logger.info("STEP 1: Bhavcopy checkbox checked successfully")

                # Also check 52 Week High Low checkbox if found
                if highlow_checkbox:
                    print("☑️ Checking 52 Week High Low Report checkbox...")
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", highlow_checkbox)
                    time.sleep(1)
                    
                    if not highlow_checkbox.is_selected():
                        self.driver.execute_script("arguments[0].click();", highlow_checkbox)
                        time.sleep(2)
                        print("✅ 52 Week High Low checkbox checked")
                        self.logger.info("STEP 1: 52 Week High Low checkbox checked successfully")
                
                # Verify selection
                selected_count = len([cb for cb in all_checkboxes if cb.is_selected()])
                print(f"📊 Total selected checkboxes: {selected_count}")
                
            except Exception as e:
                print(f"❌ Error checking Bhavcopy checkbox: {e}")
                self.logger.error(f"STEP 1 ERROR: Error checking Bhavcopy checkbox: {e}")
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
                self.logger.info("STEP 1: Download button clicked, waiting for file...")
                
                # Wait for download
                possible_filenames = ["Reports-Daily-Multiple.zip", "Reports-Daily.zip"]
                downloaded_file = None
                
                for filename in possible_filenames:
                    print(f"⏳ Checking for download: {filename}")
                    downloaded_file = self.wait_for_download(filename, timeout=30)
                    if downloaded_file:
                        break
                
                if not downloaded_file:
                    downloaded_file = self.find_latest_zip_file()
                
                if downloaded_file:
                    self.logger.info(f"STEP 1: Download completed successfully - {os.path.basename(downloaded_file)}")
                
                return downloaded_file
            else:
                print("❌ Could not find download button")
                self.logger.error("STEP 1 ERROR: Could not find download button")
                return None
                
        except Exception as e:
            print(f"❌ Error during download: {e}")
            self.logger.error(f"STEP 1 ERROR: Error during download: {e}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
    
    def find_52wk_checkbox(self, checkboxes):
        """Find the 52 Week High Low Report checkbox by inspecting surrounding text - IMPROVED VERSION"""
        print(f"🔍 Inspecting {len(checkboxes)} checkboxes for 52 Week High Low Report...")
        
        for i, checkbox in enumerate(checkboxes):
            try:
                # Get multiple levels of context to find the right checkbox
                contexts = []
                
                # Level 1: Direct parent
                try:
                    parent = checkbox.find_element(By.XPATH, "..")
                    contexts.append(parent.text.lower())
                except:
                    pass
                
                # Level 2: Grandparent
                try:
                    grandparent = checkbox.find_element(By.XPATH, "../..")
                    contexts.append(grandparent.text.lower())
                except:
                    pass
                
                # Level 3: Great-grandparent  
                try:
                    great_grandparent = checkbox.find_element(By.XPATH, "../../..")
                    contexts.append(great_grandparent.text.lower())
                except:
                    pass
                
                # Level 4: Look for sibling elements
                try:
                    parent = checkbox.find_element(By.XPATH, "..")
                    siblings = parent.find_elements(By.XPATH, "..//*")
                    for sibling in siblings[:10]:  # Check first 10 siblings
                        try:
                            contexts.append(sibling.text.lower())
                        except:
                            pass
                except:
                    pass
                
                # Combine all contexts
                combined_text = " ".join(contexts)
                
                print(f"  Checkbox #{i+1} context: {combined_text[:200]}...")  # Show first 200 chars for debugging
                
                # Enhanced keyword matching
                high_low_keywords = [
                    '52 week high low',
                    '52week high low', 
                    '52-week high low',
                    '52 week high',
                    '52 week low',
                    'cm_52_wk_high_low',
                    'cm 52 wk high low',
                    '52wk high low',
                    'week high low report'
                ]
                
                # Check for any of the keyword patterns
                text_normalized = combined_text.replace('_', ' ').replace('-', ' ').replace('\n', ' ')
                
                for keyword in high_low_keywords:
                    if keyword in text_normalized:
                        print(f"✅ Found 52 Week High Low Report checkbox #{i+1} using keyword: '{keyword}'")
                        return checkbox
                
                # Alternative check: look for "52" and ("high" or "low") and "week"
                if ('52' in combined_text and 'week' in combined_text and 
                    ('high' in combined_text or 'low' in combined_text)):
                    print(f"✅ Found 52 Week High Low Report checkbox #{i+1} using pattern matching")
                    return checkbox
                    
            except Exception as e:
                print(f"  Error checking checkbox #{i+1}: {e}")
                continue
        
        print("❌ Could not identify 52 Week High Low Report checkbox")
        print("🔍 Available checkbox contexts:")
        
        # Debug: Show all checkbox contexts
        for i, checkbox in enumerate(checkboxes[:10]):  # Show first 10 for debugging
            try:
                parent = checkbox.find_element(By.XPATH, "..")
                context = parent.text.lower()[:100]  # First 100 chars
                print(f"  Checkbox #{i+1}: {context}")
            except:
                print(f"  Checkbox #{i+1}: <could not get context>")
                
        return None
    
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
                elif 'pr' in combined_text and ('290825' in combined_text or '280825' in combined_text):
                    print(f"✅ Found PR checkbox #{i+1}")
                    return checkbox
                    
            except Exception as e:
                continue
        
        print("❌ Could not identify Bhavcopy checkbox")
        return None
    
    def wait_for_download(self, expected_filename, timeout=60):
        """Wait for file download to complete - ENHANCED VERSION"""
        end_time = time.time() + timeout
        
        print(f"🔍 Waiting for file: {expected_filename} in {self.download_path}")
        
        while time.time() < end_time:
            filepath = os.path.join(self.download_path, expected_filename)
            
            # Check if file exists
            if os.path.exists(filepath):
                print(f"🔍 File found: {expected_filename}")
                # Check if download is complete by monitoring file size
                try:
                    time.sleep(3)  # Wait a bit
                    size1 = os.path.getsize(filepath)
                    time.sleep(3)  # Wait again
                    size2 = os.path.getsize(filepath)
                    
                    if size1 == size2 and size1 > 0:
                        print(f"✅ Download completed: {expected_filename} ({size1:,} bytes)")
                        self.logger.info(f"STEP 1: Download completed: {expected_filename} ({size1:,} bytes)")
                        return filepath
                    else:
                        print(f"⏳ File still downloading... Size: {size2:,} bytes")
                except Exception as e:
                    print(f"⚠️ Error checking file size: {e}")
            
            # List current files for debugging
            if int(time.time()) % 10 == 0:  # Every 10 seconds
                try:
                    current_files = os.listdir(self.download_path)
                    print(f"📂 Current files: {current_files}")
                except:
                    pass
            
            time.sleep(1)
        
        print(f"❌ Timeout waiting for {expected_filename}")
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
                self.logger.info(f"STEP 1: Found latest zip file: {latest_name}")
                return latest_file
            else:
                print("❌ No zip files found")
                self.logger.error("STEP 1 ERROR: No zip files found")
                return None
                
        except Exception as e:
            print(f"❌ Error finding latest zip: {e}")
            self.logger.error(f"STEP 1 ERROR: Error finding latest zip: {e}")
            return None
    
    def extract_specific_files(self, zip_filepath):
        """Extract specific files including 52 Week High Low CSV and handle archiving"""
        if not zip_filepath or not os.path.exists(zip_filepath):
            print("❌ ZIP file not found")
            self.logger.error("STEP 1 ERROR: ZIP file not found for extraction")
            return []
        
        try:
            # Create Archives folder
            archives_path = os.path.join(self.download_path, "Archives")
            if not os.path.exists(archives_path):
                os.makedirs(archives_path)
                print(f"✅ Created Archives folder")
                self.logger.info("STEP 1: Created Archives folder")
            
            print(f"\n📦 Processing: {os.path.basename(zip_filepath)}")
            self.logger.info(f"STEP 1: Processing ZIP file: {os.path.basename(zip_filepath)}")
            
            extracted_csvs = []
            
            # Extract main zip and look for files
            with zipfile.ZipFile(zip_filepath, 'r') as main_zip:
                file_list = main_zip.namelist()
                print(f"📂 Files in main ZIP: {file_list}")
                self.logger.info(f"STEP 1: Files in main ZIP: {file_list}")
                
                # Look for different types of files:
                # 1. PR*.zip file (contains PD and MCAP files)
                # 2. CM_52_wk_High_low*.csv (52 Week High Low report)
                
                pr_zip_name = None
                actual_date = None
                highlow_csv_name = None
                
                for filename in file_list:
                    # Check for PR zip file
                    if filename.startswith('PR') and filename.lower().endswith('.zip'):
                        pr_zip_name = filename
                        # Extract date from PR filename (e.g., PR290825.zip -> 290825)
                        actual_date = filename[2:8]  # Extract DDMMYY
                        print(f"🗓️ Detected actual date: {actual_date}")
                        self.logger.info(f"STEP 1: Detected actual date from PR file: {actual_date}")
                    
                    # Check for 52 Week High Low CSV file
                    elif filename.lower().startswith('cm_52_wk_high_low') and filename.lower().endswith('.csv'):
                        highlow_csv_name = filename
                        print(f"📈 Found 52 Week High Low CSV: {highlow_csv_name}")
                        self.logger.info(f"STEP 1: Found 52 Week High Low CSV: {highlow_csv_name}")
                
                # Set parsed_date early for batch tracking
                if actual_date:
                    # Convert DDMMYY to YYYY-MM-DD format for database operations
                    day = actual_date[:2]
                    month = actual_date[2:4] 
                    year_yy = actual_date[4:6]
                    self.parsed_date = f"20{year_yy}-{month}-{day}"
                    print(f"📅 Set tracking date: {self.parsed_date}")
                    self.logger.info(f"STEP 1: Set tracking date: {self.parsed_date}")
                
                # Extract 52 Week High Low CSV if found
                if highlow_csv_name:
                    print(f"📈 Extracting 52 Week High Low CSV: {highlow_csv_name}")
                    main_zip.extract(highlow_csv_name, self.download_path)
                    extracted_path = os.path.join(self.download_path, highlow_csv_name)
                    extracted_csvs.append(extracted_path)
                    
                    # Store file reference for database loading
                    self.extracted_files["highlow_file"] = extracted_path
                    print(f"✅ Extracted 52 Week High Low CSV: {highlow_csv_name}")
                    self.logger.info(f"STEP 1: Successfully extracted 52 Week High Low CSV: {highlow_csv_name}")
                    
                    # Archive the 52 Week High Low file
                    self.archive_52wk_file(extracted_path)
                
                # Process PR zip file for PD and MCAP files
                if pr_zip_name:
                    # Extract PR*.zip temporarily
                    temp_pr_path = os.path.join(self.download_path, "temp_pr.zip")
                    main_zip.extract(pr_zip_name, self.download_path)
                    
                    # Rename to standard name
                    extracted_pr_path = os.path.join(self.download_path, pr_zip_name)
                    if extracted_pr_path != temp_pr_path:
                        shutil.move(extracted_pr_path, temp_pr_path)
                    
                    print(f"✅ Extracted: {pr_zip_name}")
                    
                    # Copy PR*.zip to Archives with correct date
                    if actual_date:
                        archive_dest = os.path.join(archives_path, f"PR{actual_date}.zip")
                    else:
                        archive_dest = os.path.join(archives_path, "PR_latest.zip")
                    
                    shutil.copy2(temp_pr_path, archive_dest)
                    print(f"📚 Archived: {os.path.basename(archive_dest)}")
                    self.logger.info(f"STEP 1: Archived PR file: {os.path.basename(archive_dest)}")
                    
                    # Dynamically construct target filenames based on detected date
                    if actual_date:
                        # Convert DDMMYY to DDMMYYYY for MCAP file
                        day = actual_date[:2]
                        month = actual_date[2:4] 
                        year_yy = actual_date[4:6]
                        year_yyyy = "20" + year_yy  # Assume 20xx
                        
                        target_files = [
                            f"MCAP{day}{month}{year_yyyy}.csv",  # MCAP29082025.csv
                            f"Pd{actual_date}.csv"              # Pd290825.csv
                        ]
                        print(f"🎯 Target files: {target_files}")
                        self.logger.info(f"STEP 1: Target files for extraction: {target_files}")
                    else:
                        print("❌ Could not determine date from PR filename")
                        self.logger.error("STEP 1 ERROR: Could not determine date from PR filename")
                        return extracted_csvs  # Return what we have so far
                    
                    # Extract PD and MCAP files from PR zip
                    with zipfile.ZipFile(temp_pr_path, 'r') as pr_zip:
                        pr_file_list = pr_zip.namelist()
                        print(f"📂 Files in PR ZIP: {pr_file_list}")
                        self.logger.info(f"STEP 1: Files in PR ZIP: {pr_file_list}")
                        
                        for target_file in target_files:
                            if target_file in pr_file_list:
                                # Extract with original name
                                pr_zip.extract(target_file, self.download_path)
                                extracted_path = os.path.join(self.download_path, target_file)
                                extracted_csvs.append(extracted_path)
                                print(f"✅ Extracted: {target_file}")
                                self.logger.info(f"STEP 1: Successfully extracted: {target_file}")
                                
                                # Store file references for database loading
                                if target_file.startswith("Pd"):
                                    self.extracted_files["pd_file"] = extracted_path
                                elif target_file.startswith("MCAP"):
                                    self.extracted_files["mcap_file"] = extracted_path
                                
                            else:
                                print(f"⚠️ File not found: {target_file}")
                                self.logger.warning(f"STEP 1: File not found: {target_file}")
                                # List similar files for debugging
                                similar_files = [f for f in pr_file_list if target_file[:4].lower() in f.lower()]
                                if similar_files:
                                    print(f"   🔍 Similar files found: {similar_files}")
                                    self.logger.info(f"STEP 1: Similar files found: {similar_files}")
                    
                    # Clean up temporary PR zip
                    try:
                        os.remove(temp_pr_path)
                    except:
                        pass
                
                else:
                    print("⚠️ No PR*.zip file found in main ZIP")
                    self.logger.warning("STEP 1: No PR*.zip file found in main ZIP")
            
            # Clean up temporary files (but preserve our extracted CSVs)
            self.cleanup_except_targets(extracted_csvs)
            
            print(f"✅ Extraction completed - {len(extracted_csvs)} files extracted")
            for csv_file in extracted_csvs:
                print(f"   • {os.path.basename(csv_file)}")
            
            self.logger.info(f"STEP 1: File extraction completed - {len(extracted_csvs)} files extracted")
            return extracted_csvs
            
        except Exception as e:
            print(f"❌ Error extracting files: {e}")
            self.logger.error(f"STEP 1 ERROR: Error extracting files: {e}")
            return []
    
    def archive_52wk_file(self, csv_filepath):
        """Archive the 52 Week High Low CSV file"""
        try:
            # Create Archives folder if not exists
            archives_path = os.path.join(self.download_path, "Archives")
            if not os.path.exists(archives_path):
                os.makedirs(archives_path)
                print(f"✅ Created Archives folder")
                self.logger.info("STEP 1B: Created Archives folder")
            
            # Copy file to Archives with original name
            archive_dest = os.path.join(archives_path, os.path.basename(csv_filepath))
            shutil.copy2(csv_filepath, archive_dest)
            print(f"📚 Archived 52 Week High Low Report: {os.path.basename(archive_dest)}")
            self.logger.info(f"STEP 1B: Archived 52 Week High Low Report file: {os.path.basename(archive_dest)}")
            
        except Exception as e:
            print(f"❌ Error archiving 52 Week High Low file: {e}")
            self.logger.error(f"STEP 1B ERROR: Error archiving 52 Week High Low file: {e}")
    
    def cleanup_except_targets(self, csv_files_to_keep):
        """Clean up temporary files - PRESERVE ONLY CURRENT DAY CSV FILES"""
        try:
            print("\n🧹 Cleaning up temporary files...")
            
            # Files and folders to keep
            keep_items = {"Archives"}
            
            # Add target CSV filenames to keep (current day files)
            for csv_path in csv_files_to_keep:
                keep_items.add(os.path.basename(csv_path))
            
            # IMPORTANT: Keep current day 52 Week High Low file
            if self.extracted_files.get("highlow_file"):
                highlow_filename = os.path.basename(self.extracted_files["highlow_file"])
                keep_items.add(highlow_filename)
                print(f"🔒 Protecting current day 52 Week High Low file: {highlow_filename}")
            
            # REMOVED: The problematic block that protected ALL 52-week files
            # This was causing previous day CM_52_wk_High_low files to be preserved
            
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
            
            print(f"✅ Cleanup completed - deleted {deleted_count} files")
            self.logger.info(f"STEP 1: Cleaned up {deleted_count} temporary files")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            self.logger.error(f"STEP 1 ERROR: Error during cleanup: {e}")
    
    def save_extraction_info(self):
        """Save extracted file information to JSON for database script"""
        try:
            extraction_info = {
                "extracted_files": self.extracted_files,
                "parsed_date": str(self.parsed_date) if self.parsed_date else None,
                "download_path": self.download_path,
                "extraction_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            info_file = os.path.join(self.download_path, "extraction_info.json")
            with open(info_file, 'w') as f:
                json.dump(extraction_info, f, indent=2)
            
            print(f"💾 Saved extraction info to: {info_file}")
            self.logger.info(f"STEP 1: Saved extraction info to: {info_file}")
            
            return info_file
            
        except Exception as e:
            print(f"❌ Error saving extraction info: {e}")
            self.logger.error(f"STEP 1 ERROR: Error saving extraction info: {e}")
            return None
    
    def run_download_extract_pipeline(self, target_date=None):
        """Run the download and extraction pipeline only"""
        if target_date is None:
            target_date = self.get_current_date_string()
        
        print("=" * 90)
        print("                    NSE DOWNLOAD & EXTRACT PIPELINE")
        print("                    Download → Extract → Archive")
        print("=" * 90)
        print(f"📅 Processing data for date: {target_date}")
        
        pipeline_start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info(f"DOWNLOAD & EXTRACT PIPELINE START: Processing data for date: {target_date}")
        self.logger.info("=" * 80)
        
        # VALIDATION: Check if today is a trading day (not weekend or holiday)
        print("\n🔍 VALIDATING TRADING DAY")
        print("-" * 50)
        
        is_trading_day, reason = self.validate_trading_day()
        if not is_trading_day:
            print(f"❌ PIPELINE STOPPED: {reason}")
            self.logger.info(f"PIPELINE STOPPED: {reason}")
            return False
        
        print(f"✅ {reason} - Proceeding with download pipeline")
        
        # Step 1: Download and extract files
        print("\n🚀 STEP 1: DOWNLOAD & EXTRACT NSE FILES")
        print("-" * 50)
        step_name = "STEP 1 - DOWNLOAD & EXTRACT NSE FILES"
        self.update_batch_status(step_name, "Started")
        self.logger.info("STEP 1: Starting download and extraction process")
        
        try:
            zip_file = self.download_bhavcopy(target_date)
            if not zip_file:
                print("❌ PIPELINE FAILED: Could not download files")
                self.logger.error("PIPELINE FAILED: Could not download files")
                self.update_batch_status(step_name, "Failed")
                return False
            
            csv_files = self.extract_specific_files(zip_file)
            if not csv_files:
                print("❌ PIPELINE FAILED: Could not extract CSV files")
                self.logger.error("PIPELINE FAILED: Could not extract CSV files")
                self.update_batch_status(step_name, "Failed")
                return False
            
            print(f"✅ STEP 1 COMPLETED: Processed {len(csv_files)} files")
            for csv_file in csv_files:
                print(f"   • {os.path.basename(csv_file)}")
            self.logger.info(f"STEP 1 COMPLETED: Processed {len(csv_files)} files")
            self.update_batch_status(step_name, "Completed")
            
            # Save extraction information for database script
            info_file = self.save_extraction_info()
            if not info_file:
                print("⚠️ Could not save extraction info - database script may fail")
                self.logger.warning("Could not save extraction info - database script may fail")
            
        except Exception as e:
            print(f"❌ STEP 1 FAILED: {e}")
            self.logger.error(f"STEP 1 FAILED: {e}")
            self.update_batch_status(step_name, "Failed")
            return False
        
        # Pipeline completion
        pipeline_duration = time.time() - pipeline_start_time
        
        print("\n" + "=" * 90)
        print("🎉 DOWNLOAD & EXTRACT PIPELINE FINISHED SUCCESSFULLY! 🎉")
        print("=" * 90)
        print(f"⏱️ Total pipeline duration: {pipeline_duration:.2f} seconds")
        print(f"📁 Files location: {self.download_path}")
        print(f"📚 Archive location: {os.path.join(self.download_path, 'Archives')}")
        print(f"💾 Extraction info saved: extraction_info.json")
        print(f"📅 Process date: {self.parsed_date}")
        print("\n📋 COMPLETED OPERATIONS:")
        print("   ✅ Downloaded NSE files from website")
        print("   ✅ Extracted PD, MCAP, and 52 Week High Low CSV files")
        print("   ✅ Archived downloaded files")
        print("   ✅ Saved extraction info for database script")
        print("\n🔄 NEXT STEP:")
        print("   ➡️ Run nse_database_load.py to load data into database")
        print("=" * 90)
        
        self.logger.info("=" * 80)
        self.logger.info(f"DOWNLOAD & EXTRACT PIPELINE COMPLETED SUCCESSFULLY in {pipeline_duration:.2f} seconds")
        self.logger.info(f"Process date: {self.parsed_date}")
        self.logger.info("Files ready for database loading: PD, MCAP, 52 Week High Low")
        self.logger.info("Extraction info saved for database script consumption")
        self.logger.info("=" * 80)
        
        return True

def main():
    print("📋 NSE Download & Extract Script")
    print("=" * 50)
    print("Requirements:")
    print("   • pip install selenium requests psycopg2")
    print("   • ChromeDriver: https://chromedriver.chromium.org/")
    print("   • PostgreSQL database (for batch tracking)")
    print("\n📄 Log File Generated:")
    print("   • nse_download_extract.log")
    print("\n🔄 Next Step:")
    print("   • Run nse_database_load.py after this completes")
    print("")
    
    # Initialize download & extract pipeline
    pipeline = NSEDownloadExtract(
        download_path=r"C:\Python Code\Files",
        headless=False  # Set to True to hide browser
    )
    
    try:
        # Run download and extract pipeline only
        success = pipeline.run_download_extract_pipeline()
        
        if success:
            print("\n✅ Download and extraction completed successfully!")
            print("📊 Check nse_download_extract.log for detailed operation records")
            print("💾 extraction_info.json created for database script")
            print("🔄 Now run nse_database_load.py to complete the process")
        else:
            print("\n❌ Download and extraction pipeline failed.")
            print("📊 Check nse_download_extract.log for details")
            
    except Exception as e:
        print(f"\n💥 Unexpected pipeline error: {e}")
        pipeline.logger.error(f"UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()