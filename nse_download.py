import os
import time
import zipfile
import shutil
import json
import psycopg2
from psycopg2 import Error
from datetime import datetime
import logging
import csv
from io import StringIO
import requests

class NSEDownloadExtract:
    def __init__(self, download_path=r"C:\Python Code\Files", headless=False):
        self.download_path = download_path
        self.headless = headless
        self.driver = None
        # UPDATED: Added bc_file to extracted_files dictionary
        self.extracted_files = {"pd_file": None, "mcap_file": None, "bc_file": None, "highlow_file": None}
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
    
    def download_bhavcopy(self, target_date=None):
        """Download PR zip and 52 Week High Low CSV directly using URLs for the specified date"""
        if target_date is None:
            target_date = self.get_current_date_string()
        
        try:
            # Compute date formats
            current_date = datetime.now()
            ddmmyy = current_date.strftime("%d%m%y")
            ddmmyyyy = current_date.strftime("%d%m%Y")
            
            # PR zip file
            pr_filename = f"PR{ddmmyy}.zip"
            pr_url = f"https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/{pr_filename}"
            print(f"⬇️ Downloading PR zip: {pr_url}")
            self.logger.info(f"STEP 1: Downloading PR zip from {pr_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/',
            }
            
            pr_path = self.download_file(pr_url, os.path.join(self.download_path, pr_filename), headers)
            if not pr_path:
                print(f"❌ Failed to download {pr_filename}")
                self.logger.error(f"STEP 1 ERROR: Failed to download {pr_filename}")
                return False
            
            # 52 Week High Low CSV
            highlow_filename = f"CM_52_wk_High_low_{ddmmyyyy}.csv"
            highlow_url = f"https://nsearchives.nseindia.com/content/{highlow_filename}"
            print(f"⬇️ Downloading 52 Week High Low CSV: {highlow_url}")
            self.logger.info(f"STEP 1: Downloading 52 Week High Low CSV from {highlow_url}")
            
            highlow_path = self.download_file(highlow_url, os.path.join(self.download_path, highlow_filename), headers)
            if highlow_path:
                self.extracted_files["highlow_file"] = highlow_path
                self.archive_52wk_file(highlow_path)
                print(f"✅ Downloaded and archived {highlow_filename}")
                self.logger.info(f"STEP 1: Downloaded and archived {highlow_filename}")
            else:
                print(f"⚠️ Failed to download {highlow_filename} - proceeding without it")
                self.logger.warning(f"STEP 1: Failed to download {highlow_filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during download: {e}")
            self.logger.error(f"STEP 1 ERROR: Error during download: {e}")
            return False
    
    def download_file(self, url, save_path, headers):
        """Helper to download file using requests"""
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {os.path.basename(save_path)} ({len(response.content):,} bytes)")
                return save_path
            else:
                print(f"❌ HTTP error {response.status_code} for {url}")
                return None
        except Exception as e:
            print(f"❌ Download error for {url}: {e}")
            return None
    
    def extract_specific_files(self, pr_zip_path):
        """Extract specific files from PR zip, handle archiving"""
        if not pr_zip_path or not os.path.exists(pr_zip_path):
            print("❌ PR ZIP file not found")
            self.logger.error("STEP 1 ERROR: PR ZIP file not found for extraction")
            return []
        
        try:
            # Create Archives folder
            archives_path = os.path.join(self.download_path, "Archives")
            if not os.path.exists(archives_path):
                os.makedirs(archives_path)
                print(f"✅ Created Archives folder")
                self.logger.info("STEP 1: Created Archives folder")
            
            print(f"\n📦 Processing PR ZIP: {os.path.basename(pr_zip_path)}")
            self.logger.info(f"STEP 1: Processing PR ZIP file: {os.path.basename(pr_zip_path)}")
            
            extracted_csvs = []
            
            pr_zip_name = os.path.basename(pr_zip_path)
            actual_date = pr_zip_name[2:-4]  # PR300925.zip -> 300925
            
            # Set parsed_date
            if actual_date:
                day = actual_date[:2]
                month = actual_date[2:4] 
                year_yy = actual_date[4:6]
                self.parsed_date = f"20{year_yy}-{month}-{day}"
                print(f"📅 Set tracking date: {self.parsed_date}")
                self.logger.info(f"STEP 1: Set tracking date: {self.parsed_date}")
            
            # Archive PR zip
            archive_dest = os.path.join(archives_path, pr_zip_name)
            shutil.copy2(pr_zip_path, archive_dest)
            print(f"📚 Archived: {os.path.basename(archive_dest)}")
            self.logger.info(f"STEP 1: Archived PR file: {os.path.basename(archive_dest)}")
            
            # UPDATED: Dynamically construct target filenames including BC file
            if actual_date:
                # Convert DDMMYY to DDMMYYYY for MCAP file
                day = actual_date[:2]
                month = actual_date[2:4] 
                year_yy = actual_date[4:6]
                year_yyyy = "20" + year_yy
                
                target_files = [
                    f"MCAP{day}{month}{year_yyyy}.csv",  # MCAP30092025.csv
                    f"Pd{actual_date}.csv",              # Pd300925.csv
                    f"Bc{actual_date}.csv"               # Bc300925.csv (NEW)
                ]
                print(f"🎯 Target files: {target_files}")
                self.logger.info(f"STEP 1: Target files for extraction: {target_files}")
            else:
                print("❌ Could not determine date from PR filename")
                self.logger.error("STEP 1 ERROR: Could not determine date from PR filename")
                return extracted_csvs  # Return what we have so far
            
            # Extract PD, MCAP, and BC files from PR zip
            with zipfile.ZipFile(pr_zip_path, 'r') as pr_zip:
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
                        
                        # UPDATED: Store file references for database loading including BC file
                        if target_file.startswith("Pd"):
                            self.extracted_files["pd_file"] = extracted_path
                        elif target_file.startswith("MCAP"):
                            self.extracted_files["mcap_file"] = extracted_path
                        elif target_file.startswith("Bc"):
                            self.extracted_files["bc_file"] = extracted_path
                        
                    else:
                        print(f"⚠️ File not found: {target_file}")
                        self.logger.warning(f"STEP 1: File not found: {target_file}")
                        # List similar files for debugging
                        similar_files = [f for f in pr_file_list if target_file[:4].lower() in f.lower()]
                        if similar_files:
                            print(f"   🔍 Similar files found: {similar_files}")
                            self.logger.info(f"STEP 1: Similar files found: {similar_files}")
            
            # Add highlow file if downloaded
            if self.extracted_files.get("highlow_file"):
                extracted_csvs.append(self.extracted_files["highlow_file"])
            
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
            
            # UPDATED: Also protect BC file
            if self.extracted_files.get("bc_file"):
                bc_filename = os.path.basename(self.extracted_files["bc_file"])
                keep_items.add(bc_filename)
                print(f"🔒 Protecting current day BC file: {bc_filename}")
            
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
            success = self.download_bhavcopy(target_date)
            if not success:
                print("❌ PIPELINE FAILED: Could not download files")
                self.logger.error("PIPELINE FAILED: Could not download files")
                self.update_batch_status(step_name, "Failed")
                return False
            
            # Get PR path from downloaded file (assuming PR zip is always downloaded successfully)
            pr_filename = f"PR{datetime.now().strftime('%d%m%y')}.zip"
            pr_path = os.path.join(self.download_path, pr_filename)
            
            csv_files = self.extract_specific_files(pr_path)
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
        print("   ✅ Extracted PD, MCAP, BC, and 52 Week High Low CSV files")
        print("   ✅ Archived downloaded files")
        print("   ✅ Saved extraction info for database script")
        print("\n📄 NEXT STEP:")
        print("   ➡️ Run nse_database_load.py to load data into database")
        print("=" * 90)
        
        self.logger.info("=" * 80)
        self.logger.info(f"DOWNLOAD & EXTRACT PIPELINE COMPLETED SUCCESSFULLY in {pipeline_duration:.2f} seconds")
        self.logger.info(f"Process date: {self.parsed_date}")
        self.logger.info("Files ready for database loading: PD, MCAP, BC, 52 Week High Low")
        self.logger.info("Extraction info saved for database script consumption")
        self.logger.info("=" * 80)
        
        return True

def main():
    print("📋 NSE Download & Extract Script")
    print("=" * 50)
    print("Requirements:")
    print("   • pip install requests psycopg2")
    print("   • PostgreSQL database (for batch tracking)")
    print("\n📄 Log File Generated:")
    print("   • nse_download_extract.log")
    print("\n📄 Next Step:")
    print("   • Run nse_database_load.py after this completes")
    print("")
    
    # Initialize download & extract pipeline
    pipeline = NSEDownloadExtract(
        download_path=r"C:\Python Code\Files",
        headless=False  # Ignored since no Selenium
    )
    
    try:
        # Run download and extract pipeline only
        success = pipeline.run_download_extract_pipeline()
        
        if success:
            print("\n✅ Download and extraction completed successfully!")
            print("📊 Check nse_download_extract.log for detailed operation records")
            print("💾 extraction_info.json created for database script")
            print("📄 Now run nse_database_load.py to complete the process")
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