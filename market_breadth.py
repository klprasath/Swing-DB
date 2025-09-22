import pandas as pd
import numpy as np
import warnings
import logging
import time
import sys
from datetime import datetime, timedelta
from db_connection import get_db_connection

warnings.filterwarnings("ignore")

# Global variables for base and previous trading days
BASE_TRADING_DAY = None
PREVIOUS_TRADING_DAY = None

# ================================================================================
# MARKET BREADTH LOGGING SYSTEM
# ================================================================================

class FileBeautifulFormatter(logging.Formatter):
    """Custom formatter that writes beautiful output to file (without timestamps)"""
    
    def format(self, record):
        # Return just the message for file - no timestamps
        return record.getMessage()

class MarketBreadthLogger:
    """Logging system for market breadth calculations"""
    
    def __init__(self, log_filename='market_breadth.log'):
        self.start_time = time.time()
        self.process_start_time = datetime.now()
        
        # Create logger
        self.logger = logging.getLogger('market_breadth')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler gets the beautiful formatted output (NO timestamps)
        # Use UTF-8 encoding to support emojis on Windows
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_formatter = FileBeautifulFormatter()
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        
        # Add only file handler - no console handler
        self.logger.addHandler(file_handler)
        
    def print_header(self):
        """Print the main header"""
        header = """
📊 MARKET BREADTH INDICATORS CALCULATION ENGINE
==================================================================================
📅 Process started: {start_time}
📊 Computing ADI, HLI, and SMA200I for market breadth analysis
==================================================================================
        """.format(start_time=self.process_start_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Beautiful output goes to file only
        self.logger.info(header)
    
    def print_phase_header(self, phase_number, phase_name, description=""):
        """Print phase header with formatting"""
        phase_header = f"""
📋 **PHASE {phase_number}: {phase_name}**
--------------------------------------------------"""
        
        if description:
            phase_header += f"\n📌 {description}"
        
        # Beautiful output to file only
        self.logger.info(phase_header)
    
    def print_step(self, step_name, details=""):
        """Print step information"""
        if details:
            file_msg = f"🔧 {step_name}: {details}"
        else:
            file_msg = f"🔧 {step_name}"
        
        self.logger.info(file_msg)
    
    def print_success(self, message, emoji="✅"):
        """Print success message"""
        self.logger.info(f"{emoji} {message}")
    
    def print_warning(self, message, emoji="⚠️"):
        """Print warning message (to file only)"""
        self.logger.warning(f"{emoji} {message}")
    
    def print_error(self, message, emoji="❌"):
        """Print error message"""
        self.logger.error(f"{emoji} {message}")
    
    def print_database_operation(self, operation, details="", duration=None):
        """Print database operation"""
        if duration:
            file_message = f"🗄️ {operation} completed in {duration:.2f} seconds"
            if details:
                file_message += f" - {details}"
        else:
            file_message = f"🗄️ {operation}"
            if details:
                file_message += f": {details}"
        
        self.logger.info(file_message)
    
    def print_final_summary(self, adi_value=None, hli_value=None, sma200i_value=None, trading_date=None):
        """Print final summary"""
        end_time = datetime.now()
        duration = time.time() - self.start_time
        
        # Beautiful summary for file
        file_summary = f"""
==================================================================================
🎉 **MARKET BREADTH CALCULATION COMPLETED SUCCESSFULLY** 🎉
==================================================================================
⏱️ **Processing Summary:**
   📅 Started: {self.process_start_time.strftime('%Y-%m-%d %H:%M:%S')}
   📅 Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
   ⏱️ Total Duration: {duration:.2f} seconds
   📅 Trading Date: {trading_date if trading_date else 'Current Date'}
   
📊 **Market Breadth Results:**
   📊 ADI (Advance Decline Index): {adi_value if adi_value is not None else 'Not calculated'}
   📈 HLI (High Low Index): {hli_value if hli_value is not None else 'Not calculated'}
   📉 SMA200I (SMA200 Index): {f'{sma200i_value:.2%}' if sma200i_value is not None else 'Not calculated'}
   
📋 **Completed Operations:**
   ✅ Advance Decline Index (ADI) calculation
   ✅ High Low Index (HLI) cumulative calculation  
   ✅ SMA200 Index (SMA200I) percentage calculation
   ✅ Market breadth data inserted/updated in database
==================================================================================
        """
        
        self.logger.info(file_summary)

# ================================================================================
# MARKET BREADTH CALCULATION FUNCTIONS
# ================================================================================

def get_latest_trading_date_from_stock_data(db):
    """Get the latest trading date from stock_data table"""
    try:
        # Get the latest date where we have stock data
        latest_date_query = """
            SELECT MAX(s.trd_dt) as latest_trading_date
            FROM stg_nse_data s
            INNER JOIN stock_data sd ON s.symbol = sd.symbol
            WHERE sd.mcap IS NOT NULL AND sd.mcap >= 300
        """
        
        db.execute(latest_date_query)
        result = db.fetchone()
        
        if result and result[0]:
            return result[0]
        else:
            return None
            
    except Exception as e:
        return None

def get_previous_trading_day_from_base(db, base_date):
    """Get the previous trading day from a given base date (excluding weekends and holidays)"""
    try:
        if isinstance(base_date, str):
            base_date = datetime.strptime(base_date, '%Y-%m-%d').date()
        elif isinstance(base_date, datetime):
            base_date = base_date.date()
        
        # Query to get the previous trading day from the base date
        previous_day_query = """
            SELECT MAX(trd_dt) as prev_trading_day
            FROM stg_nse_data s
            WHERE s.trd_dt < %s
            AND NOT EXISTS (
                SELECT 1 FROM stg_nse_holidays h 
                WHERE h.trd_dt = s.trd_dt
            )
            AND EXTRACT(DOW FROM s.trd_dt) NOT IN (0, 6)
        """
        
        db.execute(previous_day_query, (base_date,))
        result = db.fetchone()
        
        if result and result[0]:
            return result[0]
        else:
            return None
            
    except Exception as e:
        return None

def initialize_trading_days(db):
    """Initialize both base and previous trading day variables globally"""
    global BASE_TRADING_DAY, PREVIOUS_TRADING_DAY
    
    try:
        # Get the latest trading date from stock_data as base
        BASE_TRADING_DAY = get_latest_trading_date_from_stock_data(db)
        
        if not BASE_TRADING_DAY:
            return None, None
        
        # Get the previous trading day from that base date
        PREVIOUS_TRADING_DAY = get_previous_trading_day_from_base(db, BASE_TRADING_DAY)
        
        return BASE_TRADING_DAY, PREVIOUS_TRADING_DAY
            
    except Exception as e:
        BASE_TRADING_DAY = None
        PREVIOUS_TRADING_DAY = None
        return None, None

def calculate_advance_decline_index(db):
    """Calculate Advance Decline Index (ADI) using base trading day"""
    global BASE_TRADING_DAY
    
    try:
        if BASE_TRADING_DAY is None:
            return None
        
        # Calculate ADI: Ratio of net advancing/declining stocks over total moving stocks
        adi_query = """
            WITH stock_changes AS (
                SELECT 
                    sd.symbol,
                    CASE 
                        WHEN (sd.close_pr - sd.prev_cl_pr) > 0 THEN 1 
                        ELSE 0 
                    END as advancing,
                    CASE 
                        WHEN (sd.close_pr - sd.prev_cl_pr) < 0 THEN 1 
                        ELSE 0 
                    END as declining
                FROM stock_data sd
                INNER JOIN stg_nse_data s ON sd.symbol = s.symbol AND s.trd_dt = %s
                WHERE sd.mcap IS NOT NULL 
                AND sd.mcap >= 300
                AND sd.close_pr IS NOT NULL 
                AND sd.prev_cl_pr IS NOT NULL
            )
            SELECT 
                SUM(advancing) as advancing_count,
                SUM(declining) as declining_count,
                ROUND(CAST(SUM(advancing) - SUM(declining) AS DECIMAL)/CAST(SUM(advancing)+ SUM(declining) AS DECIMAL),2) as adi
            FROM stock_changes
        """
        
        db.execute(adi_query, (BASE_TRADING_DAY,))
        result = db.fetchone()
        
        if result:
            advancing_count, declining_count, adi_value = result
            return {
                'adi': adi_value,
                'advancing_count': advancing_count,
                'declining_count': declining_count
            }
        else:
            return None
            
    except Exception as e:
        return None

def calculate_high_low_index(db):
    """Calculate High Low Index (HLI) using pre-calculated 52-week high/low values"""
    global PREVIOUS_TRADING_DAY
    
    try:
        if PREVIOUS_TRADING_DAY is None:
            return None
        
        # Calculate number of stocks that closed at 52-week high
        highs_query = """
            SELECT COUNT(*) as highs_count
            FROM stock_data a
            WHERE EXISTS (
                SELECT 1 FROM stg_nse_data b 
                WHERE a.symbol = b.symbol 
                AND b.trd_dt = %s 
                AND a.close_pr > b.hi_52_wk
            )
            AND a.mcap IS NOT NULL 
            AND a.mcap >= 300
        """
        
        db.execute(highs_query, (PREVIOUS_TRADING_DAY,))
        highs_result = db.fetchone()
        highs_count = highs_result[0] if highs_result else 0
        
        # Calculate number of stocks that closed at 52-week low
        lows_query = """
            SELECT COUNT(*) as lows_count
            FROM stock_data a
            WHERE EXISTS (
                SELECT 1 FROM stg_nse_data b 
                WHERE a.symbol = b.symbol 
                AND b.trd_dt = %s 
                AND a.close_pr < b.lo_52_wk
            )
            AND a.mcap IS NOT NULL 
            AND a.mcap >= 300
        """
        
        db.execute(lows_query, (PREVIOUS_TRADING_DAY,))
        lows_result = db.fetchone()
        lows_count = lows_result[0] if lows_result else 0
        
        # Calculate net H-L for the day
        net_hl = highs_count - lows_count
        
        # Get previous day's HLI for cumulative calculation
        previous_hli = 0
        previous_day_of_previous = get_previous_trading_day_from_base(db, PREVIOUS_TRADING_DAY)
        
        if previous_day_of_previous:
            hli_query = "SELECT hli FROM market_breadth WHERE trd_dt = %s"
            db.execute(hli_query, (previous_day_of_previous,))
            prev_result = db.fetchone()
            if prev_result and prev_result[0] is not None:
                previous_hli = prev_result[0]
        
        # Calculate cumulative HLI
        current_hli = previous_hli + net_hl
        
        return {
            'hli': current_hli,
            'net_hl': net_hl,
            'highs_count': highs_count,
            'lows_count': lows_count,
            'previous_hli': previous_hli
        }
        
    except Exception as e:
        return None

def calculate_sma200_index(db):
    """Calculate SMA200 Index (SMA200I) using base trading day"""
    global BASE_TRADING_DAY
    
    try:
        if BASE_TRADING_DAY is None:
            return None
        
        # Calculate percentage of stocks above their 200-day SMA using base trading day data
        sma200_query = """
            WITH sma200_analysis AS (
                SELECT 
                    sd.symbol,
                    sd.close_pr,
                    sd.sma200,
                    CASE 
                        WHEN sd.close_pr > sd.sma200 THEN 1 
                        ELSE 0 
                    END as above_sma200
                FROM stock_data sd
                INNER JOIN stg_nse_data s ON sd.symbol = s.symbol AND s.trd_dt = %s
                WHERE sd.mcap IS NOT NULL 
                AND sd.mcap >= 300
                AND sd.close_pr IS NOT NULL 
                AND sd.sma200 IS NOT NULL
            )
            SELECT 
                COUNT(*) as total_stocks,
                SUM(above_sma200) as stocks_above_sma200,
                CASE 
                    WHEN COUNT(*) > 0 THEN (SUM(above_sma200)::FLOAT / COUNT(*)::FLOAT)
                    ELSE 0 
                END as sma200_index
            FROM sma200_analysis
        """
        
        db.execute(sma200_query, (BASE_TRADING_DAY,))
        result = db.fetchone()
        
        if result:
            total_stocks, stocks_above_sma200, sma200_index = result
            return {
                'sma200i': sma200_index,
                'total_stocks': total_stocks,
                'stocks_above_sma200': stocks_above_sma200
            }
        else:
            return None
            
    except Exception as e:
        return None

def insert_or_update_market_breadth(db, trading_date, adi_value=None, hli_value=None, sma200i_value=None):
    """Insert or update market breadth data"""
    try:
        if isinstance(trading_date, str):
            trading_date = datetime.strptime(trading_date, '%Y-%m-%d').date()
        elif isinstance(trading_date, datetime):
            trading_date = trading_date.date()
        
        # Check if record exists
        check_query = "SELECT COUNT(*) FROM market_breadth WHERE trd_dt = %s"
        db.execute(check_query, (trading_date,))
        exists = db.fetchone()[0] > 0
        
        if exists:
            # Update existing record
            update_parts = []
            params = []
            
            if adi_value is not None:
                update_parts.append("adi = %s")
                params.append(adi_value)
            if hli_value is not None:
                update_parts.append("hli = %s")
                params.append(hli_value)
            if sma200i_value is not None:
                update_parts.append("sma200i = %s")
                params.append(sma200i_value)
            
            if update_parts:
                update_parts.append("updated = CURRENT_TIMESTAMP")
                params.append(trading_date)
                
                update_query = f"""
                    UPDATE market_breadth 
                    SET {', '.join(update_parts)}
                    WHERE trd_dt = %s
                """
                db.execute(update_query, params)
                return "updated"
        else:
            # Insert new record
            insert_query = """
                INSERT INTO market_breadth (trd_dt, adi, hli, sma200i)
                VALUES (%s, %s, %s, %s)
            """
            db.execute(insert_query, (trading_date, adi_value, hli_value, sma200i_value))
            return "inserted"
        
        return "success"
        
    except Exception as e:
        return f"error: {e}"

# ================================================================================
# MAIN FUNCTION
# ================================================================================

def main():
    """Main function to calculate market breadth indicators"""
    
    try:
        # Initialize logger
        log = MarketBreadthLogger('market_breadth.log')
        
        # Print main header to file
        log.print_header()

        # Market breadth results
        market_breadth_results = {
            'adi_value': None,
            'hli_value': None,
            'sma200i_value': None
        }

        with get_db_connection() as db:
            
            # ================================================================================
            # PHASE 1: DATABASE CONNECTION & VALIDATION
            # ================================================================================
            log.print_phase_header(1, "DATABASE CONNECTION & VALIDATION")
            
            log.print_step("Validating database connection")
            # Test database connection
            db.execute("SELECT 1")
            log.print_success("Database connection established")
            
            log.print_step("Validating required tables and data")
            # Check if required tables exist
            required_tables = ['stock_data', 'stg_nse_data', 'stg_nse_holidays', 'market_breadth']
            for table in required_tables:
                db.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                log.print_success(f"Table '{table}' validated")
            
            # Check symbol count
            log.print_step("Checking eligible symbols for market breadth calculation")
            db.execute("SELECT COUNT(DISTINCT symbol) FROM stock_data WHERE mcap IS NOT NULL AND mcap >= 300")
            symbol_count = db.fetchone()[0]
            log.print_success(f"Found {symbol_count:,} eligible symbols (MCAP >= 300)")
            
            if symbol_count == 0:
                log.print_error("No eligible symbols found. Exiting.")
                return False

            # ================================================================================
            # PHASE 2: MARKET BREADTH CALCULATIONS
            # ================================================================================
            log.print_phase_header(2, "MARKET BREADTH INDICATORS CALCULATION",
                                 "Computing ADI, HLI, and SMA200I indicators")
            
            # Initialize both base and previous trading days globally
            log.print_step("Initializing base and previous trading days")
            base_date, previous_trading_day = initialize_trading_days(db)
            
            if not base_date or not previous_trading_day:
                log.print_error("Unable to determine base date or previous trading day. Exiting.")
                return False
            
            log.print_success(f"Base trading date: {base_date}")
            log.print_success(f"Previous trading date: {previous_trading_day}")
            
            try:
                # Calculate Advance Decline Index (ADI) using base date
                log.print_step(f"Calculating Advance Decline Index (ADI) for {base_date}")
                adi_result = calculate_advance_decline_index(db)
                
                if adi_result:
                    adi_value = adi_result['adi']
                    advancing_count = adi_result['advancing_count']
                    declining_count = adi_result['declining_count']
                    log.print_success(f"ADI calculated: {adi_value}")
                    log.print_success(f"Advancing stocks: {advancing_count}, Declining stocks: {declining_count}")
                    market_breadth_results['adi_value'] = adi_value
                else:
                    log.print_warning("ADI calculation failed - no valid stock data found")
                
                # Calculate High Low Index (HLI) using previous date
                log.print_step(f"Calculating High Low Index (HLI) using data from {previous_trading_day}")
                hli_result = calculate_high_low_index(db)
                
                if hli_result:
                    hli_value = hli_result['hli']
                    highs_count = hli_result['highs_count']
                    lows_count = hli_result['lows_count']
                    net_hl = hli_result['net_hl']
                    previous_hli = hli_result['previous_hli']
                    log.print_success(f"HLI calculated: {hli_value}")
                    log.print_success(f"52-week Highs: {highs_count}, 52-week Lows: {lows_count}")
                    log.print_success(f"Net H-L: {net_hl}, Previous HLI: {previous_hli}")
                    market_breadth_results['hli_value'] = hli_value
                else:
                    log.print_warning("HLI calculation failed - insufficient 52-week data or previous trading day not available")
                
                # Calculate SMA200 Index (SMA200I) using base date
                log.print_step(f"Calculating SMA200 Index (SMA200I) for {base_date}")
                sma200i_result = calculate_sma200_index(db)
                
                if sma200i_result:
                    sma200i_value = sma200i_result['sma200i']
                    total_stocks = sma200i_result['total_stocks']
                    stocks_above_sma200 = sma200i_result['stocks_above_sma200']
                    log.print_success(f"SMA200I calculated: {sma200i_value:.2%}")
                    log.print_success(f"Stocks above SMA200: {stocks_above_sma200}/{total_stocks}")
                    market_breadth_results['sma200i_value'] = sma200i_value
                else:
                    log.print_warning("SMA200I calculation failed - insufficient SMA200 data")
                
                # ================================================================================
                # PHASE 3: DATABASE UPDATE
                # ================================================================================
                log.print_phase_header(3, "DATABASE UPDATE & FINALIZATION")
                
                log.print_step("Inserting/updating market breadth data")
                insert_result = insert_or_update_market_breadth(
                    db, base_date,  # Use base_date instead of current_date
                    market_breadth_results['adi_value'], 
                    market_breadth_results['hli_value'], 
                    market_breadth_results['sma200i_value']
                )
                
                if insert_result == "inserted":
                    log.print_success("New market breadth record inserted into market_breadth table")
                    log.print_database_operation("Market breadth data inserted", f"Date: {base_date}")
                elif insert_result == "updated":
                    log.print_success("Market breadth record updated in market_breadth table")
                    log.print_database_operation("Market breadth data updated", f"Date: {base_date}")
                elif insert_result.startswith("error"):
                    log.print_error(f"Database operation failed: {insert_result}")
                    return False
                else:
                    log.print_warning(f"Database operation result: {insert_result}")
                
                db.commit()
                log.print_success("All market breadth data committed to database")
                
            except Exception as e:
                log.print_error(f"Error in Market Breadth calculations: {e}")
                return False

            # Final Summary
            log.print_final_summary(
                market_breadth_results['adi_value'],
                market_breadth_results['hli_value'],
                market_breadth_results['sma200i_value'],
                base_date  # Use base_date instead of current_date
            )

            # Simple console success message
            print("Market breadth calculation completed successfully!")
            return True

    except Exception as e:
        # Simple console error message
        print(f"Market breadth calculation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()