import pandas as pd
import numpy as np
import talib
import warnings
import logging
import time
import sys
from datetime import datetime, timedelta
from db_connection import get_db_connection

warnings.filterwarnings("ignore")

# ================================================================================
# SIMPLIFIED LOGGING SYSTEM - BEAUTIFUL OUTPUT TO FILE ONLY
# ================================================================================

class FileBeautifulFormatter(logging.Formatter):
    """Custom formatter that writes beautiful output to file (without timestamps)"""
    
    def format(self, record):
        # Return just the message for file - no timestamps
        return record.getMessage()

class StockDataLogger:
    """Simplified logging system - detailed output to file, minimal to console"""
    
    def __init__(self, log_filename='stock_data.log'):
        self.start_time = time.time()
        self.process_start_time = datetime.now()
        
        # Create logger
        self.logger = logging.getLogger('stock_data')
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
        
        # Tracking variables
        self.current_phase = None
        self.batch_count = 0
        self.total_batches = 0
        
    def print_header(self):
        """Print the main header"""
        header = """
🚀 STOCK DATA PROCESSING & TECHNICAL ANALYSIS ENGINE
==================================================================================
📅 Process started: {start_time}
📊 Calculating technical indicators, momentum, persistence & relative strength
==================================================================================
        """.format(start_time=self.process_start_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Beautiful output goes to file only
        self.logger.info(header)
    
    def print_phase_header(self, phase_number, phase_name, description=""):
        """Print phase header with formatting"""
        self.current_phase = f"PHASE {phase_number}"
        
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
    
    def print_progress(self, message, emoji="⏳"):
        """Print progress message"""
        self.logger.info(f"{emoji} {message}")
    
    def print_success(self, message, emoji="✅"):
        """Print success message"""
        self.logger.info(f"{emoji} {message}")
    
    def print_warning(self, message, emoji="⚠️"):
        """Print warning message (to file only)"""
        self.logger.warning(f"{emoji} {message}")
    
    def print_error(self, message, emoji="❌"):
        """Print error message"""
        self.logger.error(f"{emoji} {message}")
    
    def print_batch_progress(self, current_batch, total_batches, processed_count, updated_count):
        """Print batch progress"""
        self.batch_count = current_batch
        self.total_batches = total_batches
        
        # Beautiful progress bar for file
        progress_bar = "█" * int(20 * current_batch / total_batches)
        remaining_bar = "░" * (20 - int(20 * current_batch / total_batches))
        percentage = (current_batch / total_batches) * 100
        
        file_message = f"📊 Batch {current_batch}/{total_batches} [{progress_bar}{remaining_bar}] {percentage:.1f}% - Processed: {processed_count}, Updated: {updated_count}"
        self.logger.info(file_message)
    
    def print_summary_section(self, section_name, stats_dict):
        """Print summary section"""
        # Beautiful summary for file
        file_output = f"\n📈 **{section_name.upper()} SUMMARY**\n"
        file_output += "--------------------------------------------------\n"
        
        for key, value in stats_dict.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                file_output += f"   {formatted_key}: {value:,}\n"
            else:
                file_output += f"   {formatted_key}: {value}\n"
        
        self.logger.info(file_output)
    
    def print_database_operation(self, operation, details="", duration=None):
        """Print database operation"""
        if duration:
            file_message = f"🗄️  {operation} completed in {duration:.2f} seconds"
            if details:
                file_message += f" - {details}"
        else:
            file_message = f"🗄️  {operation}"
            if details:
                file_message += f": {details}"
        
        self.logger.info(file_message)
    
    def print_final_summary(self, total_processed, total_updated, stock_rs_count=0, industry_rs_count=0):
        """Print final summary"""
        end_time = datetime.now()
        duration = time.time() - self.start_time
        
        # Beautiful summary for file
        file_summary = f"""
==================================================================================
🎉 **STOCK DATA PROCESSING COMPLETED SUCCESSFULLY** 🎉
==================================================================================
⏱️  **Processing Summary:**
   📅 Started: {self.process_start_time.strftime('%Y-%m-%d %H:%M:%S')}
   📅 Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
   ⏱️  Total Duration: {duration:.2f} seconds
   
📊 **Results:**
   📈 Total Symbols Processed: {total_processed:,}
   ✅ Total Symbols Updated: {total_updated:,}
   🔄 Stock RS Calculations: {stock_rs_count:,}
   🏭 Industry RS Calculations: {industry_rs_count:,}
   
📋 **Completed Operations:**
   ✅ Technical Indicators (EMA10, EMA20, EMA50, SMA200)
   ✅ Momentum Calculations (M5, M10, M20)  
   ✅ Persistence Calculations (P5, P10, P20, P60, P126, P252)
   ✅ Stock Grades (A+ to E rating system)
   ✅ Relative Strength Ratings (1-99 percentile)
   ✅ Sector-wise RS Rankings
   ✅ Stock & Industry Relative Strength (Mswing methodology)
==================================================================================
        """
        
        self.logger.info(file_summary)

# ================================================================================
# CORE CALCULATION FUNCTIONS
# ================================================================================

def f_momentum(src, length):
    """Calculate momentum using your exact algorithm from rs.txt"""
    if len(src) < 2:
        return np.nan
    
    available_len = min(len(src) - 1, length)
    if available_len < 1:
        return np.nan
    
    try:
        start_val = float(src.iloc[-available_len-1])
        end_val = float(src.iloc[-1])
        
        if pd.isna(start_val) or pd.isna(end_val) or start_val == 0:
            return np.nan
            
        momentum = (end_val - start_val) * 100 / start_val / available_len
        
        if pd.isna(momentum):
            return np.nan
            
        return float(momentum)
    except (IndexError, TypeError, ValueError, OverflowError):
        return np.nan

def _safe_extract_last_value(array):
    """Safely extract the last value from a TA-Lib result array"""
    try:
        if array is None or len(array) == 0:
            return None
        
        last_val = array[-1]
        
        if pd.isna(last_val) or (isinstance(last_val, (float, np.floating)) and np.isnan(last_val)):
            return None
        
        return float(last_val)
        
    except Exception:
        return None

def calculate_indicators_with_talib(close_prices, num_days):
    """Calculate indicators using TA-Lib with FIXED error handling"""
    try:
        talib.set_compatibility(1)
        
        if isinstance(close_prices, pd.Series):
            price_array = close_prices.values.astype(np.float64)
        else:
            price_array = np.array(close_prices, dtype=np.float64)
        
        ema10_value = ema20_value = ema50_value = sma200_value = None
        
        if len(price_array) < 10:
            return {'ema10': None, 'ema20': None, 'ema50': None, 'sma200': None}
        
        if num_days >= 200 and len(price_array) >= 200:
            try:
                ema10_array = talib.EMA(price_array, timeperiod=10)
                ema20_array = talib.EMA(price_array, timeperiod=20) 
                ema50_array = talib.EMA(price_array, timeperiod=50)
                sma200_array = talib.SMA(price_array, timeperiod=200)
                
                ema10_value = _safe_extract_last_value(ema10_array)
                ema20_value = _safe_extract_last_value(ema20_array)
                ema50_value = _safe_extract_last_value(ema50_array)
                sma200_value = _safe_extract_last_value(sma200_array)
                
            except Exception:
                pass
                
        elif num_days >= 50 and len(price_array) >= 50:
            try:
                ema10_array = talib.EMA(price_array, timeperiod=10)
                ema20_array = talib.EMA(price_array, timeperiod=20)
                ema50_array = talib.EMA(price_array, timeperiod=50)
                
                ema10_value = _safe_extract_last_value(ema10_array)
                ema20_value = _safe_extract_last_value(ema20_array)
                ema50_value = _safe_extract_last_value(ema50_array)
                
            except Exception:
                pass
                
        elif num_days >= 20 and len(price_array) >= 20:
            try:
                ema10_array = talib.EMA(price_array, timeperiod=10)
                ema20_array = talib.EMA(price_array, timeperiod=20)
                
                ema10_value = _safe_extract_last_value(ema10_array)
                ema20_value = _safe_extract_last_value(ema20_array)
                
            except Exception:
                pass
        
        return {
            'ema10': ema10_value,
            'ema20': ema20_value,
            'ema50': ema50_value,
            'sma200': sma200_value
        }
        
    except Exception:
        return {'ema10': None, 'ema20': None, 'ema50': None, 'sma200': None}

def calculate_momentum_trading_days_only(db, symbol, reference_date, period):
    """Calculate momentum using only actual trading days"""
    try:
        from datetime import datetime
        
        if isinstance(reference_date, str):
            reference_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
        elif isinstance(reference_date, datetime):
            reference_date = reference_date.date()
        
        momentum_query = """
            WITH recent_trading_days AS (
                SELECT trd_dt, close_pr,
                       ROW_NUMBER() OVER (ORDER BY trd_dt DESC) as rn
                FROM stg_nse_data s
                WHERE s.symbol = %s 
                AND s.trd_dt <= %s
                AND s.close_pr IS NOT NULL 
                AND s.close_pr > 0
                AND NOT EXISTS (
                    SELECT 1 FROM stg_nse_holidays h 
                    WHERE h.trd_dt = s.trd_dt
                )
                AND EXTRACT(DOW FROM s.trd_dt) NOT IN (0, 6)
            ),
            last_n_days AS (
                SELECT trd_dt, close_pr
                FROM recent_trading_days 
                WHERE rn <= %s
                ORDER BY trd_dt ASC
            )
            SELECT 
                AVG(close_pr) as avg_price,
                (SELECT close_pr FROM last_n_days ORDER BY trd_dt DESC LIMIT 1) as recent_close,
                COUNT(*) as days_count
            FROM last_n_days
        """
        
        db.execute(momentum_query, (symbol, reference_date, period))
        result = db.fetchone()
        
        if result and result[2] == period:
            avg_price, recent_close, days_count = result
            
            if avg_price and avg_price > 0 and recent_close and recent_close > 0:
                momentum = recent_close / avg_price
                return momentum
        
        return None
            
    except Exception:
        return None

def calculate_mswing(close, length1=20, length2=50, enable_ipo_mswing=True, min_data_points=20):
    """Calculate Mswing using your exact algorithm from rs.txt"""
    if len(close) < min_data_points:
        return np.nan
    
    adjusted_length1 = min(length1, len(close) - 1)
    adjusted_length2 = min(length2, len(close) - 1)
    
    if enable_ipo_mswing:
        momo_20 = f_momentum(close, adjusted_length1)
        momo_50 = f_momentum(close, adjusted_length2)
    else:
        momo_20 = (close.iloc[-1] - close.iloc[-adjusted_length1-1]) * 100 / close.iloc[-adjusted_length1-1] / adjusted_length1 if len(close) > adjusted_length1 else np.nan
        momo_50 = (close.iloc[-1] - close.iloc[-adjusted_length2-1]) * 100 / close.iloc[-adjusted_length2-1] / adjusted_length2 if len(close) > adjusted_length2 else np.nan
    
    if pd.isna(momo_20) or pd.isna(momo_50):
        return np.nan
    
    mswing = momo_20 + momo_50
    return round(mswing, 2)

def calculate_index_mswing(db, current_date, data_fetch_days=120):
    """Calculate Index Mswing using NIFTYMIDSML400 stocks from database"""
    try:
        start_date = current_date - timedelta(days=data_fetch_days)
        
        index_query = """
            SELECT s.symbol, s.trd_dt, s.close_pr
            FROM stg_nse_data s
            INNER JOIN stg_niftymidsml400_data n ON s.symbol = n.symbol
            WHERE s.trd_dt >= %s AND s.trd_dt <= %s
            ORDER BY s.symbol, s.trd_dt ASC
        """
        
        db.execute(index_query, (start_date, current_date))
        index_data = db.fetchall()
        
        if not index_data:
            return np.nan
        
        index_df = pd.DataFrame(index_data, columns=['symbol', 'trd_dt', 'close_pr'])
        index_df['close_pr'] = pd.to_numeric(index_df['close_pr'], errors='coerce')
        
        pivot_df = index_df.pivot(index='trd_dt', columns='symbol', values='close_pr')
        index_price = pivot_df.mean(axis=1, skipna=True)
        index_price = index_price.dropna()
        
        if len(index_price) < 20:
            return np.nan
        
        index_mswing = calculate_mswing(index_price, 20, 50, True, 20)
        
        if not pd.isna(index_mswing):
            return index_mswing
        else:
            return np.nan
            
    except Exception:
        return np.nan

def calculate_stock_relative_strength(db, current_date, data_fetch_days=120):
    """Calculate relative strength for all stocks using your exact methodology"""
    try:
        index_mswing = calculate_index_mswing(db, current_date, data_fetch_days)
        
        if pd.isna(index_mswing):
            return []
        
        start_date = current_date - timedelta(days=data_fetch_days)
        
        stock_query = """
            SELECT 
                sd.symbol,
                sd.name,
                COALESCE(i.sector, 'Unknown') as sector,
                COALESCE(i.industry_new, 'Unknown') as sub_sector,
                COALESCE(i.group_name, 'Unknown') as industry,
                COALESCE(i.group_sub_name, 'Unknown') as sub_industry
            FROM stock_data sd
            LEFT JOIN stg_ind_mapping i ON sd.symbol = i.symbol
            WHERE sd.mcap IS NOT NULL AND sd.mcap > 300
        """
        
        db.execute(stock_query)
        stock_info = db.fetchall()
        
        stock_rs_data = []
        
        for stock_row in stock_info:
            symbol, name, sector, sub_sector, industry, sub_industry = stock_row
            
            try:
                price_query = """
                    SELECT trd_dt, close_pr
                    FROM stg_nse_data
                    WHERE symbol = %s AND trd_dt >= %s AND trd_dt <= %s
                    ORDER BY trd_dt ASC
                """
                
                db.execute(price_query, (symbol, start_date, current_date))
                price_data = db.fetchall()
                
                if len(price_data) < 20:
                    continue
                
                close_prices = pd.Series([float(row[1]) for row in price_data], dtype=float)
                stock_mswing = calculate_mswing(close_prices, 20, 50, True, 20)
                
                if not pd.isna(stock_mswing):
                    stock_mswing_float = float(stock_mswing)
                    index_mswing_float = float(index_mswing)
                    relative_strength = round(stock_mswing_float - index_mswing_float, 2)
                    
                    stock_rs_data.append({
                        'symbol': symbol,
                        'name': name,
                        'sector': sector,
                        'sub_sector': sub_sector,
                        'industry': industry,
                        'sub_industry': sub_industry,
                        'relative_strength': relative_strength,
                        'mswing': stock_mswing
                    })
                    
            except Exception:
                continue
        
        return stock_rs_data
        
    except Exception:
        return []

def calculate_industry_relative_strength(stock_rs_data):
    """Calculate industry relative strength using your exact methodology"""
    try:
        if not stock_rs_data:
            return []
        
        df = pd.DataFrame(stock_rs_data)
        
        industry_stats = df.groupby('industry').agg({
            'relative_strength': ['mean', 'count'],
            'symbol': lambda x: (df.loc[x.index, 'relative_strength'] > 0).sum()
        }).round(2)
        
        industry_stats.columns = ['avg_rs', 'stock_count', 'positive_rs_count']
        industry_stats = industry_stats.reset_index()
        
        industry_stats['ratio'] = (industry_stats['positive_rs_count'] / industry_stats['stock_count']).round(2)
        industry_stats['rank'] = industry_stats['avg_rs'].rank(ascending=False, method='min').astype(int)
        industry_stats = industry_stats.sort_values('rank')
        
        industry_mapping = df.groupby('industry').first()[['sector', 'sub_sector', 'sub_industry']].to_dict('index')
        
        industry_rs_data = []
        for _, row in industry_stats.iterrows():
            industry = row['industry']
            industry_info = industry_mapping.get(industry, {})
            
            industry_rs_data.append({
                'sector': industry_info.get('sector', 'Unknown'),
                'sub_sector': industry_info.get('sub_sector', 'Unknown'),
                'industry': industry,
                'sub_industry': industry_info.get('sub_industry', 'Unknown'),
                'ind_rs': row['avg_rs'],
                'rank': row['rank'],
                'stock_cnt': row['stock_count'],
                'positive_rs_cnt': row['positive_rs_count'],
                'ratio': row['ratio']
            })
        
        return industry_rs_data
        
    except Exception:
        return []

def calculate_ibd_rs_score(stock_prices, market_prices, symbol="Unknown"):
    """Calculate IBD-style Relative Strength Score with adaptive periods"""
    min_days = 50
    if len(stock_prices) < min_days or len(market_prices) < min_days:
        return None
    
    try:
        available_days = len(stock_prices)
        periods = {}
        weights = {}
        
        if available_days >= 252:
            periods = {'3m': 63, '6m': 126, '9m': 189, '12m': 252}
            weights = {'3m': 0.4, '6m': 0.2, '9m': 0.2, '12m': 0.2}
        elif available_days >= 189:
            periods = {'3m': 63, '6m': 126, '9m': 189}
            weights = {'3m': 0.5, '6m': 0.25, '9m': 0.25}
        elif available_days >= 126:
            periods = {'3m': 63, '6m': 126}
            weights = {'3m': 0.6, '6m': 0.4}
        elif available_days >= 63:
            periods = {'3m': 63}
            weights = {'3m': 1.0}
        else:
            periods = {'available': min(available_days, 50)}
            weights = {'available': 1.0}
        
        stock_returns = {}
        market_returns = {}
        
        for period_name, days in periods.items():
            try:
                actual_days = min(days, available_days)
                
                stock_start = stock_prices.iloc[-actual_days]
                stock_end = stock_prices.iloc[-1]
                
                if stock_start == 0 or pd.isna(stock_start) or pd.isna(stock_end):
                    stock_returns[period_name] = None
                else:
                    stock_returns[period_name] = (stock_end / stock_start - 1) * 100
                
                market_start = market_prices.iloc[-actual_days]
                market_end = market_prices.iloc[-1]
                
                if market_start == 0 or pd.isna(market_start) or pd.isna(market_end):
                    market_returns[period_name] = None
                else:
                    market_returns[period_name] = (market_end / market_start - 1) * 100
                    
            except Exception:
                stock_returns[period_name] = None
                market_returns[period_name] = None
        
        relative_performance = {}
        for period in periods.keys():
            if (stock_returns[period] is not None and market_returns[period] is not None and 
                not pd.isna(stock_returns[period]) and not pd.isna(market_returns[period])):
                relative_performance[period] = stock_returns[period] - market_returns[period]
            else:
                relative_performance[period] = None
        
        weighted_score = 0
        total_weight = 0
        
        for period, weight in weights.items():
            if (relative_performance[period] is not None and 
                not pd.isna(relative_performance[period]) and 
                not np.isinf(relative_performance[period])):
                weighted_score += relative_performance[period] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            if pd.isna(final_score) or np.isinf(final_score):
                return None
            return final_score
        else:
            return None
            
    except Exception:
        return None

def convert_to_rs_rating(scores, min_rating=1, max_rating=99):
    """Convert RS scores to percentile ratings (1-99 scale)"""
    if not scores:
        return []
    
    try:
        valid_scores = [(i, score) for i, score in enumerate(scores) 
                       if (score is not None and not pd.isna(score) and not np.isinf(score))]
        
        if len(valid_scores) == 0:
            return [None] * len(scores)
        
        valid_scores.sort(key=lambda x: x[1])
        ratings = [None] * len(scores)
        
        if len(valid_scores) == 1:
            original_index, score = valid_scores[0]
            ratings[original_index] = 50
            return ratings
        
        for rank, (original_index, score) in enumerate(valid_scores):
            try:
                percentile = (rank / (len(valid_scores) - 1)) * (max_rating - min_rating) + min_rating
                rating = max(min_rating, min(max_rating, int(round(percentile))))
                ratings[original_index] = rating
            except Exception:
                ratings[original_index] = None
        
        return ratings
        
    except Exception:
        return [None] * len(scores)

def insert_stock_relative_strength(db, stock_rs_data, current_date):
    """Insert stock relative strength data into stg_relative_strength table"""
    try:
        if not stock_rs_data:
            return
        
        delete_query = "DELETE FROM stg_relative_strength WHERE trd_dt = %s"
        db.execute(delete_query, (current_date,))
        
        industry_rs_map = {}
        
        df = pd.DataFrame(stock_rs_data)
        if not df.empty:
            industry_avg_rs = df.groupby('industry')['relative_strength'].mean().round(2)
            industry_rs_map = industry_avg_rs.to_dict()
        
        insert_data = []
        for stock in stock_rs_data:
            industry_rs = industry_rs_map.get(stock['industry'], 0)
            insert_data.append((
                stock['symbol'],
                stock['name'],
                current_date,
                stock['sector'],
                stock['sub_sector'],
                stock['industry'],
                stock['sub_industry'],
                stock['relative_strength'],
                industry_rs
            ))
        
        insert_query = """
            INSERT INTO stg_relative_strength (
                symbol, name, trd_dt, sector, sub_sector, industry, sub_industry, rs, ind_rs
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        db.executemany(insert_query, insert_data)
        
    except Exception:
        pass

def insert_industry_relative_strength(db, industry_rs_data, current_date):
    """Insert industry relative strength data into stg_ind_relative_strength table"""
    try:
        if not industry_rs_data:
            return
        
        delete_query = "DELETE FROM stg_ind_relative_strength WHERE trd_dt = %s"
        db.execute(delete_query, (current_date,))
        
        insert_data = []
        for industry in industry_rs_data:
            insert_data.append((
                industry['sector'],
                industry['sub_sector'],
                industry['industry'],
                industry['sub_industry'],
                current_date,
                industry['ind_rs'],
                industry['rank'],
                industry['stock_cnt'],
                industry['positive_rs_cnt'],
                industry['ratio']
            ))
        
        insert_query = """
            INSERT INTO stg_ind_relative_strength (
                sector, sub_sector, industry, sub_industry, trd_dt, ind_rs, rank, stock_cnt, positive_rs_cnt, ratio
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        db.executemany(insert_query, insert_data)
        
    except Exception:
        pass

def calculate_stock_grade(current_price, ema10, ema20, ema50, sma200):
    """Calculate stock grade based on price position relative to moving averages"""
    try:
        if (current_price is None or pd.isna(current_price) or
            any(pd.isna(val) or val is None for val in [ema10, ema20, ema50, sma200])):
            return 'E'
        
        current_price = float(current_price)
        ema10 = float(ema10) if ema10 is not None else None
        ema20 = float(ema20) if ema20 is not None else None
        ema50 = float(ema50) if ema50 is not None else None
        sma200 = float(sma200) if sma200 is not None else None
        
        above_ema10 = ema10 is not None and current_price > ema10
        above_ema20 = ema20 is not None and current_price > ema20
        above_ema50 = ema50 is not None and current_price > ema50
        above_sma200 = sma200 is not None and current_price > sma200
        
        ema_trend_aligned = (ema10 is not None and ema20 is not None and 
                           ema50 is not None and sma200 is not None and
                           ema10 > ema20 > ema50 > sma200)
        
        ema20_trend_aligned = (ema20 is not None and ema50 is not None and 
                             sma200 is not None and ema20 > ema50 > sma200)
        
        if above_ema10 and above_ema20 and above_ema50 and above_sma200 and ema_trend_aligned:
            return 'A+'
        elif above_ema10 and above_ema20 and above_ema50 and above_sma200:
            return 'A'
        elif above_ema20 and above_ema50 and above_sma200 and not above_ema10 and ema20_trend_aligned:
            return 'B+'
        elif above_ema20 and above_ema50 and above_sma200 and not above_ema10:
            return 'B'
        elif above_ema50 and above_sma200 and not above_ema10 and not above_ema20:
            return 'C'
        elif above_sma200 and not above_ema10 and not above_ema20 and not above_ema50:
            return 'D'
        else:
            return 'E'
            
    except Exception:
        return 'E'

def calculate_persistence_fixed_logic(db, symbol, reference_date):
    """Calculate persistence by ensuring exactly N rows are processed for each period"""
    periods = [252, 126, 60, 20, 10, 5]
    result = {}
    
    try:
        from datetime import datetime, timedelta
        
        if isinstance(reference_date, str):
            reference_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
        elif isinstance(reference_date, datetime):
            reference_date = reference_date.date()
        
        for period in periods:
            try:
                corrected_query = """
                    WITH recent_trading_days AS (
                        SELECT trd_dt, close_pr,
                               ROW_NUMBER() OVER (ORDER BY trd_dt DESC) as rn
                        FROM stg_nse_data s
                        WHERE s.symbol = %s 
                        AND s.trd_dt <= %s
                        AND s.close_pr IS NOT NULL 
                        AND s.close_pr > 0
                        AND NOT EXISTS (
                            SELECT 1 FROM stg_nse_holidays h 
                            WHERE h.trd_dt = s.trd_dt
                        )
                        AND EXTRACT(DOW FROM s.trd_dt) NOT IN (0, 6)
                    ),
                    limited_recent_days AS (
                        SELECT trd_dt, close_pr
                        FROM recent_trading_days 
                        WHERE rn <= %s + 1
                        ORDER BY trd_dt ASC
                    ),
                    with_lag AS (
                        SELECT trd_dt, close_pr,
                               LAG(close_pr) OVER (ORDER BY trd_dt) as prev_close,
                               ROW_NUMBER() OVER (ORDER BY trd_dt) as day_num
                        FROM limited_recent_days
                    ),
                    final_period AS (
                        SELECT trd_dt, close_pr, prev_close,
                               CASE WHEN close_pr > prev_close THEN 1 ELSE 0 END as is_green
                        FROM with_lag
                        WHERE prev_close IS NOT NULL
                        ORDER BY trd_dt ASC
                        LIMIT %s
                    )
                    SELECT COUNT(*) FILTER (WHERE is_green = 1) as green_days,
                           COUNT(*) as total_days
                    FROM final_period
                """
                
                db.execute(corrected_query, (symbol, reference_date, period, period))
                query_result = db.fetchone()
                
                if query_result:
                    green_days, total_days = query_result
                    result[f'p{period}'] = green_days
                else:
                    result[f'p{period}'] = 0
                
            except Exception:
                result[f'p{period}'] = 0
        
        return result
        
    except Exception:
        for period in periods:
            result[f'p{period}'] = 0
        return result
def calculate_volume_metrics(db, symbol, reference_date):
    """
    Calculate ONLY:
    1. Average Weekly volume (52 weeks) 
    2. Average 50D SMA volume using TA-Lib
    3. Average 50D SMA price using TA-Lib (needed for R$Vol)
    
    NO 30D calculations - only 50D SMA using TA-Lib
    """
    try:
        from datetime import datetime, timedelta
        
        if isinstance(reference_date, str):
            reference_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
        elif isinstance(reference_date, datetime):
            reference_date = reference_date.date()
        
        result = {
            'avg_wk_vol': None,
            'avg_50d_sma_vol': None,
            'avg_50d_price': None
        }
        
        # Auto-detect volume column name
        volume_column = 'volume'
        try:
            volume_col_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'stg_nse_data' 
                AND column_name IN ('volume', 'vol_traded', 'traded_qty', 'vol', 'volume_traded', 'qty_traded', 'turnover_qty')
                LIMIT 1
            """
            db.execute(volume_col_query)
            volume_col_result = db.fetchone()
            if volume_col_result:
                volume_column = volume_col_result[0]
        except Exception:
            pass
        
        # Check if holidays table exists
        holidays_condition = ""
        try:
            holiday_check = """
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = 'stg_nse_holidays'
            """
            db.execute(holiday_check)
            if db.fetchone()[0] > 0:
                holidays_condition = """
                    AND NOT EXISTS (
                        SELECT 1 FROM stg_nse_holidays h 
                        WHERE h.trd_dt = s.trd_dt
                    )
                """
        except Exception:
            pass
        
        # ========================================
        # 1. Calculate Average Weekly Volume (52 weeks)
        # ========================================
        try:
            start_date_52w = reference_date - timedelta(weeks=52)
            
            weekly_volume_query = f"""
                WITH weekly_data AS (
                    SELECT 
                        DATE_TRUNC('week', trd_dt) as week_start,
                        SUM(CAST({volume_column} AS FLOAT)) as weekly_volume
                    FROM stg_nse_data s
                    WHERE s.symbol = %s 
                    AND s.trd_dt <= %s
                    AND s.trd_dt >= %s
                    AND s.{volume_column} IS NOT NULL 
                    AND CAST(s.{volume_column} AS FLOAT) > 0
                    {holidays_condition}
                    AND EXTRACT(DOW FROM s.trd_dt) NOT IN (0, 6)
                    GROUP BY DATE_TRUNC('week', trd_dt)
                    HAVING COUNT(*) > 0
                )
                SELECT 
                    AVG(weekly_volume) as avg_weekly_volume,
                    COUNT(*) as week_count
                FROM weekly_data
            """
            
            db.execute(weekly_volume_query, (symbol, reference_date, start_date_52w))
            weekly_result = db.fetchone()
            
            if weekly_result and weekly_result[0] and weekly_result[1] >= 10:
                result['avg_wk_vol'] = float(weekly_result[0])
                
        except Exception:
            result['avg_wk_vol'] = None
        
        # ========================================
        # 2. Calculate 50D SMA Volume and Price using TA-Lib ONLY
        # ========================================
        try:
            # Fetch 70 days to ensure we have enough data for 50D SMA
            daily_volume_query = f"""
                SELECT 
                    CAST(s.{volume_column} AS FLOAT) as volume,
                    CAST(s.close_pr AS FLOAT) as close_pr,
                    s.trd_dt
                FROM stg_nse_data s
                WHERE s.symbol = %s 
                AND s.trd_dt <= %s
                AND s.{volume_column} IS NOT NULL 
                AND s.close_pr IS NOT NULL
                AND CAST(s.{volume_column} AS FLOAT) > 0
                AND CAST(s.close_pr AS FLOAT) > 0
                {holidays_condition}
                AND EXTRACT(DOW FROM s.trd_dt) NOT IN (0, 6)
                ORDER BY s.trd_dt DESC
                LIMIT 70
            """
            
            db.execute(daily_volume_query, (symbol, reference_date))
            volume_data = db.fetchall()
            
            if len(volume_data) >= 50:
                # Reverse to get chronological order (oldest to newest)
                volume_data = list(reversed(volume_data))
                
                volumes = []
                prices = []
                
                for row in volume_data:
                    try:
                        vol = float(row[0])
                        price = float(row[1])
                        
                        if vol > 0 and price > 0:
                            volumes.append(vol)
                            prices.append(price)
                    except (ValueError, TypeError):
                        continue
                
                if len(volumes) >= 50:
                    # Convert to numpy arrays for TA-Lib
                    volume_array = np.array(volumes, dtype=np.float64)
                    price_array = np.array(prices, dtype=np.float64)
                    
                    try:
                        # Use TA-Lib to calculate 50-day SMA
                        talib.set_compatibility(1)
                        volume_sma50_array = talib.SMA(volume_array, timeperiod=50)
                        price_sma50_array = talib.SMA(price_array, timeperiod=50)
                        
                        # Extract the last (most recent) SMA values
                        vol_sma50_value = _safe_extract_last_value(volume_sma50_array)
                        price_sma50_value = _safe_extract_last_value(price_sma50_array)
                        
                        if vol_sma50_value is not None and price_sma50_value is not None:
                            result['avg_50d_sma_vol'] = float(vol_sma50_value)
                            result['avg_50d_price'] = float(price_sma50_value)
                            
                    except Exception:
                        # If TA-Lib fails, leave as None (will show blank in database)
                        result['avg_50d_sma_vol'] = None
                        result['avg_50d_price'] = None
                        
        except Exception:
            result['avg_50d_sma_vol'] = None
            result['avg_50d_price'] = None
        
        return result
        
    except Exception:
        return {
            'avg_wk_vol': None,
            'avg_50d_sma_vol': None,
            'avg_50d_price': None
        }


def calculate_relative_volumes(current_volume, current_price, avg_50d_sma_vol, avg_50d_price):
    """
    Calculate:
    1. RVol as percentage using avg_50d_sma_vol
    2. R$Vol using avg_50d_sma_vol and avg_50d_price
    
    Args:
        current_volume: Current trading volume from stock_data.volume
        current_price: Current price from stock_data.close_pr
        avg_50d_sma_vol: 50D SMA volume calculated using TA-Lib
        avg_50d_price: 50D SMA price calculated using TA-Lib
    
    Returns:
        dict: {
            'rvol': float or None (as percentage - e.g., 200% for 2x volume),
            'r_dollar_vol': float or None
        }
    """
    try:
        result = {
            'rvol': None,
            'r_dollar_vol': None
        }
        
        # ========================================
        # 1. Calculate RVol as Percentage
        # RVol = (Current Volume / avg_50d_sma_vol) × 100
        # Example: 2x volume = 200%
        # ========================================
        if (current_volume is not None and avg_50d_sma_vol is not None and 
            current_volume > 0 and avg_50d_sma_vol > 0 and 
            not pd.isna(current_volume) and not pd.isna(avg_50d_sma_vol)):
            
            try:
                current_volume = float(current_volume)
                avg_50d_sma_vol = float(avg_50d_sma_vol)
                
                rvol_ratio = current_volume / avg_50d_sma_vol
                rvol_percentage = rvol_ratio * 100
                
                if not pd.isna(rvol_percentage) and not np.isinf(rvol_percentage):
                    result['rvol'] = round(rvol_percentage, 1)  # e.g., 150.5%
                    
            except Exception:
                result['rvol'] = None
        
        # ========================================
        # 2. Calculate R$Vol (Relative Dollar Volume)
        # R$Vol = Current Dollar Volume / Average 50D Dollar Volume
        # ========================================
        if (current_volume is not None and current_price is not None and 
            avg_50d_sma_vol is not None and avg_50d_price is not None and
            all(val > 0 for val in [current_volume, current_price, avg_50d_sma_vol, avg_50d_price]) and
            not any(pd.isna(val) for val in [current_volume, current_price, avg_50d_sma_vol, avg_50d_price])):
            
            try:
                current_volume = float(current_volume)
                current_price = float(current_price)
                avg_50d_sma_vol = float(avg_50d_sma_vol)
                avg_50d_price = float(avg_50d_price)
                
                # Current Dollar Volume = Current volume × Current price
                current_dollar_volume = current_volume * current_price
                
                # Average 50D Dollar Volume = 50D SMA volume × 50D SMA price
                avg_50d_dollar_volume = avg_50d_sma_vol * avg_50d_price
                
                if avg_50d_dollar_volume > 0:
                    r_dollar_vol = current_dollar_volume / avg_50d_dollar_volume
                    
                    if not pd.isna(r_dollar_vol) and not np.isinf(r_dollar_vol):
                        result['r_dollar_vol'] = round(r_dollar_vol, 3)
                        
            except Exception:
                result['r_dollar_vol'] = None
        
        return result
        
    except Exception:
        return {
            'rvol': None,
            'r_dollar_vol': None
        }

def calculate_price_changes(db, symbol, current_price, reference_date):
    """
    Calculate 1 month and 3 month price change percentages
    
    Args:
        db: Database connection
        symbol: Stock symbol
        current_price: Current close price from stock_data table
        reference_date: Current date for calculation
    
    Returns:
        dict: {
            '1m_pr_chng': float or None,
            '3m_pr_chng': float or None
        }
    """
    try:
        from datetime import datetime, timedelta
        
        if isinstance(reference_date, str):
            reference_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
        elif isinstance(reference_date, datetime):
            reference_date = reference_date.date()
        
        result = {
            '1m_pr_chng': None,
            '3m_pr_chng': None
        }
        
        # Validate current price
        if current_price is None or pd.isna(current_price) or current_price <= 0:
            return result
        
        current_price = float(current_price)
        
        # ========================================
        # 1. Calculate 1 Month Price Change (30 calendar days)
        # ========================================
        try:
            one_month_ago = reference_date - timedelta(days=30)
            
            # Get the closest trading date to 30 days ago
            one_month_query = """
                SELECT close_pr, trd_dt
                FROM stg_nse_data
                WHERE symbol = %s 
                AND trd_dt <= %s
                AND close_pr IS NOT NULL
                AND close_pr > 0
                ORDER BY trd_dt DESC
                LIMIT 1
            """
            
            db.execute(one_month_query, (symbol, one_month_ago))
            one_month_result = db.fetchone()
            
            if one_month_result:
                old_price = float(one_month_result[0])
                old_date = one_month_result[1]
                
                if old_price > 0:
                    # Calculate percentage change: ((current - old) / old) * 100
                    price_change_1m = ((current_price - old_price) / old_price) * 100
                    
                    if not pd.isna(price_change_1m) and not np.isinf(price_change_1m):
                        result['1m_pr_chng'] = round(price_change_1m, 2)
                        
        except Exception:
            result['1m_pr_chng'] = None
        
        # ========================================
        # 2. Calculate 3 Month Price Change (90 calendar days)
        # ========================================
        try:
            three_months_ago = reference_date - timedelta(days=90)
            
            # Get the closest trading date to 90 days ago
            three_month_query = """
                SELECT close_pr, trd_dt
                FROM stg_nse_data
                WHERE symbol = %s 
                AND trd_dt <= %s
                AND close_pr IS NOT NULL
                AND close_pr > 0
                ORDER BY trd_dt DESC
                LIMIT 1
            """
            
            db.execute(three_month_query, (symbol, three_months_ago))
            three_month_result = db.fetchone()
            
            if three_month_result:
                old_price = float(three_month_result[0])
                old_date = three_month_result[1]
                
                if old_price > 0:
                    # Calculate percentage change: ((current - old) / old) * 100
                    price_change_3m = ((current_price - old_price) / old_price) * 100
                    
                    if not pd.isna(price_change_3m) and not np.isinf(price_change_3m):
                        result['3m_pr_chng'] = round(price_change_3m, 2)
                        
        except Exception:
            result['3m_pr_chng'] = None
        
        return result
        
    except Exception:
        return {
            '1m_pr_chng': None,
            '3m_pr_chng': None
        }        
# ================================================================================
# MAIN FUNCTION WITH SIMPLIFIED LOGGING
# ================================================================================

def main():
    """Main function to process stock data"""
    
    try:
        # Initialize simplified logger (beautiful output to file only)
        log = StockDataLogger('stock_data.log')
        
        # Print main header to file
        log.print_header()

        # Local summary stats
        summary_stats = {
            'ema_indicators': {'calculated': 0, 'failed': 0},
            'momentum': {'m5_calculated': 0, 'm10_calculated': 0, 'm20_calculated': 0, 'failed': 0},
            'persistence': {'calculated': 0, 'failed': 0},
            'grades': {'calculated': 0, 'failed': 0},
            'rs_ratings': {'calculated': 0, 'failed': 0},
            'volume_metrics': {'avg_wk_vol_calculated': 0, 'avg_50d_sma_vol_calculated': 0, 'rvol_calculated': 0, 'r_dollar_vol_calculated': 0, 'failed': 0},
            'price_changes': {'1m_calculated': 0, '3m_calculated': 0, 'failed': 0},  # ADD THIS LINE
            'total_processed': 0,
            'total_updated': 0
        }

        with get_db_connection() as db:
            
            # ================================================================================
            # PHASE 1: DATABASE SETUP & PREPARATION
            # ================================================================================
            log.print_phase_header(1, "DATABASE SETUP & PREPARATION")
            
            log.print_step("Creating required indexes")
            db.create_indexes()
            log.print_success("Database indexes created")
            
            log.print_step("Fetching symbols with MCAP > 300")
            db.execute("SELECT DISTINCT symbol FROM stock_data WHERE mcap IS NOT NULL AND mcap > 300")
            symbols = [row[0] for row in db.fetchall()]
            summary_stats['total_processed'] = len(symbols)
            
            if not symbols:
                log.print_error("No symbols found with MCAP > 300. Exiting.")
                return False
                
            log.print_success(f"Found {len(symbols):,} symbols for processing")

            log.print_step("Creating temporary tables for analysis")
            # Drop and create temporary tables
            db.execute("DROP TABLE IF EXISTS stock_data_temp")
            db.execute("DROP TABLE IF EXISTS market_data_temp")
            
            # Create stock data temp table
            create_temp_table_query = """
                CREATE TABLE stock_data_temp (
                    symbol VARCHAR(50),
                    close_pr FLOAT,
                    trd_dt DATE,
                    industry VARCHAR(200)
                )
            """
            db.execute(create_temp_table_query)
            
            # Create market data temp table  
            create_market_temp_query = """
                CREATE TABLE market_data_temp (
                    trd_dt DATE,
                    market_close FLOAT
                )
            """
            db.execute(create_market_temp_query)

            # Create indexes on temporary tables
            temp_indexes = [
                "CREATE INDEX idx_stock_data_temp_symbol_trd_dt ON stock_data_temp(symbol, trd_dt)",
                "CREATE INDEX idx_stock_data_temp_industry ON stock_data_temp(industry)",
                "CREATE INDEX idx_market_data_temp_date ON market_data_temp(trd_dt)"
            ]
            
            for index_query in temp_indexes:
                db.execute(index_query)
            
            log.print_success("Temporary tables and indexes created")

            # ================================================================================
            # PHASE 2: HISTORICAL DATA COLLECTION
            # ================================================================================
            log.print_phase_header(2, "HISTORICAL DATA COLLECTION", 
                                 "Fetching stock prices and market data for analysis")
            
            log.print_step("Fetching historical stock data (up to 252 trading days)")
            symbols_tuple = tuple(symbols)
            fetch_historical_query = """
                INSERT INTO stock_data_temp (symbol, close_pr, trd_dt, industry)
                SELECT s.symbol, s.close_pr, s.trd_dt, COALESCE(i.industry, 'Unknown') as industry
                FROM (
                    SELECT symbol, close_pr, trd_dt,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trd_dt DESC) AS rn
                    FROM stg_nse_data
                    WHERE symbol IN %s
                ) s
                LEFT JOIN stg_ind_mapping i ON s.symbol = i.symbol
                WHERE s.rn <= 252
            """
            db.execute(fetch_historical_query, (symbols_tuple,))
            log.print_success("Historical stock data with industry mapping collected")

            log.print_step("Fetching NIFTYMIDSML400 market data for relative analysis")
            market_data_query = """
                INSERT INTO market_data_temp (trd_dt, market_close)
                SELECT s.trd_dt, AVG(s.close_pr) as market_close
                FROM stg_nse_data s
                INNER JOIN stg_niftymidsml400_data n ON s.symbol = n.symbol
                WHERE s.trd_dt >= (
                    SELECT MIN(trd_dt) FROM stock_data_temp
                )
                GROUP BY s.trd_dt
                ORDER BY s.trd_dt
            """
            db.execute(market_data_query)
            log.print_success("Market data (NIFTYMIDSML400 average) collected")
            
            db.commit()

            # Create temporary table for results
            create_results_table_query = """
                CREATE TEMP TABLE temp_results (
                    symbol VARCHAR(50),
                    ema10 FLOAT,
                    ema20 FLOAT,
                    ema50 FLOAT,
                    sma200 FLOAT,
                    m5 FLOAT,
                    m10 FLOAT,
                    m20 FLOAT,
                    rs_score FLOAT,
                    rs_score_sector FLOAT,
                    industry VARCHAR(200),
                    grade VARCHAR(2),
                    p252 INTEGER,
                    p126 INTEGER,
                    p60 INTEGER,
                    p20 INTEGER,
                    p10 INTEGER,
                    p5 INTEGER,
                    avg_wk_vol FLOAT,
                    avg_50d_sma_vol FLOAT,
                    avg_50d_price FLOAT,
                    rvol FLOAT,
                    r_dollar_vol FLOAT,
                    one_m_pr_chng FLOAT,
                    three_m_pr_chng FLOAT
                )
            """


            # ================================================================================
            # PHASE 3: RS SCORE CALCULATIONS 
            # ================================================================================
            log.print_phase_header(3, "RELATIVE STRENGTH SCORE CALCULATIONS", 
                                 "Computing IBD-style RS ratings using market comparison")
            
            # Process symbols in batches
            batch_size = 100
            all_rs_scores = []
            all_sector_rs_scores = {}
            all_results = []

            log.print_step("Calculating RS scores for all symbols")
            total_batches_rs = (len(symbols) + batch_size - 1) // batch_size
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                current_batch_rs = (i // batch_size) + 1
                
                for symbol in batch_symbols:
                    try:
                        # Fetch stock data
                        stock_query = """
                            SELECT close_pr, trd_dt, industry
                            FROM stock_data_temp
                            WHERE symbol = %s
                            ORDER BY trd_dt ASC
                        """
                        db.execute(stock_query, (symbol,))
                        stock_data = db.fetchall()
                        
                        if not stock_data:
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': pd.Series([], dtype=float),
                                'rs_score': None,
                                'industry': 'Unknown'
                            })
                            summary_stats['rs_ratings']['failed'] += 1
                            continue
                            
                        # Extract data with validation
                        close_prices = []
                        dates = []
                        industry = stock_data[0][2] if stock_data else 'Unknown'
                        
                        for row in stock_data:
                            price = row[0]
                            date = row[1]
                            
                            # Validate price data
                            if price is None or pd.isna(price) or price <= 0:
                                continue
                                
                            close_prices.append(float(price))
                            dates.append(date)
                        
                        if len(close_prices) == 0:
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': pd.Series([], dtype=float),
                                'rs_score': None,
                                'industry': industry
                            })
                            summary_stats['rs_ratings']['failed'] += 1
                            continue
                        
                        # Convert to pandas Series
                        close_prices = pd.Series(close_prices, dtype=float)
                        
                        # Check if we have minimum data for RS calculation (50 days)
                        if len(close_prices) < 50:
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': close_prices,
                                'rs_score': None,
                                'industry': industry
                            })
                            summary_stats['rs_ratings']['failed'] += 1
                            continue
                        
                        # Fetch corresponding market data
                        if dates:
                            date_tuple = tuple(dates)
                            market_query = """
                                SELECT market_close
                                FROM market_data_temp
                                WHERE trd_dt IN %s
                                ORDER BY trd_dt ASC
                            """
                            db.execute(market_query, (date_tuple,))
                            market_data = db.fetchall()
                            
                            if market_data and len(market_data) == len(close_prices):
                                # Validate market data
                                market_prices = []
                                for row in market_data:
                                    price = row[0]
                                    if price is None or pd.isna(price) or price <= 0:
                                        market_prices.append(None)
                                    else:
                                        market_prices.append(float(price))
                                
                                # Check if we have enough valid market data
                                valid_market_data = [p for p in market_prices if p is not None]
                                if len(valid_market_data) < len(market_prices) * 0.8:  # At least 80% valid data
                                    rs_score = None
                                    summary_stats['rs_ratings']['failed'] += 1
                                else:
                                    market_prices = pd.Series(market_prices).fillna(method='ffill')  # Forward fill missing values
                                    
                                    # Calculate IBD RS Score
                                    rs_score = calculate_ibd_rs_score(close_prices, market_prices, symbol)
                                    if rs_score is not None:
                                        summary_stats['rs_ratings']['calculated'] += 1
                                    else:
                                        summary_stats['rs_ratings']['failed'] += 1
                                    
                                # Store for overall ranking
                                all_rs_scores.append(rs_score)
                                
                                # Store for sector ranking
                                if industry not in all_sector_rs_scores:
                                    all_sector_rs_scores[industry] = []
                                all_sector_rs_scores[industry].append(rs_score)
                                
                                # Store result
                                all_results.append({
                                    'symbol': symbol,
                                    'prices': close_prices,
                                    'rs_score': rs_score,
                                    'industry': industry
                                })
                            else:
                                all_rs_scores.append(None)
                                all_results.append({
                                    'symbol': symbol,
                                    'prices': close_prices,
                                    'rs_score': None,
                                    'industry': industry
                                })
                                summary_stats['rs_ratings']['failed'] += 1
                        else:
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': close_prices,
                                'rs_score': None,
                                'industry': industry
                            })
                            summary_stats['rs_ratings']['failed'] += 1
                            
                    except Exception:
                        all_rs_scores.append(None)
                        all_results.append({
                            'symbol': symbol,
                            'prices': pd.Series([], dtype=float),
                            'rs_score': None,
                            'industry': 'Unknown'
                        })
                        summary_stats['rs_ratings']['failed'] += 1

                # Update RS progress
                log.print_batch_progress(current_batch_rs, total_batches_rs, len(batch_symbols), len(batch_symbols))

            # Convert RS scores to ratings
            log.print_step("Converting RS scores to percentile ratings (1-99)")
            try:
                overall_rs_ratings = convert_to_rs_rating(all_rs_scores)
            except Exception:
                overall_rs_ratings = [None] * len(all_rs_scores)
            
            # Convert sector RS scores to ratings
            sector_rs_ratings = {}
            for industry, scores in all_sector_rs_scores.items():
                try:
                    sector_rs_ratings[industry] = convert_to_rs_rating(scores)
                except Exception:
                    sector_rs_ratings[industry] = [None] * len(scores)

            log.print_success("RS score calculations completed")

            # Track updated symbols for final reporting
            updated_symbols = set()

            # ================================================================================
            # PHASE 4: TECHNICAL INDICATORS & ANALYSIS
            # ================================================================================
            log.print_phase_header(4, "TECHNICAL INDICATORS, VOLUME METRICS, PRICE CHANGES & ANALYSIS",
                                 "Computing EMA, SMA, momentum, persistence, TA-Lib 50D SMA volume metrics, price changes, and grades")

            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                current_batch = (i // batch_size) + 1
                
                try:
                    # Create temp_results for batch
                    db.execute("DROP TABLE IF EXISTS temp_results")
                    db.execute(create_results_table_query)
                    
                    batch_results = []
                    batch_updated_symbols = []
                    
                    for j, symbol in enumerate(batch_symbols):
                        result_idx = i + j
                        if result_idx >= len(all_results):
                            continue
                        
                        # Individual stock error handling to prevent batch failure
                        try:
                            result = all_results[result_idx]
                            close_prices = result['prices']
                            industry = result['industry']
                            
                            # Ensure close_prices is ALWAYS a pandas Series
                            if not isinstance(close_prices, pd.Series):
                                if hasattr(close_prices, '__iter__') and not isinstance(close_prices, str):
                                    close_prices = pd.Series(list(close_prices), dtype=float)
                                else:
                                    close_prices = pd.Series([], dtype=float)
                            
                            # Double-check it's now a Series
                            if not isinstance(close_prices, pd.Series):
                                continue
                            
                            num_days = len(close_prices)
                            
                            # Skip if no price data
                            if num_days == 0:
                                continue
                            
                            # Initialize values
                            ema10_value = ema20_value = ema50_value = sma200_value = None
                            m5_value = m10_value = m20_value = None

                            # Calculate technical indicators based on available data
                            try:
                                indicators = calculate_indicators_with_talib(close_prices, num_days)
                                
                                ema10_value = indicators['ema10']
                                ema20_value = indicators['ema20']
                                ema50_value = indicators['ema50']
                                sma200_value = indicators['sma200']
                                
                                if any(v is not None for v in indicators.values()):
                                    summary_stats['ema_indicators']['calculated'] += 1
                                else:
                                    summary_stats['ema_indicators']['failed'] += 1
                                    
                            except Exception:
                                ema10_value = ema20_value = ema50_value = sma200_value = None
                                summary_stats['ema_indicators']['failed'] += 1

                            # Calculate momentum for 5, 10, and 20 days using trading days only
                            reference_date = None
                            if 'dates' in locals() and dates:
                                reference_date = dates[-1]
                            else:
                                # Get latest date for this symbol
                                db.execute("SELECT MAX(trd_dt) FROM stock_data_temp WHERE symbol = %s", (symbol,))
                                date_result = db.fetchone()
                                if date_result and date_result[0]:
                                    reference_date = date_result[0]

                            if reference_date:
                                try:
                                    # Calculate M5 (requires at least 5 trading days)
                                    if num_days >= 5:
                                        m5_value = calculate_momentum_trading_days_only(db, symbol, reference_date, 5)
                                        if m5_value is not None:
                                            summary_stats['momentum']['m5_calculated'] += 1
                                    
                                    # Calculate M10 (requires at least 10 trading days)
                                    if num_days >= 10:
                                        m10_value = calculate_momentum_trading_days_only(db, symbol, reference_date, 10)
                                        if m10_value is not None:
                                            summary_stats['momentum']['m10_calculated'] += 1
                                    
                                    # Calculate M20 (requires at least 20 trading days)
                                    if num_days >= 20:
                                        m20_value = calculate_momentum_trading_days_only(db, symbol, reference_date, 20)
                                        if m20_value is not None:
                                            summary_stats['momentum']['m20_calculated'] += 1
                                        
                                except Exception:
                                    m5_value = m10_value = m20_value = None
                                    summary_stats['momentum']['failed'] += 1
                            else:
                                summary_stats['momentum']['failed'] += 1
                            
                            # Get RS ratings
                            rs_rating = None
                            if result_idx < len(overall_rs_ratings):
                                rs_rating = overall_rs_ratings[result_idx]
                            
                            # Get sector RS rating
                            rs_rating_sector = None
                            if industry in sector_rs_ratings:
                                try:
                                    # Find this stock's position in the sector
                                    industry_scores = all_sector_rs_scores[industry]
                                    stock_rs_score = result['rs_score']
                                    if stock_rs_score is not None:
                                        # Find matching scores (handle potential duplicates)
                                        matching_indices = [idx for idx, score in enumerate(industry_scores) if score == stock_rs_score]
                                        if matching_indices:
                                            stock_position = matching_indices[0]  # Take first match
                                            if stock_position < len(sector_rs_ratings[industry]):
                                                rs_rating_sector = sector_rs_ratings[industry][stock_position]
                                except Exception:
                                    pass

                            # Calculate grade
                            current_price = close_prices.iloc[-1] if len(close_prices) > 0 else None
                            grade = calculate_stock_grade(current_price, ema10_value, ema20_value, ema50_value, sma200_value)
                            if grade != 'E':
                                summary_stats['grades']['calculated'] += 1
                            else:
                                summary_stats['grades']['failed'] += 1

                            # Calculate persistence using the working function
                            try:
                                if reference_date:
                                    persistence = calculate_persistence_fixed_logic(db, symbol, reference_date)
                                    if any(v > 0 for v in persistence.values()):
                                        summary_stats['persistence']['calculated'] += 1
                                    else:
                                        summary_stats['persistence']['failed'] += 1
                                else:
                                    persistence = {f'p{p}': 0 for p in [252, 126, 60, 20, 10, 5]}
                                    summary_stats['persistence']['failed'] += 1
                            except Exception:
                                persistence = {f'p{p}': 0 for p in [252, 126, 60, 20, 10, 5]}
                                summary_stats['persistence']['failed'] += 1
                            
                            # Calculate volume metrics
                            avg_wk_vol_value = avg_50d_sma_vol_value = avg_50d_price_value = None
                            rvol_value = r_dollar_vol_value = None
                            
                            try:
                                if reference_date:
                                    volume_metrics = calculate_volume_metrics(db, symbol, reference_date)
                                    avg_wk_vol_value = volume_metrics['avg_wk_vol']
                                    avg_50d_sma_vol_value = volume_metrics['avg_50d_sma_vol']
                                    avg_50d_price_value = volume_metrics['avg_50d_price']
                                    
                                    if avg_wk_vol_value is not None:
                                        summary_stats['volume_metrics']['avg_wk_vol_calculated'] += 1
                                    if avg_50d_sma_vol_value is not None:
                                        summary_stats['volume_metrics']['avg_50d_sma_vol_calculated'] += 1
                                    
                                    # Get current volume and price for relative volume calculations
                                    current_volume = None
                                    current_price = close_prices.iloc[-1] if len(close_prices) > 0 else None
                                    
                                    # Get current volume from database
                                    current_data_query = """
                                        SELECT volume, close_pr 
                                        FROM stock_data 
                                        WHERE symbol = %s
                                    """
                                    db.execute(current_data_query, (symbol,))
                                    current_data_result = db.fetchone()
                                    
                                    if current_data_result:
                                        current_volume = current_data_result[0]
                                        if current_data_result[1]:  # Use database price if available
                                            current_price = current_data_result[1]
                                    
                                    # Calculate relative volumes only if we have avg_50d_sma_vol
                                    if (current_volume is not None and current_price is not None and 
                                        avg_50d_sma_vol_value is not None and avg_50d_price_value is not None):
                                        
                                        relative_volumes = calculate_relative_volumes(
                                            current_volume, current_price, avg_50d_sma_vol_value, avg_50d_price_value
                                        )
                                        rvol_value = relative_volumes['rvol']
                                        r_dollar_vol_value = relative_volumes['r_dollar_vol']
                                        
                                        if rvol_value is not None:
                                            summary_stats['volume_metrics']['rvol_calculated'] += 1
                                        if r_dollar_vol_value is not None:
                                            summary_stats['volume_metrics']['r_dollar_vol_calculated'] += 1
                                    # If avg_50d_sma_vol is None, leave rvol and r_dollar_vol as None (blank)
                                else:
                                    summary_stats['volume_metrics']['failed'] += 1
                                    
                            except Exception:
                                summary_stats['volume_metrics']['failed'] += 1
                            
                            # Calculate price changes
                            one_m_pr_chng_value = three_m_pr_chng_value = None
                            
                            try:
                                if reference_date and current_price is not None:
                                    price_changes = calculate_price_changes(db, symbol, current_price, reference_date)
                                    one_m_pr_chng_value = price_changes['1m_pr_chng']
                                    three_m_pr_chng_value = price_changes['3m_pr_chng']
                                    
                                    if one_m_pr_chng_value is not None:
                                        summary_stats['price_changes']['1m_calculated'] += 1
                                    if three_m_pr_chng_value is not None:
                                        summary_stats['price_changes']['3m_calculated'] += 1
                                    
                                    if one_m_pr_chng_value is None and three_m_pr_chng_value is None:
                                        summary_stats['price_changes']['failed'] += 1
                                else:
                                    summary_stats['price_changes']['failed'] += 1
                                    
                            except Exception:
                                summary_stats['price_changes']['failed'] += 1
                            
                            batch_results.append((
                                symbol, ema10_value, ema20_value, ema50_value, sma200_value, 
                                m5_value, m10_value, m20_value, rs_rating, rs_rating_sector, industry, grade,
                                persistence['p252'], persistence['p126'], persistence['p60'],
                                persistence['p20'], persistence['p10'], persistence['p5'],
                                avg_wk_vol_value, avg_50d_sma_vol_value, avg_50d_price_value, rvol_value, r_dollar_vol_value,
                                one_m_pr_chng_value, three_m_pr_chng_value  # ADD THESE TWO
                            ))
                            
                            batch_updated_symbols.append(symbol)
                            
                        except Exception:
                            # Don't add to batch_results, continue to next stock
                            pass

                    # Insert batch results into temp_results
                    if batch_results:
                        try:
                            insert_results_query = """
                                INSERT INTO temp_results (
                                    symbol, ema10, ema20, ema50, sma200, m5, m10, m20, rs_score, rs_score_sector, 
                                    industry, grade, p252, p126, p60, p20, p10, p5,
                                    avg_wk_vol, avg_50d_sma_vol, avg_50d_price, rvol, r_dollar_vol,
                                    one_m_pr_chng, three_m_pr_chng
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            db.executemany(insert_results_query, batch_results)
                            
                            current_timestamp = datetime.now()
                            update_query = """
                                UPDATE stock_data
                                SET EMA10 = temp_results.ema10,
                                    EMA20 = temp_results.ema20,
                                    EMA50 = temp_results.ema50,
                                    SMA200 = temp_results.sma200,
                                    m5 = temp_results.m5,
                                    m10 = temp_results.m10,
                                    m20 = temp_results.m20,
                                    rs_rating = temp_results.rs_score,
                                    rs_rating_sector = temp_results.rs_score_sector,
                                    grade = temp_results.grade,
                                    p252 = temp_results.p252,
                                    p126 = temp_results.p126,
                                    p60 = temp_results.p60,
                                    p20 = temp_results.p20,
                                    p10 = temp_results.p10,
                                    p5 = temp_results.p5,
                                    avg_wk_vol = temp_results.avg_wk_vol,
                                    avg_50d_sma_vol = temp_results.avg_50d_sma_vol,
                                    rvol = temp_results.rvol,
                                    r_dollar_vol = temp_results.r_dollar_vol,
                                    one_m_pr_chng = temp_results.one_m_pr_chng,
                                    three_m_pr_chng = temp_results.three_m_pr_chng,
                                    last_updated = %s
                                FROM temp_results
                                WHERE stock_data.symbol = temp_results.symbol
                            """
                            db.execute(update_query, (current_timestamp,))
                            rows_updated = db.rowcount
                            db.commit()
                            
                            # Track updated symbols
                            updated_symbols.update(batch_updated_symbols)
                            summary_stats['total_updated'] += len(batch_updated_symbols)
                            
                        except Exception:
                            db.rollback()
                            continue

                except Exception:
                    db.rollback()
                    continue

                # Update progress for this batch
                log.print_batch_progress(current_batch, total_batches, 
                                       len(batch_symbols), len(batch_updated_symbols))

            # Print technical indicators summaries
            log.print_summary_section("Technical Indicators", {
                'ema_calculated': summary_stats['ema_indicators']['calculated'],
                'ema_failed': summary_stats['ema_indicators']['failed'],
                'success_rate': f"{(summary_stats['ema_indicators']['calculated'] / max(summary_stats['ema_indicators']['calculated'] + summary_stats['ema_indicators']['failed'], 1) * 100):.1f}%"
            })
            
            log.print_summary_section("Momentum Analysis", {
                'm5_calculated': summary_stats['momentum']['m5_calculated'],
                'm10_calculated': summary_stats['momentum']['m10_calculated'], 
                'm20_calculated': summary_stats['momentum']['m20_calculated'],
                'total_failed': summary_stats['momentum']['failed']
            })

            log.print_summary_section("Persistence & Grades", {
                'persistence_calculated': summary_stats['persistence']['calculated'],
                'persistence_failed': summary_stats['persistence']['failed'],
                'grades_calculated': summary_stats['grades']['calculated'],
                'grades_failed': summary_stats['grades']['failed']
            })

            log.print_summary_section("RS Ratings", {
                'rs_calculated': summary_stats['rs_ratings']['calculated'],
                'rs_failed': summary_stats['rs_ratings']['failed'],
                'total_symbols': len(symbols)
            })

            log.print_summary_section("Volume Metrics", {
                'avg_wk_vol_calculated': summary_stats['volume_metrics']['avg_wk_vol_calculated'],
                'avg_50d_sma_vol_calculated': summary_stats['volume_metrics']['avg_50d_sma_vol_calculated'],
                'rvol_calculated_as_percentage': summary_stats['volume_metrics']['rvol_calculated'],
                'r_dollar_vol_calculated': summary_stats['volume_metrics']['r_dollar_vol_calculated'],
                'volume_failed': summary_stats['volume_metrics']['failed'],
                'success_rate': f"{((summary_stats['volume_metrics']['avg_wk_vol_calculated'] + summary_stats['volume_metrics']['avg_50d_sma_vol_calculated']) / max((summary_stats['volume_metrics']['avg_wk_vol_calculated'] + summary_stats['volume_metrics']['avg_50d_sma_vol_calculated'] + summary_stats['volume_metrics']['failed']), 1) * 100):.1f}%"
            })
            
            log.print_summary_section("Price Changes", {
                '1m_price_changes_calculated': summary_stats['price_changes']['1m_calculated'],
                '3m_price_changes_calculated': summary_stats['price_changes']['3m_calculated'],
                'price_changes_failed': summary_stats['price_changes']['failed'],
                'success_rate': f"{((summary_stats['price_changes']['1m_calculated'] + summary_stats['price_changes']['3m_calculated']) / max((summary_stats['price_changes']['1m_calculated'] + summary_stats['price_changes']['3m_calculated'] + summary_stats['price_changes']['failed']), 1) * 100):.1f}%"
            })
            
            # ================================================================================
            # PHASE 5: RELATIVE STRENGTH ANALYSIS
            # ================================================================================
            log.print_phase_header(5, "RELATIVE STRENGTH ANALYSIS",
                                 "Computing stock and industry relative strength using Mswing methodology")
            
            current_date = datetime.now().date()
            
            try:
                log.print_step("Calculating stock relative strength using Mswing methodology")
                stock_rs_data = calculate_stock_relative_strength(db, current_date, data_fetch_days=120)
                
                # Insert stock relative strength data
                if stock_rs_data:
                    log.print_success(f"Stock relative strength calculated for {len(stock_rs_data)} stocks")
                    insert_stock_relative_strength(db, stock_rs_data, current_date)
                    db.commit()
                    log.print_database_operation("Stock RS data inserted into stg_relative_strength")
                else:
                    log.print_warning("No stock relative strength data calculated")
                
                log.print_step("Calculating industry relative strength rankings")
                industry_rs_data = calculate_industry_relative_strength(stock_rs_data)
                
                # Insert industry relative strength data
                if industry_rs_data:
                    log.print_success(f"Industry relative strength calculated for {len(industry_rs_data)} industries")
                    insert_industry_relative_strength(db, industry_rs_data, current_date)
                    db.commit()
                    log.print_database_operation("Industry RS data inserted into stg_ind_relative_strength")
                else:
                    log.print_warning("No industry relative strength data calculated")
                
            except Exception as e:
                log.print_error(f"Error in Relative Strength calculations: {e}")
                stock_rs_data = []
                industry_rs_data = []

            # ================================================================================
            # PHASE 6: CLEANUP & FINALIZATION
            # ================================================================================
            log.print_phase_header(6, "CLEANUP & FINALIZATION")
            
            log.print_step("Dropping temporary tables")
            try:
                db.execute("DROP TABLE IF EXISTS stock_data_temp")
                db.execute("DROP TABLE IF EXISTS market_data_temp") 
                db.execute("DROP TABLE IF EXISTS temp_results")
                log.print_success("Temporary tables cleaned up")
            except Exception:
                log.print_warning("Some temporary tables could not be dropped")


            # ================================================================================
            # PHASE 7: HISTORICAL DATA & ATH UPDATES
            # ================================================================================
            log.print_phase_header(7, "HISTORICAL DATA & ALL-TIME HIGH UPDATES",
                                 "Updating stock_data_hist and stg_ath tables with latest data")
            
            # Initialize tracking variables for Phase 7
            hist_inserted_count = 0
            ath_updated_count = 0
            
            try:
                # ========================================
                # 1. INSERT into stock_data_hist table
                # ========================================
                log.print_step("Checking and updating stock_data_hist table")
                
                # Get the distinct trd_dt from stock_data table
                get_trd_dt_query = "SELECT DISTINCT trd_dt FROM stock_data WHERE trd_dt IS NOT NULL"
                db.execute(get_trd_dt_query)
                trd_dt_result = db.fetchone()
                
                if trd_dt_result and trd_dt_result[0]:
                    current_trd_dt = trd_dt_result[0]
                    log.print_progress(f"Found trd_dt in stock_data: {current_trd_dt}")
                    
                    # Check if records exist in stock_data_hist for this trd_dt
                    check_hist_query = """
                        SELECT COUNT(*) FROM stock_data_hist 
                        WHERE trd_dt = %s
                    """
                    db.execute(check_hist_query, (current_trd_dt,))
                    hist_count_result = db.fetchone()
                    
                    if hist_count_result and hist_count_result[0] == 0:
                        # No records exist for this trd_dt, so insert all records from stock_data
                        log.print_step("No existing records found, inserting data into stock_data_hist")
                        
                        insert_hist_query = """
                            INSERT INTO stock_data_hist
                            SELECT * FROM stock_data
                        """
                        
                        db.execute(insert_hist_query)
                        hist_inserted_count = db.rowcount
                        db.commit()
                        
                        log.print_success(f"Inserted {hist_inserted_count:,} records into stock_data_hist")
                        log.print_database_operation("Historical data insertion", f"{hist_inserted_count:,} records inserted")
                        
                    else:
                        log.print_progress("Records already exist in stock_data_hist for current trd_dt - no insertion needed")
                        log.print_database_operation("Historical data check", "No insertion required")
                        
                else:
                    log.print_warning("No trd_dt found in stock_data table")
                    
            except Exception as e:
                log.print_error(f"Error updating stock_data_hist: {e}")
                db.rollback()
            
            try:
                # ========================================
                # 2. UPDATE stg_ath table
                # ========================================
                log.print_step("Updating all-time high records in stg_ath table")
                
                # Update stg_ath where current close_pr is higher than existing ath_pr
                update_ath_query = """
                    UPDATE stg_ath a
                    SET ath_pr = (
                            SELECT close_pr 
                            FROM stock_data d 
                            WHERE d.symbol = a.symbol
                        ),
                        ath_dt = (
                            SELECT DISTINCT trd_dt 
                            FROM stock_data d 
                            WHERE d.symbol = a.symbol
                        ),
                        updated_at = CURRENT_DATE
                    WHERE EXISTS (
                        SELECT 1 
                        FROM stock_data d 
                        WHERE d.symbol = a.symbol 
                        AND a.ath_pr < d.close_pr
                    )
                """
                
                db.execute(update_ath_query)
                ath_updated_count = db.rowcount
                db.commit()
                
                if ath_updated_count > 0:
                    log.print_success(f"Updated {ath_updated_count:,} all-time high records in stg_ath")
                    log.print_database_operation("ATH records update", f"{ath_updated_count:,} records updated")
                else:
                    log.print_progress("No new all-time highs found - no ATH records updated")
                    log.print_database_operation("ATH records check", "No updates required")
                    
            except Exception as e:
                log.print_error(f"Error updating stg_ath: {e}")
                db.rollback()
            
            # Initialize tracking variables for Highest Volume Updates
            hvy_updated_count = 0
            hve_updated_count = 0
            
            try:
                # ========================================
                # 3. UPDATE stg_highest_volume table - HVY (Highest Volume Yearly)
                # ========================================
                log.print_step("Updating highest volume records (HVY) in stg_highest_volume table")
                
                # First HVY update query
                hvy_update_query = """
                    WITH SymbolsToUpdate AS (
                        select d.symbol,round(d.volume/1000000,2) volume,d.trd_dt from stg_highest_volume a, stock_data d
                        where round(d.volume/1000000,2) >= a.volume
                        and d.symbol = a.symbol
                        and mcap > 300
                        except (select d.symbol,round(d.volume/1000000,2) volume,d.trd_dt from stg_highest_volume a, stock_data d
                        where round(d.volume/1000000,2) >= a.volume
                        and d.symbol = a.symbol
                        and mcap > 300
                        and type = 'HVE')
                    )
                    UPDATE stg_highest_volume a
                    SET
                        volume = s.volume,
                        type = 'HVY',
                        trd_dt = s.trd_dt
                    FROM
                        SymbolsToUpdate s
                    WHERE
                        a.symbol = s.symbol
                        and type = 'HVY'
                """
                
                db.execute(hvy_update_query)
                hvy_updated_count = db.rowcount
                db.commit()
                
                if hvy_updated_count > 0:
                    log.print_success(f"Updated {hvy_updated_count:,} HVY records in stg_highest_volume")
                    log.print_database_operation("HVY records update", f"{hvy_updated_count:,} records updated")
                else:
                    log.print_progress("No new highest volumes found for HVY - no HVY records updated")
                    log.print_database_operation("HVY records check", "No updates required")
                    
            except Exception as e:
                log.print_error(f"Error updating HVY records in stg_highest_volume: {e}")
                db.rollback()
            
            try:
                # ========================================
                # 4. UPDATE stg_highest_volume table - HVE (Highest Volume Ever)
                # ========================================
                log.print_step("Updating highest volume records (HVE) in stg_highest_volume table")
                
                # Second HVE update query
                hve_update_query = """
                    WITH SymbolsToUpdate AS (
                        select d.symbol,round(d.volume/1000000,2) volume,d.trd_dt from stg_highest_volume a, stock_data d
                        where round(d.volume/1000000,2) >= a.volume
                        and d.symbol = a.symbol
                        and mcap > 300
                        and type = 'HVE'
                    )
                    UPDATE stg_highest_volume a
                    SET
                        volume = s.volume,
                        type = 'HVE',
                        trd_dt = s.trd_dt
                    FROM
                        SymbolsToUpdate s
                    WHERE
                        a.symbol = s.symbol
                        and type = 'HVE'
                """
                
                db.execute(hve_update_query)
                hve_updated_count = db.rowcount
                db.commit()
                
                if hve_updated_count > 0:
                    log.print_success(f"Updated {hve_updated_count:,} HVE records in stg_highest_volume")
                    log.print_database_operation("HVE records update", f"{hve_updated_count:,} records updated")
                else:
                    log.print_progress("No new highest volumes found for HVE - no HVE records updated")
                    log.print_database_operation("HVE records check", "No updates required")
                    
            except Exception as e:
                log.print_error(f"Error updating HVE records in stg_highest_volume: {e}")
                db.rollback()

            # Print Phase 7 summary
            log.print_summary_section("Historical Data & ATH Updates", {
                'STOCK_DATA_HIST Records Inserted': hist_inserted_count,
                'ATH Records Updated': ath_updated_count,
                'HVY Records Updated': hvy_updated_count,
                'HVE Records Updated': hve_updated_count,
                'STOCK_DATA_HIST Insert Status': 'Completed' if hist_inserted_count >= 0 else 'Failed',
                'ATH UPDATE Status': 'Completed' if ath_updated_count >= 0 else 'Failed',
                'HVY UPDATE Status': 'Completed' if hvy_updated_count >= 0 else 'Failed',
                'HVE UPDATE Status': 'Completed' if hve_updated_count >= 0 else 'Failed',
            })    

            # Final Summary
            log.print_final_summary(
                summary_stats['total_processed'],
                summary_stats['total_updated'],
                len(stock_rs_data) if stock_rs_data else 0,
                len(industry_rs_data) if industry_rs_data else 0
            )

            # Simple console success message
            print("Script completed successfully!")
            return True

    except Exception as e:
        # Simple console error message
        print(f"Script failed: {e}")
        return False

if __name__ == "__main__":
    success = main()