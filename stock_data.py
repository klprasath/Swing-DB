import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator
import warnings
import logging
import time
from datetime import datetime, timedelta
from db_connection import get_db_connection

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='stock_data.log',
    filemode='a'  # Append to log file
)
warnings.filterwarnings("ignore")  # Suppress pandas warnings

def f_momentum(src, length):
    """
    Calculate momentum using your exact algorithm from rs.txt
    """
    if len(src) < 2:
        logging.warning(f"No valid data points for momentum calculation with length {length}")
        return np.nan
    
    available_len = min(len(src) - 1, length)
    if available_len < 1:
        logging.warning(f"Insufficient data points ({len(src)}) for momentum calculation with length {length}")
        return np.nan
    
    try:
        start_val = float(src.iloc[-available_len-1])
        end_val = float(src.iloc[-1])
        
        # Validate that we have numeric values
        if pd.isna(start_val) or pd.isna(end_val) or start_val == 0:
            logging.warning(f"Invalid values for momentum calculation: start={start_val}, end={end_val}")
            return np.nan
            
        momentum = (end_val - start_val) * 100 / start_val / available_len
        
        if pd.isna(momentum):
            logging.warning(f"Invalid momentum calculation result: {momentum}")
            return np.nan
            
        return float(momentum)
    except (IndexError, TypeError, ValueError, OverflowError) as e:
        logging.error(f"Error in f_momentum: {e}, available_len={available_len}, src_len={len(src)}")
        return np.nan

def calculate_mswing(close, length1=20, length2=50, enable_ipo_mswing=True, min_data_points=20):
    """
    Calculate Mswing using your exact algorithm from rs.txt
    """
    if len(close) < min_data_points:
        logging.warning(f"Insufficient data points ({len(close)}) for Mswing calculation, need at least {min_data_points}")
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
        logging.warning(f"Invalid momentum values: momo_20={momo_20}, momo_50={momo_50}")
        return np.nan
    
    mswing = momo_20 + momo_50
    return round(mswing, 2)

def calculate_index_mswing(db, current_date, data_fetch_days=120):
    """
    Calculate Index Mswing using NIFTYMIDSML400 stocks from database
    """
    try:
        # Get NIFTY MID SMALL 400 stocks data from database
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
            logging.error("No index data found for Mswing calculation")
            return np.nan
        
        # Convert to DataFrame with proper type conversion
        index_df = pd.DataFrame(index_data, columns=['symbol', 'trd_dt', 'close_pr'])
        index_df['close_pr'] = pd.to_numeric(index_df['close_pr'], errors='coerce')
        
        # Pivot to get each stock as a column
        pivot_df = index_df.pivot(index='trd_dt', columns='symbol', values='close_pr')
        
        # Calculate average index price (equal-weighted)
        index_price = pivot_df.mean(axis=1, skipna=True)
        index_price = index_price.dropna()
        
        if len(index_price) < 20:
            logging.error(f"Insufficient index data points ({len(index_price)}) for Mswing calculation")
            return np.nan
        
        # Calculate index Mswing
        index_mswing = calculate_mswing(index_price, 20, 50, True, 20)
        
        if not pd.isna(index_mswing):
            logging.info(f"Index Mswing calculated: {index_mswing}")
            return index_mswing
        else:
            logging.error("Failed to calculate index Mswing")
            return np.nan
            
    except Exception as e:
        logging.error(f"Error calculating index Mswing: {e}")
        return np.nan

def calculate_stock_relative_strength(db, current_date, data_fetch_days=120):
    """
    Calculate relative strength for all stocks using your exact methodology
    """
    try:
        # First calculate index Mswing
        index_mswing = calculate_index_mswing(db, current_date, data_fetch_days)
        
        if pd.isna(index_mswing):
            logging.error("Cannot calculate stock RS: Index Mswing calculation failed")
            return []
        
        # Get all stocks with sufficient data
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
                # Get stock price data
                price_query = """
                    SELECT trd_dt, close_pr
                    FROM stg_nse_data
                    WHERE symbol = %s AND trd_dt >= %s AND trd_dt <= %s
                    ORDER BY trd_dt ASC
                """
                
                db.execute(price_query, (symbol, start_date, current_date))
                price_data = db.fetchall()
                
                if len(price_data) < 20:  # min_data_points
                    continue
                
                # Convert to pandas Series with proper float conversion
                close_prices = pd.Series([float(row[1]) for row in price_data], dtype=float)
                
                # Calculate stock Mswing
                stock_mswing = calculate_mswing(close_prices, 20, 50, True, 20)
                
                if not pd.isna(stock_mswing):
                    # Calculate relative strength = Stock Mswing - Index Mswing
                    # Ensure both values are float to avoid decimal.Decimal vs float issues
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
                    
            except Exception as e:
                logging.error(f"Error calculating RS for {symbol}: {e}")
                continue
        
        logging.info(f"Calculated relative strength for {len(stock_rs_data)} stocks")
        return stock_rs_data
        
    except Exception as e:
        logging.error(f"Error in stock RS calculation: {e}")
        return []

def calculate_industry_relative_strength(stock_rs_data):
    """
    Calculate industry relative strength using your exact methodology
    """
    try:
        if not stock_rs_data:
            return []
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(stock_rs_data)
        
        # Group by industry and calculate statistics
        industry_stats = df.groupby('industry').agg({
            'relative_strength': ['mean', 'count'],
            'symbol': lambda x: (df.loc[x.index, 'relative_strength'] > 0).sum()
        }).round(2)
        
        # Flatten column names
        industry_stats.columns = ['avg_rs', 'stock_count', 'positive_rs_count']
        industry_stats = industry_stats.reset_index()
        
        # Calculate ratio
        industry_stats['ratio'] = (industry_stats['positive_rs_count'] / industry_stats['stock_count']).round(2)
        
        # Rank by average relative strength
        industry_stats['rank'] = industry_stats['avg_rs'].rank(ascending=False, method='min').astype(int)
        
        # Sort by rank
        industry_stats = industry_stats.sort_values('rank')
        
        # Get additional industry info from first stock in each industry
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
        
        logging.info(f"Calculated industry relative strength for {len(industry_rs_data)} industries")
        return industry_rs_data
        
    except Exception as e:
        logging.error(f"Error calculating industry RS: {e}")
        return []

def calculate_ibd_rs_score(stock_prices, market_prices, symbol="Unknown"):
    """
    Calculate IBD-style Relative Strength Score with adaptive periods
    
    IBD RS Rating weights (adjusted for available data):
    - If >= 252 days: 40% (3m), 20% (6m), 20% (9m), 20% (12m)
    - If >= 189 days: 50% (3m), 25% (6m), 25% (9m)
    - If >= 126 days: 60% (3m), 40% (6m)
    - If >= 63 days: 100% (3m)
    - If 50-62 days: 100% (available period)
    """
    min_days = 50
    if len(stock_prices) < min_days or len(market_prices) < min_days:
        return None
    
    try:
        # Define periods based on available data
        available_days = len(stock_prices)
        periods = {}
        weights = {}
        
        if available_days >= 252:  # Full IBD calculation
            periods = {'3m': 63, '6m': 126, '9m': 189, '12m': 252}
            weights = {'3m': 0.4, '6m': 0.2, '9m': 0.2, '12m': 0.2}
        elif available_days >= 189:  # 9 months available
            periods = {'3m': 63, '6m': 126, '9m': 189}
            weights = {'3m': 0.5, '6m': 0.25, '9m': 0.25}
        elif available_days >= 126:  # 6 months available
            periods = {'3m': 63, '6m': 126}
            weights = {'3m': 0.6, '6m': 0.4}
        elif available_days >= 63:  # 3 months available
            periods = {'3m': 63}
            weights = {'3m': 1.0}
        else:  # Less than 3 months, use available data
            periods = {'available': min(available_days, 50)}
            weights = {'available': 1.0}
        
        stock_returns = {}
        market_returns = {}
        
        for period_name, days in periods.items():
            try:
                # Adjust days if not enough data available
                actual_days = min(days, available_days)
                
                # Stock return with division by zero check
                stock_start = stock_prices.iloc[-actual_days]
                stock_end = stock_prices.iloc[-1]
                
                if stock_start == 0 or pd.isna(stock_start) or pd.isna(stock_end):
                    stock_returns[period_name] = None
                else:
                    stock_returns[period_name] = (stock_end / stock_start - 1) * 100
                
                # Market return with division by zero check
                market_start = market_prices.iloc[-actual_days]
                market_end = market_prices.iloc[-1]
                
                if market_start == 0 or pd.isna(market_start) or pd.isna(market_end):
                    market_returns[period_name] = None
                else:
                    market_returns[period_name] = (market_end / market_start - 1) * 100
                    
            except Exception:
                stock_returns[period_name] = None
                market_returns[period_name] = None
        
        # Calculate relative performance for each period
        relative_performance = {}
        for period in periods.keys():
            if (stock_returns[period] is not None and market_returns[period] is not None and 
                not pd.isna(stock_returns[period]) and not pd.isna(market_returns[period])):
                relative_performance[period] = stock_returns[period] - market_returns[period]
            else:
                relative_performance[period] = None
        
        # Apply weights
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
            logging.warning("No valid RS scores found for rating conversion")
            return [None] * len(scores)
        
        # Sort by score
        valid_scores.sort(key=lambda x: x[1])
        
        # Create rating array
        ratings = [None] * len(scores)
        
        # Handle case with only one valid score
        if len(valid_scores) == 1:
            original_index, score = valid_scores[0]
            ratings[original_index] = 50  # Middle rating for single stock
            return ratings
        
        for rank, (original_index, score) in enumerate(valid_scores):
            try:
                # Convert rank to percentile (1-99)
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
        
        # Delete existing data for current date
        delete_query = "DELETE FROM stg_relative_strength WHERE trd_dt = %s"
        db.execute(delete_query, (current_date,))
        
        # Prepare insert data with industry RS mapping
        industry_rs_map = {}
        
        # Calculate industry RS for mapping to individual stocks
        df = pd.DataFrame(stock_rs_data)
        if not df.empty:
            industry_avg_rs = df.groupby('industry')['relative_strength'].mean().round(2)
            industry_rs_map = industry_avg_rs.to_dict()
        
        # Insert new data
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
        logging.info(f"Inserted {len(insert_data)} stock relative strength records")
        
    except Exception as e:
        logging.error(f"Error inserting stock relative strength data: {e}")

def insert_industry_relative_strength(db, industry_rs_data, current_date):
    """Insert industry relative strength data into stg_ind_relative_strength table"""
    try:
        if not industry_rs_data:
            return
        
        # Delete existing data for current date
        delete_query = "DELETE FROM stg_ind_relative_strength WHERE trd_dt = %s"
        db.execute(delete_query, (current_date,))
        
        # Insert new data
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
        logging.info(f"Inserted {len(insert_data)} industry relative strength records")
        
    except Exception as e:
        logging.error(f"Error inserting industry relative strength data: {e}")

def main():
    """Main function to process stock data"""
    start_time = time.time()
    
    # Initialize warning counters
    warning_stats = {
        'division_by_zero_stock': [],
        'division_by_zero_market': [],
        'invalid_price_data': [],
        'no_stock_data': [],
        'market_data_mismatch': [],
        'insufficient_data': [],
        'calculation_errors': []
    }

    try:
        with get_db_connection() as db:
            logging.info("Starting stock data processing with RS ratings calculation")
            
            # Create required indexes
            db.create_indexes()
            
            # Create relative strength tables if they don't exist
            create_rs_tables_queries = [
                """
                CREATE TABLE IF NOT EXISTS stg_relative_strength (
                    symbol VARCHAR(50),
                    name VARCHAR(200),
                    trd_dt DATE,
                    sector VARCHAR(200),
                    sub_sector VARCHAR(200),
                    industry VARCHAR(200),
                    sub_industry VARCHAR(200),
                    rs FLOAT,
                    ind_rs FLOAT,
                    PRIMARY KEY (symbol, trd_dt)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS stg_ind_relative_strength (
                    sector VARCHAR(200),
                    sub_sector VARCHAR(200),
                    industry VARCHAR(200),
                    sub_industry VARCHAR(200),
                    trd_dt DATE,
                    ind_rs FLOAT,
                    rank INTEGER,
                    stock_cnt INTEGER,
                    positive_rs_cnt INTEGER,
                    ratio FLOAT,
                    PRIMARY KEY (industry, trd_dt)
                )
                """
            ]
            
            for table_query in create_rs_tables_queries:
                db.execute(table_query)
            
            logging.info("Ensured relative strength tables exist.")

            # Fetch unique symbols from stock_data where mcap is not null and mcap > 300
            db.execute("SELECT DISTINCT symbol FROM stock_data WHERE mcap IS NOT NULL AND mcap > 300")
            symbols = [row[0] for row in db.fetchall()]
            logging.info(f"Fetched {len(symbols)} unique symbols from stock_data with mcap > 300.")

            if not symbols:
                logging.info("No symbols found in stock_data with mcap > 300. Exiting.")
                return

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
            
            logging.info("Temporary tables created.")

            # Create indexes on temporary tables
            temp_indexes = [
                "CREATE INDEX idx_stock_data_temp_symbol_trd_dt ON stock_data_temp(symbol, trd_dt)",
                "CREATE INDEX idx_stock_data_temp_industry ON stock_data_temp(industry)",
                "CREATE INDEX idx_market_data_temp_date ON market_data_temp(trd_dt)"
            ]
            
            for index_query in temp_indexes:
                db.execute(index_query)
            
            logging.info("Indexes created on temporary tables.")

            # Fetch historical stock data with industry mapping (up to 252 days)
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
            logging.info("Historical stock data with industry mapping inserted into stock_data_temp.")

            # Fetch market data (NIFTY MID SMALL 400 average)
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
            logging.info("Market data (NIFTYMIDSML400) inserted into market_data_temp.")
            
            db.commit()

            # Create temporary table for results
            create_results_table_query = """
                CREATE TEMP TABLE temp_results (
                    symbol VARCHAR(50),
                    ema10 FLOAT,
                    ema20 FLOAT,
                    ema50 FLOAT,
                    sma200 FLOAT,
                    mom10 FLOAT,
                    rs_score FLOAT,
                    rs_score_sector FLOAT,
                    industry VARCHAR(200)
                )
            """

            # Process symbols in batches
            batch_size = 100
            all_rs_scores = []
            all_sector_rs_scores = {}
            all_results = []

            # First pass: Calculate all RS scores
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logging.info(f"Calculating RS scores for batch {i//batch_size + 1} with {len(batch_symbols)} symbols.")
                
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
                            warning_stats['no_stock_data'].append(symbol)
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': pd.Series([], dtype=float),
                                'rs_score': None,
                                'industry': 'Unknown'
                            })
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
                                warning_stats['invalid_price_data'].append(f"{symbol} on {date}")
                                continue
                                
                            close_prices.append(float(price))
                            dates.append(date)
                        
                        if len(close_prices) == 0:
                            warning_stats['no_stock_data'].append(symbol)
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': pd.Series([], dtype=float),
                                'rs_score': None,
                                'industry': industry
                            })
                            continue
                        
                        # Convert to pandas Series
                        close_prices = pd.Series(close_prices, dtype=float)
                        
                        # Check if we have minimum data for RS calculation (50 days)
                        if len(close_prices) < 50:
                            warning_stats['insufficient_data'].append(f"{symbol} ({len(close_prices)} days)")
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': close_prices,
                                'rs_score': None,
                                'industry': industry
                            })
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
                                        warning_stats['invalid_price_data'].append(f"Market data for {symbol}")
                                        market_prices.append(None)
                                    else:
                                        market_prices.append(float(price))
                                
                                # Check if we have enough valid market data
                                valid_market_data = [p for p in market_prices if p is not None]
                                if len(valid_market_data) < len(market_prices) * 0.8:  # At least 80% valid data
                                    warning_stats['market_data_mismatch'].append(f"{symbol} (insufficient valid market data)")
                                    rs_score = None
                                else:
                                    market_prices = pd.Series(market_prices).fillna(method='ffill')  # Forward fill missing values
                                    
                                    # Calculate IBD RS Score
                                    rs_score = calculate_ibd_rs_score(close_prices, market_prices, symbol)
                                    
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
                                warning_stats['market_data_mismatch'].append(f"{symbol}: expected {len(close_prices)}, got {len(market_data) if market_data else 0}")
                                all_rs_scores.append(None)
                                all_results.append({
                                    'symbol': symbol,
                                    'prices': close_prices,
                                    'rs_score': None,
                                    'industry': industry
                                })
                        else:
                            warning_stats['no_stock_data'].append(f"{symbol} (no dates)")
                            all_rs_scores.append(None)
                            all_results.append({
                                'symbol': symbol,
                                'prices': close_prices,
                                'rs_score': None,
                                'industry': industry
                            })
                            
                    except Exception as e:
                        warning_stats['calculation_errors'].append(f"{symbol} overall: {str(e)}")
                        all_rs_scores.append(None)
                        all_results.append({
                            'symbol': symbol,
                            'prices': pd.Series([], dtype=float),
                            'rs_score': None,
                            'industry': 'Unknown'
                        })

            # Log consolidated warning statistics
            logging.info("=== DATA QUALITY SUMMARY ===")
            total_processed = len(symbols)
            total_valid_rs = len([r for r in all_rs_scores if r is not None])
            
            logging.info(f"Total stocks processed: {total_processed}")
            logging.info(f"Successfully calculated RS ratings: {total_valid_rs}")
            logging.info(f"Failed RS calculations: {total_processed - total_valid_rs}")
            
            # Log warning details (show first 10 examples only)
            if warning_stats['no_stock_data']:
                examples = warning_stats['no_stock_data'][:10]
                suffix = "..." if len(warning_stats['no_stock_data']) > 10 else ""
                logging.warning(f"No stock data found ({len(warning_stats['no_stock_data'])} stocks): {', '.join(examples)}{suffix}")
            
            if warning_stats['insufficient_data']:
                examples = warning_stats['insufficient_data'][:10]
                suffix = "..." if len(warning_stats['insufficient_data']) > 10 else ""
                logging.warning(f"Insufficient data for RS calculation ({len(warning_stats['insufficient_data'])} stocks): {', '.join(examples)}{suffix}")
            
            if warning_stats['invalid_price_data']:
                examples = warning_stats['invalid_price_data'][:5]
                suffix = "..." if len(warning_stats['invalid_price_data']) > 5 else ""
                logging.warning(f"Invalid price data found ({len(warning_stats['invalid_price_data'])} instances): {', '.join(examples)}{suffix}")
            
            if warning_stats['market_data_mismatch']:
                examples = warning_stats['market_data_mismatch'][:10]
                suffix = "..." if len(warning_stats['market_data_mismatch']) > 10 else ""
                logging.warning(f"Market data mismatches ({len(warning_stats['market_data_mismatch'])} stocks): {', '.join(examples)}{suffix}")
            
            if warning_stats['calculation_errors']:
                examples = warning_stats['calculation_errors'][:5]
                suffix = "..." if len(warning_stats['calculation_errors']) > 5 else ""
                logging.error(f"Calculation errors ({len(warning_stats['calculation_errors'])} instances): {', '.join(examples)}{suffix}")
            
            logging.info("=== END DATA QUALITY SUMMARY ===")

            # Convert RS scores to ratings
            logging.info("Converting RS scores to ratings...")
            try:
                overall_rs_ratings = convert_to_rs_rating(all_rs_scores)
                logging.info(f"Converted {len([r for r in overall_rs_ratings if r is not None])} valid overall RS ratings out of {len(all_rs_scores)} stocks")
            except Exception as e:
                logging.error(f"Error converting overall RS scores to ratings: {e}")
                overall_rs_ratings = [None] * len(all_rs_scores)
            
            # Convert sector RS scores to ratings
            sector_rs_ratings = {}
            for industry, scores in all_sector_rs_scores.items():
                try:
                    sector_rs_ratings[industry] = convert_to_rs_rating(scores)
                    valid_count = len([r for r in sector_rs_ratings[industry] if r is not None])
                    logging.info(f"Converted {valid_count} valid sector RS ratings for industry '{industry}' out of {len(scores)} stocks")
                except Exception as e:
                    logging.error(f"Error converting sector RS scores to ratings for industry '{industry}': {e}")
                    sector_rs_ratings[industry] = [None] * len(scores)

            # Track updated symbols for final reporting
            updated_symbols = set()

            # Second pass: Calculate other indicators and prepare final results
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logging.info(f"Processing indicators for batch {i//batch_size + 1} with {len(batch_symbols)} symbols.")
                
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
                                logging.warning(f"Converted non-Series price data to pandas Series for {symbol}")
                            
                            # Double-check it's now a Series
                            if not isinstance(close_prices, pd.Series):
                                logging.error(f"Failed to convert price data to pandas Series for {symbol}, skipping")
                                continue
                            
                            num_days = len(close_prices)
                            
                            # Skip if no price data
                            if num_days == 0:
                                continue
                            
                            # Initialize values
                            ema10_value = ema20_value = ema50_value = sma200_value = mom10_value = None
                            
                            # Calculate momentum (requires 10 days)
                            if num_days >= 10:
                                try:
                                    last_10_slice = close_prices.iloc[-10:]
                                    last_10_days_avg = last_10_slice.mean()
                                    recent_close = close_prices.iloc[-1]
                                        
                                    if (last_10_days_avg != 0 and not pd.isna(last_10_days_avg) and 
                                        not pd.isna(recent_close) and not np.isinf(last_10_days_avg)):
                                        mom10_value = recent_close / last_10_days_avg
                                except Exception as mom_error:
                                    logging.error(f"Momentum calculation error for {symbol}: {mom_error}")
                                    mom10_value = None

                            # Calculate technical indicators based on available data
                            try:
                                if num_days >= 200:
                                    ema10 = EMAIndicator(close=close_prices, window=10, fillna=False).ema_indicator()
                                    ema20 = EMAIndicator(close=close_prices, window=20, fillna=False).ema_indicator()
                                    ema50 = EMAIndicator(close=close_prices, window=50, fillna=False).ema_indicator()
                                    sma200 = SMAIndicator(close=close_prices, window=200, fillna=False).sma_indicator()
                                    ema10_value = ema10.iloc[-1] if not ema10.empty and not pd.isna(ema10.iloc[-1]) else None
                                    ema20_value = ema20.iloc[-1] if not ema20.empty and not pd.isna(ema20.iloc[-1]) else None
                                    ema50_value = ema50.iloc[-1] if not ema50.empty and not pd.isna(ema50.iloc[-1]) else None
                                    sma200_value = sma200.iloc[-1] if not sma200.empty and not pd.isna(sma200.iloc[-1]) else None
                                elif num_days >= 50:
                                    ema10 = EMAIndicator(close=close_prices, window=10, fillna=False).ema_indicator()
                                    ema20 = EMAIndicator(close=close_prices, window=20, fillna=False).ema_indicator()
                                    ema50 = EMAIndicator(close=close_prices, window=50, fillna=False).ema_indicator()
                                    ema10_value = ema10.iloc[-1] if not ema10.empty and not pd.isna(ema10.iloc[-1]) else None
                                    ema20_value = ema20.iloc[-1] if not ema20.empty and not pd.isna(ema20.iloc[-1]) else None
                                    ema50_value = ema50.iloc[-1] if not ema50.empty and not pd.isna(ema50.iloc[-1]) else None
                                elif num_days >= 20:
                                    ema10 = EMAIndicator(close=close_prices, window=10, fillna=False).ema_indicator()
                                    ema20 = EMAIndicator(close=close_prices, window=20, fillna=False).ema_indicator()
                                    ema10_value = ema10.iloc[-1] if not ema10.empty and not pd.isna(ema10.iloc[-1]) else None
                                    ema20_value = ema20.iloc[-1] if not ema20.empty and not pd.isna(ema20.iloc[-1]) else None
                                else:
                                    continue  # Skip if insufficient data
                            except Exception as ta_error:
                                logging.error(f"Technical indicator calculation error for {symbol}: {ta_error}")
                                # Continue with None values for indicators
                            
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
                                except Exception as sector_error:
                                    logging.error(f"Sector RS rating error for {symbol}: {sector_error}")

                            batch_results.append((
                                symbol, ema10_value, ema20_value, ema50_value, sma200_value, 
                                mom10_value, rs_rating, rs_rating_sector, industry
                            ))
                            
                            batch_updated_symbols.append(symbol)
                            
                        except Exception as stock_error:
                            logging.error(f"Individual stock processing error for {symbol}: {stock_error}")
                            # Don't add to batch_results, continue to next stock

                    # Insert batch results into temp_results
                    if batch_results:
                        try:
                            insert_results_query = """
                                INSERT INTO temp_results (symbol, ema10, ema20, ema50, sma200, mom10, rs_score, rs_score_sector, industry)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            db.executemany(insert_results_query, batch_results)
                            logging.info(f"Inserted {len(batch_results)} records into temp_results for batch {i//batch_size + 1}.")
                        except Exception as e:
                            logging.error(f"Error inserting batch results for batch {i//batch_size + 1}: {e}")
                            db.rollback()
                            continue

                    # Bulk update stock_data with last_updated timestamp
                    try:
                        current_timestamp = datetime.now()
                        update_query = """
                            UPDATE stock_data
                            SET EMA10 = temp_results.ema10,
                                EMA20 = temp_results.ema20,
                                EMA50 = temp_results.ema50,
                                SMA200 = temp_results.sma200,
                                mom10 = temp_results.mom10,
                                rs_rating = temp_results.rs_score,
                                rs_rating_sector = temp_results.rs_score_sector,
                                last_updated = %s
                            FROM temp_results
                            WHERE stock_data.symbol = temp_results.symbol
                        """
                        db.execute(update_query, (current_timestamp,))
                        rows_updated = db.rowcount
                        logging.info(f"Bulk update completed for batch {i//batch_size + 1}, {rows_updated} rows affected.")
                        
                        # Track updated symbols
                        updated_symbols.update(batch_updated_symbols)

                        db.commit()
                        logging.info(f"Committed batch {i//batch_size + 1} with timestamp {current_timestamp}.")
                        
                    except Exception as e:
                        logging.error(f"Error in bulk update for batch {i//batch_size + 1}: {e}")
                        db.rollback()
                        continue

                except Exception as e:
                    logging.error(f"Error in batch {i//batch_size + 1}: {e}")
                    db.rollback()
                    continue

            # Final summary
            logging.info("=== PROCESSING COMPLETE ===")
            logging.info(f"Successfully updated {len(updated_symbols)} stocks with technical indicators and RS ratings")
            
            # NEW: Calculate and insert Relative Strength data using Mswing methodology
            logging.info("Starting Relative Strength calculations using Mswing methodology...")
            current_date = datetime.now().date()
            
            try:
                # Calculate Stock Relative Strength using your exact Mswing methodology
                logging.info("Calculating stock relative strength using Mswing...")
                stock_rs_data = calculate_stock_relative_strength(db, current_date, data_fetch_days=120)
                
                # Insert stock relative strength data
                if stock_rs_data:
                    insert_stock_relative_strength(db, stock_rs_data, current_date)
                    db.commit()  # Commit the stock RS data
                    logging.info(f"Successfully processed and committed {len(stock_rs_data)} stocks for relative strength")
                else:
                    logging.warning("No stock relative strength data to insert")
                
                # Calculate Industry Relative Strength using your exact methodology
                logging.info("Calculating industry relative strength...")
                industry_rs_data = calculate_industry_relative_strength(stock_rs_data)
                
                # Insert industry relative strength data
                if industry_rs_data:
                    insert_industry_relative_strength(db, industry_rs_data, current_date)
                    db.commit()  # Commit the industry RS data
                    logging.info(f"Successfully processed and committed {len(industry_rs_data)} industries for relative strength")
                else:
                    logging.warning("No industry relative strength data to insert")
                
                logging.info("Relative Strength calculations completed successfully using Mswing methodology")
                
            except Exception as e:
                logging.error(f"Error in Relative Strength calculations: {e}")
            
            logging.info("All processing completed successfully")

            # Cleanup temporary tables
            try:
                db.execute("DROP TABLE IF EXISTS stock_data_temp")
                db.execute("DROP TABLE IF EXISTS market_data_temp") 
                db.execute("DROP TABLE IF EXISTS temp_results")
                logging.info("Temporary tables dropped.")
            except Exception as e:
                logging.error(f"Error dropping temporary tables: {e}")

    except Exception as e:
        logging.error(f"Critical error during processing: {e}")
        return False

    finally:
        logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Stock data processing completed successfully. Check stock_data.log for details.")
    else:
        print("Stock data processing failed. Check stock_data.log for error details.")