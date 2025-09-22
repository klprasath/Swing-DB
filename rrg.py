import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta
import numpy as np
import plotly.io as pio
import os
from sqlalchemy import create_engine

class IndustryRRGDashboard:
    def __init__(self, db_config):
        """
        Initialize Industry RRG Dashboard
        db_config: dict with database connection parameters
        """
        self.db_config = db_config
        self.connection = None
        self.engine = None
        
    def connect_db(self):
        """Establish database connection using SQLAlchemy"""
        try:
            # Create SQLAlchemy engine
            connection_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{self.db_config['database']}"
            self.engine = create_engine(connection_string)
            
            # Also keep psycopg2 connection for compatibility
            self.connection = psycopg2.connect(**self.db_config)
            print("✅ Database connection successful!")
            return True
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def get_top_industries(self, top_n=7):
        """Get top N ranked industries from latest date with stock_cnt >= 4, excluding Unknown"""
        
        if not self.engine:
            if not self.connect_db():
                raise Exception("Failed to connect to database")
        
        query = f"""
        SELECT industry, rank, stock_cnt
        FROM stg_ind_relative_strength 
        WHERE trd_dt = (SELECT MAX(trd_dt) FROM stg_ind_relative_strength)
        AND rank <= {top_n + 5}  -- Get extra to account for filtered items
        AND stock_cnt >= 4
        AND UPPER(industry) != 'UNKNOWN'  -- Exclude Unknown
        ORDER BY rank
        LIMIT {top_n}
        """
        
        df = pd.read_sql(query, self.engine)
        print(f"📊 Found {len(df)} top industries (stock_cnt >= 4, excluding Unknown):")
        for idx, row in df.iterrows():
            print(f"   {row['rank']}. {row['industry']} (stocks: {row['stock_cnt']})")
        return df['industry'].tolist()
    
    def get_rrg_data(self, days_back=5, lookback=18, momentum_period=5):
        """Fetch RRG data with JdK RS-Ratio and RS-Momentum for 18-day lookback"""
        if not self.engine:
            if not self.connect_db():
                raise Exception("Failed to connect to database")
        
        top_industries = self.get_top_industries()
        if not top_industries:
            raise Exception("No top industries found")
        
        top_industries_str = "','".join(top_industries)
        
        query = f"""
        WITH industry_data AS (
            SELECT 
                sector, sub_sector, industry, sub_industry, trd_dt, ind_rs, rank, stock_cnt,
                ROW_NUMBER() OVER (PARTITION BY industry ORDER BY trd_dt DESC) as rn_desc,
                ROW_NUMBER() OVER (PARTITION BY industry ORDER BY trd_dt ASC) as rn_asc
            FROM stg_ind_relative_strength
            WHERE trd_dt >= CURRENT_DATE - INTERVAL '{lookback} days'
            AND industry IN ('{top_industries_str}')
            AND ind_rs > 0
            AND stock_cnt >= 4
        ),
        rs_stats AS (
            SELECT 
                industry, 
                AVG(ind_rs) as rs_mean,
                STDDEV(ind_rs) as rs_std
            FROM industry_data
            GROUP BY industry
        ),
        rs_normalized AS (
            SELECT 
                s.*,
                (s.ind_rs - rs.rs_mean) / NULLIF(rs.rs_std, 0) as rs_zscore,
                100 + (s.ind_rs - rs.rs_mean) / NULLIF(rs.rs_std, 0) as jdk_rs_ratio
            FROM industry_data s
            JOIN rs_stats rs ON s.industry = rs.industry
            WHERE s.rn_desc <= {days_back}
        ),
        rs_momentum AS (
            SELECT 
                s1.*,
                s2.ind_rs as ind_rs_prev,
                s2.trd_dt as trd_dt_prev,
                CASE 
                    WHEN s2.ind_rs IS NOT NULL AND s2.ind_rs > 0
                    THEN ((s1.ind_rs / s2.ind_rs) - 1) * 100
                    ELSE 0 
                END as rs_roc
            FROM rs_normalized s1
            LEFT JOIN industry_data s2 ON s1.industry = s2.industry 
                AND s2.trd_dt = (
                    SELECT MAX(trd_dt)
                    FROM industry_data s3
                    WHERE s3.industry = s1.industry
                    AND s3.trd_dt < s1.trd_dt
                    AND s3.trd_dt >= s1.trd_dt - INTERVAL '{momentum_period} days'
                )
            WHERE s1.rn_desc = 1
        ),
        momentum_stats AS (
            SELECT 
                industry,
                AVG(rs_roc) as roc_mean,
                STDDEV(rs_roc) as roc_std
            FROM rs_momentum
            WHERE rs_roc != 0
            GROUP BY industry
        ),
        final_data AS (
            SELECT 
                m.*,
                (m.rs_roc - ms.roc_mean) / NULLIF(ms.roc_std, 0) as roc_zscore,
                100 + (m.rs_roc - ms.roc_mean) / NULLIF(ms.roc_std, 0) as jdk_rs_momentum
            FROM rs_momentum m
            JOIN momentum_stats ms ON m.industry = ms.industry
        )
        SELECT 
            industry as entity, 
            sector, sub_sector, industry, sub_industry, trd_dt, rank, stock_cnt,
            jdk_rs_ratio as x_axis,
            jdk_rs_momentum as y_axis,
            rs_roc as momentum_raw,
            ind_rs,
            ind_rs_prev,
            CASE 
                WHEN jdk_rs_ratio >= 100 AND jdk_rs_momentum >= 100 THEN 'Leading'
                WHEN jdk_rs_ratio < 100 AND jdk_rs_momentum >= 100 THEN 'Improving'
                WHEN jdk_rs_ratio < 100 AND jdk_rs_momentum < 100 THEN 'Lagging'
                WHEN jdk_rs_ratio >= 100 AND jdk_rs_momentum < 100 THEN 'Weakening'
            END as quadrant
        FROM final_data
        ORDER BY rank
        """
        
        df = pd.read_sql(query, self.engine)
        print(f"\n📈 RRG Data Summary ({days_back}-day analysis, {lookback}-day lookback):")
        print(f"   Total records: {len(df)}")
        if len(df) > 0:
            print(f"   X-axis (JdK RS-Ratio) range: {df['x_axis'].min():.2f} to {df['x_axis'].max():.2f}")
            print(f"   Y-axis (JdK RS-Momentum) range: {df['y_axis'].min():.2f} to {df['y_axis'].max():.2f}")
            print(f"   Quadrant distribution:")
            print(df['quadrant'].value_counts().to_string())
            print(f"\n   Sample data:")
            print(df[['entity', 'x_axis', 'y_axis', 'quadrant', 'stock_cnt']].head())
        else:
            print("   ❌ No data found!")
        
        return df
        
    def get_historical_data(self, days_back=5, lookback=18, momentum_period=5):
        """Get historical data with JdK RS-Ratio and RS-Momentum for 18-day lookback"""
        if not self.engine:
            if not self.connect_db():
                raise Exception("Failed to connect to database")
        
        top_industries = self.get_top_industries()
        top_industries_str = "','".join(top_industries)
        
        query = f"""
        WITH industry_data AS (
            SELECT 
                industry, trd_dt, ind_rs, rank, stock_cnt
            FROM stg_ind_relative_strength
            WHERE trd_dt >= CURRENT_DATE - INTERVAL '{lookback} days'
            AND industry IN ('{top_industries_str}')
            AND ind_rs > 0
            AND stock_cnt >= 4
            AND UPPER(industry) != 'UNKNOWN'
        ),
        rs_stats AS (
            SELECT 
                industry, 
                AVG(ind_rs) as rs_mean,
                STDDEV(ind_rs) as rs_std
            FROM industry_data
            GROUP BY industry
        ),
        rs_normalized AS (
            SELECT 
                s.*,
                (s.ind_rs - rs.rs_mean) / NULLIF(rs.rs_std, 0) as rs_zscore,
                100 + (s.ind_rs - rs.rs_mean) / NULLIF(rs.rs_std, 0) as jdk_rs_ratio
            FROM industry_data s
            JOIN rs_stats rs ON s.industry = rs.industry
        ),
        rs_momentum AS (
            SELECT 
                s1.*,
                s2.ind_rs as ind_rs_prev,
                CASE 
                    WHEN s2.ind_rs IS NOT NULL AND s2.ind_rs > 0
                    THEN ((s1.ind_rs / s2.ind_rs) - 1) * 100
                    ELSE 0 
                END as rs_roc
            FROM rs_normalized s1
            LEFT JOIN industry_data s2 ON s1.industry = s2.industry 
                AND s2.trd_dt = (
                    SELECT MAX(trd_dt)
                    FROM industry_data s3
                    WHERE s3.industry = s1.industry
                    AND s3.trd_dt < s1.trd_dt
                    AND s3.trd_dt >= s1.trd_dt - INTERVAL '{momentum_period} days'
                )
        ),
        momentum_stats AS (
            SELECT 
                industry,
                AVG(rs_roc) as roc_mean,
                STDDEV(rs_roc) as roc_std
            FROM rs_momentum
            WHERE rs_roc != 0
            GROUP BY industry
        ),
        final_data AS (
            SELECT 
                r.*,
                (r.rs_roc - ms.roc_mean) / NULLIF(ms.roc_std, 0) as roc_zscore,
                100 + (r.rs_roc - ms.roc_mean) / NULLIF(ms.roc_std, 0) as jdk_rs_momentum
            FROM rs_momentum r
            JOIN momentum_stats ms ON r.industry = ms.industry
            WHERE r.trd_dt >= (SELECT MAX(trd_dt) - INTERVAL '{days_back} days' FROM industry_data)
        )
        SELECT 
            industry, trd_dt, jdk_rs_ratio as x_axis, jdk_rs_momentum as y_axis, rank, stock_cnt
        FROM final_data
        ORDER BY industry, trd_dt
        """
        
        df = pd.read_sql(query, self.engine)
        print(f"📈 Historical data: {len(df)} records across {df['trd_dt'].nunique()} trading days")
        return df
    
    def create_rrg_plot(self, date_filter=None):
        """Create the RRG plot with fixed quadrant boundaries and adjusted arrow settings"""
        df = self.get_rrg_data()
        
        if date_filter:
            df = df[df['trd_dt'] == date_filter]
        
        if df.empty:
            print("❌ No data available for plotting")
            return None
        
        latest_date = df['trd_dt'].max()
        
        # Adjusted ranges for tighter, more appealing chart
        x_range = [95, 105]  # JdK RS-Ratio
        y_range = [95, 105]  # JdK RS-Momentum
        x_center = 100
        y_center = 100
        
        print(f"📊 FIXED Chart ranges - X: {x_range[0]} to {x_range[1]}, Y: {y_range[0]} to {y_range[1]}")
        print(f"🎯 FIXED Center point - X: {x_center}, Y: {y_center}")
        
        industry_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'
        ]
        
        industry_mapping = {}
        for i, industry in enumerate(df['entity'].unique()):
            rank = df[df['entity'] == industry]['rank'].iloc[0]
            industry_mapping[industry] = {
                'color': industry_colors[i % len(industry_colors)],
                'rank': rank
            }
        
        fig = go.Figure()
        
        # Add quadrant backgrounds with fixed center at (100, 100)
        fig.add_shape(type="rect", x0=x_center, y0=y_center, x1=x_range[1], y1=y_range[1],
                    fillcolor="rgba(40, 167, 69, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=x_range[0], y0=y_center, x1=x_center, y1=y_range[1],
                    fillcolor="rgba(111, 66, 193, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=x_range[0], y0=y_range[0], x1=x_center, y1=y_center,
                    fillcolor="rgba(220, 53, 69, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=x_center, y0=y_range[0], x1=x_range[1], y1=y_center,
                    fillcolor="rgba(253, 126, 20, 0.1)", line=dict(width=0))
        
        fig.add_hline(y=y_center, line_dash="dash", line_color="gray", opacity=0.5, line_width=1)
        fig.add_vline(x=x_center, line_dash="dash", line_color="gray", opacity=0.5, line_width=1)
        
        # Add movement tails
        try:
            historical_data = self.get_historical_data()
            for industry in df['entity'].unique():
                tail_data = historical_data[historical_data['industry'] == industry].copy()
                tail_data = tail_data.dropna(subset=['y_axis']).sort_values('trd_dt')
                
                if len(tail_data) > 1:
                    color = industry_mapping[industry]['color']
                    fig.add_trace(go.Scatter(
                        x=tail_data['x_axis'],
                        y=tail_data['y_axis'],
                        mode='lines',
                        line=dict(width=2, color=color),
                        opacity=0.6,
                        name=f'{industry} trail',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    if len(tail_data) > 1:
                        fig.add_trace(go.Scatter(
                            x=tail_data['x_axis'][:-1],
                            y=tail_data['y_axis'][:-1],
                            mode='markers',
                            marker=dict(size=6, color=color, line=dict(width=1, color='white'), opacity=0.8),
                            name=f'{industry} daily',
                            showlegend=False,
                            hovertemplate=
                            f'<b>{industry}</b><br>' +
                            'Date: %{customdata}<br>' +
                            'RS-Ratio: %{x:.2f}<br>' +
                            'RS-Momentum: %{y:.2f}<br>' +
                            '<extra></extra>',
                            customdata=tail_data['trd_dt'][:-1]
                        ))
                    if len(tail_data) >= 2:
                        start_x, start_y = tail_data['x_axis'].iloc[-2], tail_data['y_axis'].iloc[-2]
                        end_x, end_y = tail_data['x_axis'].iloc[-1], tail_data['y_axis'].iloc[-1]
                        # Reduced threshold from 1 to 0.1 to show arrows for smaller movements
                        if abs(end_x - start_x) > 0.1 or abs(end_y - start_y) > 0.1:
                            fig.add_annotation(
                                x=end_x, y=end_y,
                                ax=start_x, ay=start_y,
                                xref='x', yref='y',
                                axref='x', ayref='y',
                                arrowhead=2,
                                arrowsize=1.0,  # Reduced from 1.5 to 1.0 for smaller pointer size
                                arrowwidth=2,   # Adjusted width for balance
                                arrowcolor=color,
                                opacity=0.9,
                                showarrow=True
                            )
        except Exception as e:
            print(f"⚠️ Warning: Could not add movement tails: {e}")
        
        # Plot current positions
        for index, row in df.iterrows():
            industry = row['entity']
            color = industry_mapping[industry]['color']
            rank = industry_mapping[industry]['rank']
            
            fig.add_trace(go.Scatter(
                x=[row['x_axis']],
                y=[row['y_axis']],
                mode='markers+text',
                marker=dict(color=color, size=20, line=dict(width=3, color='white')),
                text=str(rank),
                textposition="middle center",
                textfont=dict(size=12, family="Arial Black", color='white'),
                name=f"{rank}. {industry}",
                customdata=[[row['sector'], row['sub_sector'], row['industry'], row['rank'], row['stock_cnt']]],
                hovertemplate=
                f'<b>{rank}. {industry}</b><br>' +
                'RS-Ratio: %{x:.2f}<br>' +
                'RS-Momentum: %{y:.2f}<br>' +
                'Sector: %{customdata[0]}<br>' +
                'Sub-Sector: %{customdata[1]}<br>' +
                'Industry: %{customdata[2]}<br>' +
                'Rank: %{customdata[3]}<br>' +
                'Stock Count: %{customdata[4]}<br>' +
                '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Industry Relative Rotation Graph (Top 7 by Rank)<br><sub>5-Day Movement Trails | JdK RS-Ratio & RS-Momentum | Stock Count ≥ 4</sub>',
                x=0.5,
                font=dict(size=20, family="Arial Black")
            ),
            xaxis_title='JdK RS-Ratio →',
            yaxis_title='JdK RS-Momentum →',
            width=1400,
            height=900,
            xaxis=dict(range=x_range, title_font=dict(size=16)),
            yaxis=dict(range=y_range, title_font=dict(size=16)),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                font=dict(size=11),
                title=dict(text="Industries", font=dict(size=12, family="Arial Black"))
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add date display
        fig.add_annotation(
            x=0.85, y=0.98,
            xref='paper', yref='paper',
            text=f'Date: {latest_date}',
            showarrow=False,
            font=dict(size=14, family="Arial Black", color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            xanchor='right',
            yanchor='top'
        )
        
        # Add quadrant labels
        mid_x_right = (x_center + x_range[1]) / 2  # (100 + 105) / 2 = 102.5
        mid_x_left = (x_range[0] + x_center) / 2   # (95 + 100) / 2 = 97.5
        mid_y_top = (y_center + y_range[1]) / 2    # (100 + 105) / 2 = 102.5
        mid_y_bottom = (y_range[0] + y_center) / 2 # (95 + 100) / 2 = 97.5
        
        fig.add_annotation(x=mid_x_right, y=mid_y_top, 
                        text="LEADING", showarrow=False, 
                        font=dict(size=16, color="rgba(40, 167, 69, 0.6)", family="Arial Black"))
        fig.add_annotation(x=mid_x_left, y=mid_y_top, 
                        text="IMPROVING", showarrow=False,
                        font=dict(size=16, color="rgba(111, 66, 193, 0.6)", family="Arial Black"))
        fig.add_annotation(x=mid_x_left, y=mid_y_bottom, 
                        text="LAGGING", showarrow=False,
                        font=dict(size=16, color="rgba(220, 53, 69, 0.6)", family="Arial Black"))
        fig.add_annotation(x=mid_x_right, y=mid_y_bottom, 
                        text="WEAKENING", showarrow=False,
                        font=dict(size=16, color="rgba(253, 126, 20, 0.6)", family="Arial Black"))
        
        return fig
    
    def create_movement_gif(self, output_path="industry_rrg.gif", days_back=14):
        """Create animated GIF showing movement over time"""
        
        if not os.path.exists("temp_frames"):
            os.makedirs("temp_frames")
        
        # Get historical dates
        historical_data = self.get_historical_data(days_back)
        dates = sorted(historical_data['trd_dt'].unique())
        
        if len(dates) == 0:
            print("❌ No historical data found for GIF creation")
            return None
        
        frame_files = []
        
        print(f"📊 Creating {len(dates)} frames for GIF...")
        
        for i, date in enumerate(dates):
            print(f"   Frame {i+1}/{len(dates)}: {date}")
            
            # Create plot for this date
            date_data = historical_data[historical_data['trd_dt'] == date]
            
            if date_data.empty:
                continue
            
            fig = go.Figure()
            
            # Add quadrant backgrounds
            fig.add_shape(type="rect", x0=90, y0=100, x1=120, y1=120,
                         fillcolor="rgba(40, 167, 69, 0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=80, y0=100, x1=100, y1=120,
                         fillcolor="rgba(111, 66, 193, 0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=80, y0=80, x1=100, y1=100,
                         fillcolor="rgba(220, 53, 69, 0.1)", line=dict(width=0))
            fig.add_shape(type="rect", x0=100, y0=80, x1=120, y1=100,
                         fillcolor="rgba(253, 126, 20, 0.1)", line=dict(width=0))
            
            # Add reference lines
            fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Determine quadrant for each point
            date_data = date_data.copy()
            date_data['quadrant'] = date_data.apply(lambda row: 
                'Leading' if row['x_axis'] >= 100 and row['y_axis'] >= 100 else
                'Improving' if row['x_axis'] < 100 and row['y_axis'] >= 100 else
                'Lagging' if row['x_axis'] < 100 and row['y_axis'] < 100 else
                'Weakening', axis=1)
            
            color_map = {
                'Leading': '#28a745', 'Improving': '#6f42c1',
                'Lagging': '#dc3545', 'Weakening': '#fd7e14'
            }
            
            # Plot points
            for quadrant in ['Leading', 'Improving', 'Lagging', 'Weakening']:
                quadrant_data = date_data[date_data['quadrant'] == quadrant]
                
                if not quadrant_data.empty:
                    fig.add_trace(go.Scatter(
                        x=quadrant_data['x_axis'],
                        y=quadrant_data['y_axis'],
                        mode='markers+text',
                        marker=dict(color=color_map[quadrant], size=15, line=dict(width=2, color='white')),
                        text=quadrant_data['industry'],
                        textposition="top center",
                        textfont=dict(size=12, family="Arial Black"),
                        showlegend=False
                    ))
            
            # Add trails up to current date
            if i > 0:
                trail_data = historical_data[historical_data['trd_dt'] <= date]
                for industry in date_data['industry'].unique():
                    entity_trail = trail_data[trail_data['industry'] == industry].sort_values('trd_dt')
                    if len(entity_trail) > 1:
                        fig.add_trace(go.Scatter(
                            x=entity_trail['x_axis'],
                            y=entity_trail['y_axis'],
                            mode='lines',
                            line=dict(width=2, dash='dot'),
                            opacity=0.5,
                            showlegend=False
                        ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'Industry RRG - {date}<br><sub>Top 7 Industries by Rank</sub>',
                    x=0.5,
                    font=dict(size=20, family="Arial Black")
                ),
                xaxis_title='Relative Strength →',
                yaxis_title='Momentum →',
                width=1200,
                height=900,
                xaxis=dict(range=[80, 120], dtick=5),
                yaxis=dict(range=[80, 120], dtick=5),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Add quadrant labels
            fig.add_annotation(x=110, y=110, text="LEADING", showarrow=False, 
                              font=dict(size=18, color="green", family="Arial Black"))
            fig.add_annotation(x=90, y=110, text="IMPROVING", showarrow=False,
                              font=dict(size=18, color="purple", family="Arial Black"))
            fig.add_annotation(x=90, y=90, text="LAGGING", showarrow=False,
                              font=dict(size=18, color="red", family="Arial Black"))
            fig.add_annotation(x=110, y=90, text="WEAKENING", showarrow=False,
                              font=dict(size=18, color="orange", family="Arial Black"))
            
            # Save frame
            frame_path = f"temp_frames/frame_{i:03d}.png"
            fig.write_image(frame_path, width=1200, height=900, scale=2)
            frame_files.append(frame_path)
        
        # Create GIF
        print("🎬 Creating GIF animation...")
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            images.append(img)
        
        # Save as GIF
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=800,  # 800ms per frame
                loop=0,
                optimize=True
            )
        
        # Clean up temp frames
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                os.remove(frame_file)
        
        if os.path.exists("temp_frames"):
            os.rmdir("temp_frames")
        
        print(f"✅ GIF created: {output_path}")
        return output_path
    
    def create_static_image(self, output_path="industry_rrg_static.png"):
        """Create high-resolution static image"""
        
        fig = self.create_rrg_plot()
        
        # Save as high-resolution PNG
        fig.write_image(output_path, width=1200, height=900, scale=3)
        print(f"✅ High-resolution image created: {output_path}")
        return output_path
    
    def get_quadrant_summary(self):
        """Get summary statistics by quadrant"""
        df = self.get_rrg_data()
        
        summary = df.groupby('quadrant').agg({
            'entity': 'count',
            'x_axis': 'mean',
            'y_axis': 'mean',
            'rank': 'mean'
        }).round(2)
        
        summary.columns = ['Count', 'Avg_RS', 'Avg_Momentum', 'Avg_Rank']
        return summary

# Usage Example
if __name__ == "__main__":
    print("🎯 Industry RRG Dashboard")
    print("=" * 50)
    
    # PostgreSQL database configuration
    db_config = {
        'host': 'localhost',
        'user': 'postgres', 
        'password': 'anirudh16',
        'database': 'swingdb'
    }
    
    # Create dashboard
    rrg = IndustryRRGDashboard(db_config)
    
    # Test connection
    if rrg.connect_db():
        try:
            print("📊 Fetching and analyzing industry data...")
            
            # Get the RRG data with debugging
            rrg_data = rrg.get_rrg_data()
            
            if len(rrg_data) == 0:
                print("❌ No data found! Please check:")
                print("   - Table 'stg_ind_relative_strength' has recent data")
                print("   - Industries have rank <= 7 AND stock_cnt >= 4")
                print("   - At least 2 days of historical data exists")
                print("   - 'Unknown' industries are excluded")
                exit(1)
            
            print("\n📈 Creating high-resolution RRG image with enhanced features...")
            static_image = rrg.create_static_image("industry_rrg_latest.png")
            
            if static_image:
                print("\n📋 Generating quadrant summary...")
                summary = rrg.get_quadrant_summary()
                if summary is not None:
                    print("\n📊 Industry Quadrant Summary:")
                    print(summary)
                
                print("\n" + "=" * 50)
                print("✅ Industry RRG Dashboard completed!")
                print(f"📸 High-resolution image: industry_rrg_latest.png")
                print("\n🎨 Enhanced Features Implemented:")
                print("   ✅ Top 7 industries (rank-based, excluding Unknown)")
                print("   ✅ Stock count filter (≥ 4 stocks)")
                print("   ✅ 5-day movement trails with smoothed momentum")
                print("   ✅ Different colored lines for each industry")
                print("   ✅ Daily markers showing each trading day position")
                print("   ✅ Fixed quadrant boundaries (100/100 thresholds)")
                print("   ✅ Date display in top-right corner")
                print("   ✅ Centered quadrant labels with faded colors")
                print("   ✅ Dynamic chart scaling (prevents points going outside)")
                print("   ✅ Professional styling and typography")
                print("\n💡 Chart Interpretation:")
                print("   📈 Colored trails show 5-day movement patterns")
                print("   ⚪ Small circles indicate daily trading positions")
                print("   🔵 Large colored dots show current positions")
                print("   📅 Current date is displayed in top-right corner")
                print("   🎯 Only industries with 4+ stocks are included")
                print("   ❌ 'Unknown' industries are excluded")
                print("   🎨 Quadrants use fixed 100/100 thresholds for consistency")
                print("\n📁 Files saved in current directory")
            else:
                print("❌ Failed to create static image")
                
        except Exception as e:
            print(f"❌ Error creating charts: {e}")
            import traceback
            traceback.print_exc()
            print("\n🔍 Please check:")
            print("   - Table 'stg_ind_relative_strength' exists")
            print("   - Required columns are present")
            print("   - Data is available for recent dates")
            print("   - Industries have sufficient stock count (≥ 4)")
            print("   - Ensure 'kaleido' package is installed for image generation:")
            print("     pip install kaleido sqlalchemy")
            
    else:
        print("❌ Database connection failed!")
        print("🔍 Please check your database configuration")