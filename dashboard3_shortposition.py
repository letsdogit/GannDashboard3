# gann_short_options_strategy.py
"""
GANN Pro — SHORT Options Strategy Dashboard
Focus: SHORT Straddle & SHORT Strangle strategies on GANN dates
Strategy: SELL volatility (collect premium) instead of buying it
Entry: Market open on GANN dates
Exit: Market close on GANN dates
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta, time as dt_time
from dateutil.relativedelta import relativedelta
from fpdf import FPDF
import io
import base64
import time
import traceback
import math

# ---------------------------
# App config + CSS
# ---------------------------
st.set_page_config(page_title="GANN SHORT Options Strategy", layout="wide", page_icon="📉")
st.markdown("""
<style>
:root{--bg:#061026;--card:#0b1626;--muted:#94aace;--accent:#7dd3fc;--accent2:#a78bfa;}
body{background:linear-gradient(180deg,var(--bg),#020815); color:#eaf3ff;}
.block-container{padding-top:1rem;}
.stButton>button{background:linear-gradient(90deg,var(--accent),var(--accent2)); border:none; color:#012; font-weight:700;}
.card{background:rgba(255,255,255,0.03); padding:12px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.6);}
.small{color:var(--muted); font-size:13px;}
.profit{color:#4ade80; font-weight:700;}
.loss{color:#f87171; font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>📉 GANN SHORT Options Strategy Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>SHORT Straddle & SHORT Strangle strategies | SELL volatility on GANN dates | Entry: Open | Exit: Close</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------
# Helpers
# ---------------------------
def safe_fmt(val, fmt="{:.2f}", na="N/A"):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return na
        return fmt.format(val)
    except Exception:
        return na

@st.cache_data(ttl=3600, show_spinner=False)
def yf_download_robust(ticker, start, end, max_retries=3):
    """Robust Yahoo Finance download with proper error handling"""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
            
            if df is None or df.empty:
                st.warning(f"Attempt {attempt + 1}: No data returned for {ticker}")
                time.sleep(1)
                continue
            
            df = df.reset_index()
            
            if 'Datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['Datetime'])
            elif 'Date' not in df.columns:
                st.error(f"Date column missing for {ticker}")
                continue
            else:
                df['Date'] = pd.to_datetime(df['Date'])
            
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    for alt in [col.lower(), col.upper()]:
                        if alt in df.columns:
                            df[col] = df[alt]
                            break
                    else:
                        df[col] = np.nan
            
            df['Return_Pct'] = df['Close'].pct_change() * 100
            df['__SOURCE_TICKER'] = ticker
            
            st.success(f"✓ Downloaded {len(df)} daily candles for {ticker}")
            return df
            
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            time.sleep(1)
            continue
    
    st.error(f"Failed to download {ticker} after {max_retries} attempts")
    return pd.DataFrame()

# ---------------------------
# GANN generation functions
# ---------------------------
SPRING_EQ = (3, 21)

def generate_static_angles(years, angles):
    rows = []
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        for a in angles:
            offset = int(round((a/360.0)*365.25))
            rows.append({'GANN_Date': (base + timedelta(days=offset)), 'Type': f"{a}° from Equinox", 'Source':'Angle'})
    return pd.DataFrame(rows)

def generate_equinox_solstice(years):
    mapping = {'Spring Equinox':(3,21),'Summer Solstice':(6,21),'Fall Equinox':(9,23),'Winter Solstice':(12,21)}
    rows = []
    for y in years:
        for name, (m, d) in mapping.items():
            rows.append({'GANN_Date': date(y, m, d), 'Type': name, 'Source':'EquinoxSolstice'})
    return pd.DataFrame(rows)

def generate_pressure(years, methods):
    rows = []
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        quarters = [base + relativedelta(months=+q) for q in (3, 6, 9, 12)]
        if 'simple' in methods:
            cycles = [7, 14, 28]
            for cp in [base] + quarters:
                for c in cycles:
                    for n in range(1, 13):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type':f'Pressure_{c}d','Source':'Simple'})
        if 'advanced' in methods:
            cycles = [45, 60, 90, 120]
            for cp in [base] + quarters:
                for c in cycles:
                    for n in range(1, 10):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type':f'Pressure_{c}d','Source':'Advanced'})
    df = pd.DataFrame(rows)
    if not df.empty:
        df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
        df = df.drop_duplicates(subset=['GANN_Date','Type'])
    return df

def build_gann_master(years, angles, methods):
    pieces = [generate_static_angles(years, angles), generate_equinox_solstice(years), generate_pressure(years, methods)]
    df = pd.concat(pieces, ignore_index=True, sort=False)
    df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
    df = df.drop_duplicates(subset=['GANN_Date','Type']).sort_values('GANN_Date').reset_index(drop=True)
    return df

# ---------------------------
# Options pricing helpers (Simplified)
# ---------------------------
def estimate_option_premium(spot_price, strike, volatility, days_to_expiry, option_type='call'):
    """Simplified option pricing based on volatility and moneyness"""
    if option_type.lower() == 'call':
        intrinsic = max(0, spot_price - strike)
    else:
        intrinsic = max(0, strike - spot_price)
    
    moneyness = abs(spot_price - strike) / spot_price
    time_value = volatility * math.sqrt(days_to_expiry / 365) * spot_price * (1 - moneyness * 2)
    
    premium = max(intrinsic + time_value, 0.01)
    return premium

def calculate_short_straddle_pnl(entry_price, exit_price, entry_premium_per_leg, exit_premium_per_leg, contracts=1, lot_size=50):
    """
    Calculate P&L for SHORT Straddle (SELL ATM Call + ATM Put)
    We RECEIVE premium at entry and PAY to exit
    """
    # Entry: SELL both options - we RECEIVE premium (credit)
    entry_credit = entry_premium_per_leg * 2 * contracts * lot_size
    
    # Exit: BUY BACK both options - we PAY premium (debit)
    exit_cost = exit_premium_per_leg * 2 * contracts * lot_size
    
    # P&L = Premium received - Cost to buyback
    pnl = entry_credit - exit_cost
    pnl_pct = (pnl / entry_credit) * 100 if entry_credit > 0 else 0
    
    return pnl, pnl_pct, entry_credit, exit_cost

def calculate_short_strangle_pnl(entry_call_prem, entry_put_prem, exit_call_prem, exit_put_prem, contracts=1, lot_size=50):
    """
    Calculate P&L for SHORT Strangle (SELL OTM Call + OTM Put)
    We RECEIVE premium at entry and PAY to exit
    """
    # Entry: SELL both OTM options - we RECEIVE premium
    entry_credit = (entry_call_prem + entry_put_prem) * contracts * lot_size
    
    # Exit: BUY BACK both options - we PAY premium
    exit_cost = (exit_call_prem + exit_put_prem) * contracts * lot_size
    
    # P&L = Premium received - Cost to buyback
    pnl = entry_credit - exit_cost
    pnl_pct = (pnl / entry_credit) * 100 if entry_credit > 0 else 0
    
    return pnl, pnl_pct, entry_credit, exit_cost

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Strategy Settings")
    
    st.markdown("### 📊 Market Selection")
    ticker_options = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC"
    }
    
    selected_market = st.selectbox("Select Index", list(ticker_options.keys()), index=0)
    ticker = ticker_options[selected_market]
    
    st.markdown("### 📅 Date Range")
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - relativedelta(months=6))
    
    st.markdown("### 🔢 GANN Configuration")
    years = st.slider("GANN years", 2023, 2026, (2024, 2025))
    years_list = list(range(years[0], years[1]+1))
    
    all_angles = [30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    angles_sel = st.multiselect("Angles", all_angles, default=[45, 90, 180, 270])
    
    pressure_methods = st.multiselect("Pressure methods", ['simple', 'advanced'], default=['simple'])
    
    st.markdown("### ⚙️ Options Parameters")
    
    implied_vol = st.slider("Implied Volatility (%)", 10, 100, 25) / 100
    
    st.markdown("**Strangle Strike Selection:**")
    otm_percent = st.slider("OTM % from spot", 1, 10, 5)
    
    st.markdown("**Contract & Lot Size:**")
    contracts = st.number_input("Contracts to trade", min_value=1, max_value=100, value=1)
    lot_size = st.number_input("Lot size", min_value=1, max_value=200, value=50)
    
    st.markdown("**Days to Expiry:**")
    days_to_expiry = st.slider("Days until weekly expiry", 1, 7, 3)
    
    if st.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()

# ---------------------------
# Build GANN master
# ---------------------------
st.markdown("---")
st.subheader("📅 GANN Dates Generation")

with st.spinner("Generating GANN dates..."):
    gann_master = build_gann_master(years_list, angles_sel, pressure_methods)
    st.success(f"✓ Generated {len(gann_master)} GANN dates")

gann_in_range = gann_master[
    (gann_master['GANN_Date'] >= start_date) & 
    (gann_master['GANN_Date'] <= end_date)
].copy()

st.info(f"📊 Found {len(gann_in_range)} GANN dates in selected range ({start_date} to {end_date})")

with st.expander("View GANN Dates", expanded=False):
    st.dataframe(gann_in_range, use_container_width=True, height=300)

# ---------------------------
# Fetch daily data
# ---------------------------
st.markdown("---")
st.subheader("📊 Fetching Market Data")

with st.spinner(f"Downloading daily data for {selected_market}..."):
    daily_df = yf_download_robust(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    )

if daily_df.empty:
    st.error("⚠️ No data available. Check ticker or date range.")
    st.stop()

st.success(f"✓ Loaded {len(daily_df)} daily candles")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📉 SHORT Straddle", "🔻 SHORT Strangle", "📁 Export"])

# Overview
with tab1:
    st.subheader("Strategy Overview - SELLING Volatility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 SHORT Straddle Strategy")
        st.markdown(f"""
        **Concept:** SELL ATM Call + ATM Put (Collect Premium)
        
        **Entry:**
        - 📍 Strike: At-The-Money (ATM)
        - 💰 SELL both Call and Put at ATM strike
        - 💵 Entry Credit = 2 × ATM Premium × {contracts} × {lot_size}
        
        **Exit:**
        - 🔙 BUY BACK both options at end of day
        
        **Profit:** When market stays FLAT (time decay works for us)
        **Max Profit:** Premium received (limited to credit)
        **Max Loss:** Unlimited (if market moves sharply)
        
        **Example:**
        - Spot: 24,000
        - ATM Strike: 24,000
        - SELL Call Premium: ₹150, SELL Put Premium: ₹150
        - Entry Credit (We Receive): ₹{150*2*contracts*lot_size:,}
        - If market stays near 24,000, we keep most of the premium!
        """)
    
    with col2:
        st.markdown("### 🔻 SHORT Strangle Strategy")
        st.markdown(f"""
        **Concept:** SELL OTM Call + OTM Put (Collect Premium, Wider Range)
        
        **Entry:**
        - 📍 SELL Call Strike: {otm_percent}% above spot
        - 📍 SELL Put Strike: {otm_percent}% below spot
        - 💰 SELL both OTM options
        - 💵 Entry Credit = (OTM Call + OTM Put) × {contracts} × {lot_size}
        
        **Exit:**
        - 🔙 BUY BACK both options at end of day
        
        **Profit:** When market stays within strike range
        **Max Profit:** Premium received (lower than straddle but safer)
        **Max Loss:** Unlimited beyond strikes (but wider range)
        
        **Example:**
        - Spot: 24,000
        - SELL Call Strike: {int(24000 * (1 + otm_percent/100)):,}
        - SELL Put Strike: {int(24000 * (1 - otm_percent/100)):,}
        - SELL Call Premium: ₹80, SELL Put Premium: ₹80
        - Entry Credit (We Receive): ₹{(80+80)*contracts*lot_size:,}
        - Profitable if market stays between {int(24000 * (1 - otm_percent/100)):,} - {int(24000 * (1 + otm_percent/100)):,}
        """)
    
    st.markdown("---")
    st.markdown("### 📈 Market Data Preview")
    
    if not daily_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        last_price = daily_df['Close'].iloc[-1]
        period_high = daily_df['High'].max()
        period_low = daily_df['Low'].min()
        period_range = ((period_high - period_low) / period_low) * 100
        
        col1.metric("Last Price", safe_fmt(last_price))
        col2.metric("Period High", safe_fmt(period_high))
        col3.metric("Period Low", safe_fmt(period_low))
        col4.metric("Period Range %", safe_fmt(period_range, "{:.2f}%"))
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=daily_df['Date'],
            open=daily_df['Open'],
            high=daily_df['High'],
            low=daily_df['Low'],
            close=daily_df['Close'],
            name='Price'
        ))
        
        for gann_date in gann_in_range['GANN_Date']:
            fig.add_vline(x=pd.to_datetime(gann_date), line_dash="dash", line_color="yellow", opacity=0.5)
        
        fig.update_layout(
            title=f"{selected_market} - Daily Chart with GANN Dates",
            template='plotly_dark',
            height=400,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

# SHORT Straddle
with tab2:
    st.subheader("📉 SHORT Straddle Backtest")
    
    st.markdown(f"""
    **Strategy:** SELL ATM Call + ATM Put on GANN dates (Collect Premium)
    - **Contracts:** {contracts}
    - **Lot Size:** {lot_size}
    - **Implied Vol:** {implied_vol*100:.0f}%
    - **Days to Expiry:** {days_to_expiry}
    """)
    
    if st.button("▶️ Run SHORT Straddle Backtest", key="straddle_btn"):
        with st.spinner("Backtesting SHORT Straddle strategy..."):
            results = []
            
            for _, gann_row in gann_in_range.iterrows():
                gann_date = gann_row['GANN_Date']
                day_data = daily_df[daily_df['Date'].dt.date == gann_date]
                
                if day_data.empty:
                    continue
                
                entry_price = day_data.iloc[0]['Open']
                exit_price = day_data.iloc[0]['Close']
                
                atm_strike = round(entry_price / 50) * 50
                
                # Entry: SELL options (receive premium)
                entry_call_prem = estimate_option_premium(entry_price, atm_strike, implied_vol, days_to_expiry, 'call')
                entry_put_prem = estimate_option_premium(entry_price, atm_strike, implied_vol, days_to_expiry, 'put')
                entry_premium_avg = (entry_call_prem + entry_put_prem) / 2
                
                # Exit: BUY BACK options (pay premium)
                exit_call_prem = estimate_option_premium(exit_price, atm_strike, implied_vol, 0.5, 'call')
                exit_put_prem = estimate_option_premium(exit_price, atm_strike, implied_vol, 0.5, 'put')
                exit_premium_avg = (exit_call_prem + exit_put_prem) / 2
                
                # Calculate P&L (SHORT logic: credit - debit)
                pnl, pnl_pct, entry_credit, exit_cost = calculate_short_straddle_pnl(
                    entry_price, exit_price, entry_premium_avg, exit_premium_avg, contracts, lot_size
                )
                
                results.append({
                    'GANN_Date': gann_date,
                    'GANN_Type': gann_row['Type'],
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'ATM_Strike': atm_strike,
                    'Entry_Credit': entry_credit,
                    'Exit_Cost': exit_cost,
                    'P&L': pnl,
                    'P&L_%': pnl_pct,
                    'Price_Move_%': ((exit_price - entry_price) / entry_price) * 100
                })
            
            if results:
                results_df = pd.DataFrame(results)
                
                st.markdown("### 📊 Performance Summary")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_trades = len(results_df)
                winning_trades = len(results_df[results_df['P&L'] > 0])
                losing_trades = len(results_df[results_df['P&L'] < 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                total_pnl = results_df['P&L'].sum()
                avg_win = results_df[results_df['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
                avg_loss = results_df[results_df['P&L'] < 0]['P&L'].mean() if losing_trades > 0 else 0
                
                col1.metric("Total Trades", total_trades)
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Total P&L", f"₹{total_pnl:,.0f}")
                col4.metric("Avg Win", f"₹{avg_win:,.0f}")
                col5.metric("Avg Loss", f"₹{avg_loss:,.0f}")
                
                st.markdown("### 📋 Trade Details")
                st.dataframe(results_df, use_container_width=True, height=400)
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Bar(
                    x=results_df['GANN_Date'],
                    y=results_df['P&L'],
                    marker_color=['green' if x > 0 else 'red' for x in results_df['P&L']],
                    name='P&L'
                ))
                fig_pnl.update_layout(
                    title="Trade-by-Trade P&L (SHORT Straddle)",
                    template='plotly_dark',
                    height=400,
                    xaxis_title="GANN Date",
                    yaxis_title="P&L (₹)"
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                results_df['Cumulative_PnL'] = results_df['P&L'].cumsum()
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=results_df['GANN_Date'],
                    y=results_df['Cumulative_PnL'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='#7dd3fc', width=2),
                    fill='tozeroy'
                ))
                fig_cum.update_layout(
                    title="Cumulative P&L Curve (SHORT Straddle)",
                    template='plotly_dark',
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L (₹)"
                )
                st.plotly_chart(fig_cum, use_container_width=True)
                
                st.session_state['straddle_results'] = results_df
                
            else:
                st.warning("No trades found for the selected date range")

# SHORT Strangle
with tab3:
    st.subheader("🔻 SHORT Strangle Backtest")
    
    st.markdown(f"""
    **Strategy:** SELL OTM Call ({otm_percent}% above) + OTM Put ({otm_percent}% below) on GANN dates
    - **Contracts:** {contracts}
    - **Lot Size:** {lot_size}
    - **Implied Vol:** {implied_vol*100:.0f}%
    - **Days to Expiry:** {days_to_expiry}
    """)
    
    if st.button("▶️ Run SHORT Strangle Backtest", key="strangle_btn"):
        with st.spinner("Backtesting SHORT Strangle strategy..."):
            results = []
            
            for _, gann_row in gann_in_range.iterrows():
                gann_date = gann_row['GANN_Date']
                day_data = daily_df[daily_df['Date'].dt.date == gann_date]
                
                if day_data.empty:
                    continue
                
                entry_price = day_data.iloc[0]['Open']
                exit_price = day_data.iloc[0]['Close']
                
                call_strike = round((entry_price * (1 + otm_percent/100)) / 50) * 50
                put_strike = round((entry_price * (1 - otm_percent/100)) / 50) * 50
                
                # Entry: SELL OTM options
                entry_call_prem = estimate_option_premium(entry_price, call_strike, implied_vol, days_to_expiry, 'call')
                entry_put_prem = estimate_option_premium(entry_price, put_strike, implied_vol, days_to_expiry, 'put')
                
                # Exit: BUY BACK OTM options
                exit_call_prem = estimate_option_premium(exit_price, call_strike, implied_vol, 0.5, 'call')
                exit_put_prem = estimate_option_premium(exit_price, put_strike, implied_vol, 0.5, 'put')
                
                pnl, pnl_pct, entry_credit, exit_cost = calculate_short_strangle_pnl(
                    entry_call_prem, entry_put_prem, exit_call_prem, exit_put_prem, contracts, lot_size
                )
                
                results.append({
                    'GANN_Date': gann_date,
                    'GANN_Type': gann_row['Type'],
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Call_Strike': call_strike,
                    'Put_Strike': put_strike,
                    'Entry_Credit': entry_credit,
                    'Exit_Cost': exit_cost,
                    'P&L': pnl,
                    'P&L_%': pnl_pct,
                    'Price_Move_%': ((exit_price - entry_price) / entry_price) * 100
                })
            
            if results:
                results_df = pd.DataFrame(results)
                
                st.markdown("### 📊 Performance Summary")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_trades = len(results_df)
                winning_trades = len(results_df[results_df['P&L'] > 0])
                losing_trades = len(results_df[results_df['P&L'] < 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                total_pnl = results_df['P&L'].sum()
                avg_win = results_df[results_df['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
                avg_loss = results_df[results_df['P&L'] < 0]['P&L'].mean() if losing_trades > 0 else 0
                
                col1.metric("Total Trades", total_trades)
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Total P&L", f"₹{total_pnl:,.0f}")
                col4.metric("Avg Win", f"₹{avg_win:,.0f}")
                col5.metric("Avg Loss", f"₹{avg_loss:,.0f}")
                
                st.markdown("### 📋 Trade Details")
                st.dataframe(results_df, use_container_width=True, height=400)
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Bar(
                    x=results_df['GANN_Date'],
                    y=results_df['P&L'],
                    marker_color=['green' if x > 0 else 'red' for x in results_df['P&L']],
                    name='P&L'
                ))
                fig_pnl.update_layout(
                    title="Trade-by-Trade P&L (SHORT Strangle)",
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                results_df['Cumulative_PnL'] = results_df['P&L'].cumsum()
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=results_df['GANN_Date'],
                    y=results_df['Cumulative_PnL'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='#a78bfa', width=2),
                    fill='tozeroy'
                ))
                fig_cum.update_layout(
                    title="Cumulative P&L Curve (SHORT Strangle)",
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_cum, use_container_width=True)
                
                st.session_state['strangle_results'] = results_df
                
            else:
                st.warning("No trades found for the selected date range")

# Export
with tab4:
    st.subheader("📁 Export Results")
    
    st.markdown("### CSV Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'straddle_results' in st.session_state:
            csv = st.session_state['straddle_results'].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            download_link = f'<a href="data:file/csv;base64,{b64}" download="short_straddle_results_{datetime.now().strftime("%Y%m%d")}.csv" style="text-decoration:none;"><button style="background:#7dd3fc;color:#012;padding:10px 20px;border:none;border-radius:5px;font-weight:700;">📥 Download SHORT Straddle CSV</button></a>'
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.info("Run SHORT Straddle backtest first")
    
    with col2:
        if 'strangle_results' in st.session_state:
            csv = st.session_state['strangle_results'].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            download_link = f'<a href="data:file/csv;base64,{b64}" download="short_strangle_results_{datetime.now().strftime("%Y%m%d")}.csv" style="text-decoration:none;"><button style="background:#a78bfa;color:#012;padding:10px 20px;border:none;border-radius:5px;font-weight:700;">📥 Download SHORT Strangle CSV</button></a>'
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.info("Run SHORT Strangle backtest first")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: var(--muted); font-size: 12px; padding: 20px 0;'>
    <p><strong>© 2025 GANN SHORT Options Strategy Dashboard</strong></p>
    <p>⚠️ Disclaimer: SHORT options have unlimited risk. Use stop-losses. Backtesting uses estimates. Not financial advice.</p>
    <p>Built with Streamlit • Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
