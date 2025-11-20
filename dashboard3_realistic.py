# gann_short_expiry_realistic.py
"""
GANN Pro — SHORT Options Strategy (REALISTIC Pricing)
Entry: 15 minutes before close on GANN date
Exit: 15 minutes before close on EXPIRY day
FIXED: Uses more realistic Black-Scholes pricing model
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
from scipy.stats import norm

# ---------------------------
# App config + CSS
# ---------------------------
st.set_page_config(page_title="GANN SHORT Options (Realistic)", layout="wide", page_icon="📉")
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

st.markdown("<h2>📉 GANN SHORT Options (REALISTIC Pricing)</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Realistic Black-Scholes pricing | Entry: 3:15 PM on GANN | Exit: 3:15 PM on Expiry | Includes slippage & realistic P&L</div>", unsafe_allow_html=True)
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

def get_next_thursday(from_date):
    """Get next Thursday (weekly expiry)"""
    days_ahead = 3 - from_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return from_date + timedelta(days=days_ahead)

@st.cache_data(ttl=3600, show_spinner=False)
def yf_download_robust(ticker, start, end, max_retries=3):
    """Download data"""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
            if df is None or df.empty:
                st.warning(f"Attempt {attempt + 1}: No data")
                time.sleep(1)
                continue
            
            df = df.reset_index()
            if 'Datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['Datetime'])
            elif 'Date' not in df.columns:
                st.error(f"Date column missing")
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
            st.success(f"✓ Downloaded {len(df)} daily candles")
            return df
            
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(1)
            continue
    
    st.error(f"Failed to download")
    return pd.DataFrame()

# ---------------------------
# REALISTIC Black-Scholes Implementation
# ---------------------------
def black_scholes_call(S, K, T, r, sigma):
    """Realistic Black-Scholes for Call option"""
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call, 0)

def black_scholes_put(S, K, T, r, sigma):
    """Realistic Black-Scholes for Put option"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put, 0)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Greeks for risk assessment"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:  # put
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta}

# ---------------------------
# GANN generation
# ---------------------------
SPRING_EQ = (3, 21)

def generate_static_angles(years, angles):
    rows = []
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        for a in angles:
            offset = int(round((a/360.0)*365.25))
            rows.append({'GANN_Date': (base + timedelta(days=offset)), 'Type': f"{a}°", 'Source':'Angle'})
    return pd.DataFrame(rows)

def generate_equinox_solstice(years):
    mapping = {'Spring':(3,21),'Summer':(6,21),'Fall':(9,23),'Winter':(12,21)}
    rows = []
    for y in years:
        for name, (m, d) in mapping.items():
            rows.append({'GANN_Date': date(y, m, d), 'Type': name, 'Source':'Equinox'})
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
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type':f'P{c}d','Source':'Simple'})
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
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### Market")
    ticker_options = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "S&P 500": "^GSPC", "Nasdaq": "^IXIC"}
    selected_market = st.selectbox("Index", list(ticker_options.keys()), index=0)
    ticker = ticker_options[selected_market]
    
    st.markdown("### Dates")
    end_date = st.date_input("End", value=date.today())
    start_date = st.date_input("Start", value=end_date - relativedelta(months=6))
    
    st.markdown("### GANN")
    years = st.slider("Years", 2023, 2026, (2024, 2025))
    years_list = list(range(years[0], years[1]+1))
    
    angles_sel = st.multiselect("Angles", [30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], default=[45, 90, 180, 270])
    
    st.markdown("### Options")
    implied_vol = st.slider("IV (%)", 10, 100, 25) / 100
    otm_percent = st.slider("Strangle OTM %", 1, 10, 5)
    contracts = st.number_input("Contracts", min_value=1, max_value=100, value=1)
    lot_size = st.number_input("Lot size", min_value=1, max_value=200, value=50)
    
    st.markdown("### Realism Factors")
    st.info("🔧 **Slippage & Costs**")
    slippage_pct = st.slider("Exit slippage (%)", 0, 5, 1) / 100  # Realistic slippage
    transaction_cost = st.slider("Transaction cost (₹/lot)", 0, 500, 50)
    
    if st.button("🔄 Clear"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------
# Build GANN
# ---------------------------
st.markdown("---")
st.subheader("GANN Dates")

with st.spinner("Generating..."):
    gann_master = build_gann_master(years_list, angles_sel, ['simple'])
    st.success(f"✓ {len(gann_master)} GANN dates")

gann_in_range = gann_master[
    (gann_master['GANN_Date'] >= start_date) & 
    (gann_master['GANN_Date'] <= end_date)
].copy()

st.info(f"📊 {len(gann_in_range)} dates in range")

# ---------------------------
# Fetch data
# ---------------------------
st.markdown("---")
st.subheader("Market Data")

with st.spinner(f"Downloading {selected_market}..."):
    daily_df = yf_download_robust(ticker, start=start_date.strftime("%Y-%m-%d"), end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"))

if daily_df.empty:
    st.stop()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "SHORT Straddle", "SHORT Strangle", "Export"])

# Overview
with tab1:
    st.subheader("Strategy Overview")
    
    st.markdown("""
    ### ⚠️ Reality Check: Why 100% Win Rate is WRONG
    
    **Issues with naive pricing:**
    1. **Intrinsic value at expiry** - Options don't go to zero
    2. **Gap risk** - Market can gap up/down overnight
    3. **Realized volatility > IV** - Markets move more than expected
    4. **Transaction costs** - Slippage and commissions
    5. **Bid-ask spread** - Can't exit at exact theoretical price
    6. **Liquidity risk** - Large positions affect exit price
    """)
    
    st.markdown("### Example: Why SHORT Fails")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Naive Model (❌ Wrong):**
        - Entry: Sell 24000 Straddle
        - Credit: ₹300 (150 call + 150 put)
        - Exit: Buy back at ₹50
        - Profit: ₹250 (every time!)
        
        **Problem:** Ignores intrinsic value
        """)
    
    with col2:
        st.markdown("""
        **Realistic Model (✅ Correct):**
        - Entry: Sell 24000 Straddle  
        - Credit: ₹300
        - Market moves to 24500!
        - Call now worth: ₹450 (intrinsic)
        - Exit cost: ₹850+
        - Loss: -₹550 ❌
        
        **Lesson:** Large moves kill shorts
        """)

# SHORT Straddle
with tab2:
    st.subheader("SHORT Straddle (REALISTIC)")
    
    st.markdown(f"**IV:** {implied_vol*100:.0f}% | **Slippage:** {slippage_pct*100:.0f}% | **Cost:** ₹{transaction_cost}/lot")
    
    if st.button("▶️ Backtest (Realistic)", key="straddle"):
        with st.spinner("Backtesting with realistic pricing..."):
            results = []
            r = 0.06  # Risk-free rate
            
            for _, gann_row in gann_in_range.iterrows():
                gann_date = gann_row['GANN_Date']
                entry_data = daily_df[daily_df['Date'].dt.date == gann_date]
                if entry_data.empty:
                    continue
                
                entry_price = entry_data.iloc[0]['Close']
                atm_strike = round(entry_price / 50) * 50
                
                expiry_date = get_next_thursday(gann_date)
                exit_data = daily_df[daily_df['Date'].dt.date == expiry_date]
                if exit_data.empty:
                    continue
                
                exit_price = exit_data.iloc[0]['Close']
                days_held = max((expiry_date - gann_date).days, 1)
                T = days_held / 365.0
                
                # Entry: Realistic Black-Scholes
                entry_call = black_scholes_call(entry_price, atm_strike, T, r, implied_vol)
                entry_put = black_scholes_put(entry_price, atm_strike, T, r, implied_vol)
                entry_credit = (entry_call + entry_put) * contracts * lot_size
                
                # Exit: At expiry (intrinsic value)
                exit_call = max(exit_price - atm_strike, 0)  # Intrinsic at expiry
                exit_put = max(atm_strike - exit_price, 0)
                exit_cost = (exit_call + exit_put) * contracts * lot_size
                
                # Apply slippage (worse exit price)
                exit_cost = exit_cost * (1 + slippage_pct)
                
                # Subtract costs
                total_cost = transaction_cost * 2 * contracts  # 2 legs
                
                pnl = entry_credit - exit_cost - total_cost
                pnl_pct = (pnl / entry_credit) * 100 if entry_credit > 0 else 0
                
                results.append({
                    'GANN_Date': gann_date,
                    'Expiry': expiry_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Strike': atm_strike,
                    'Entry_Credit': entry_credit,
                    'Exit_Cost': exit_cost,
                    'Costs': total_cost,
                    'P&L': pnl,
                    'P&L_%': pnl_pct,
                    'Move_%': ((exit_price - entry_price) / entry_price) * 100
                })
            
            if results:
                results_df = pd.DataFrame(results)
                
                st.markdown("### 📊 Results (REALISTIC)")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_trades = len(results_df)
                winners = len(results_df[results_df['P&L'] > 0])
                losers = len(results_df[results_df['P&L'] < 0])
                win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
                
                total_pnl = results_df['P&L'].sum()
                avg_win = results_df[results_df['P&L'] > 0]['P&L'].mean() if winners > 0 else 0
                avg_loss = results_df[results_df['P&L'] < 0]['P&L'].mean() if losers > 0 else 0
                
                col1.metric("Trades", total_trades)
                col2.metric("Win Rate", f"{win_rate:.1f}% ⚠️")
                col3.metric("Total P&L", f"₹{total_pnl:,.0f}")
                col4.metric("Avg Win", f"₹{avg_win:,.0f}")
                col5.metric("Avg Loss", f"₹{avg_loss:,.0f}")
                
                st.warning(f"⚠️ **NOT 100%!** Realistic win rate: **{win_rate:.1f}%**")
                
                st.dataframe(results_df, use_container_width=True, height=300)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results_df['GANN_Date'],
                    y=results_df['P&L'],
                    marker_color=['green' if x > 0 else 'red' for x in results_df['P&L']]
                ))
                fig.update_layout(title="P&L Distribution (Realistic)", template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state['straddle'] = results_df
            else:
                st.warning("No data")

# SHORT Strangle
with tab3:
    st.subheader("SHORT Strangle (REALISTIC)")
    
    if st.button("▶️ Backtest (Realistic)", key="strangle"):
        with st.spinner("Backtesting..."):
            results = []
            r = 0.06
            
            for _, gann_row in gann_in_range.iterrows():
                gann_date = gann_row['GANN_Date']
                entry_data = daily_df[daily_df['Date'].dt.date == gann_date]
                if entry_data.empty:
                    continue
                
                entry_price = entry_data.iloc[0]['Close']
                call_strike = round((entry_price * (1 + otm_percent/100)) / 50) * 50
                put_strike = round((entry_price * (1 - otm_percent/100)) / 50) * 50
                
                expiry_date = get_next_thursday(gann_date)
                exit_data = daily_df[daily_df['Date'].dt.date == expiry_date]
                if exit_data.empty:
                    continue
                
                exit_price = exit_data.iloc[0]['Close']
                days_held = max((expiry_date - gann_date).days, 1)
                T = days_held / 365.0
                
                # Entry
                entry_call = black_scholes_call(entry_price, call_strike, T, r, implied_vol)
                entry_put = black_scholes_put(entry_price, put_strike, T, r, implied_vol)
                entry_credit = (entry_call + entry_put) * contracts * lot_size
                
                # Exit: Intrinsic
                exit_call = max(exit_price - call_strike, 0)
                exit_put = max(put_strike - exit_price, 0)
                exit_cost = (exit_call + exit_put) * contracts * lot_size
                
                # Slippage + Costs
                exit_cost = exit_cost * (1 + slippage_pct)
                total_cost = transaction_cost * 2 * contracts
                
                pnl = entry_credit - exit_cost - total_cost
                pnl_pct = (pnl / entry_credit) * 100 if entry_credit > 0 else 0
                
                results.append({
                    'GANN_Date': gann_date,
                    'Expiry': expiry_date,
                    'Entry': entry_price,
                    'Exit': exit_price,
                    'Call_K': call_strike,
                    'Put_K': put_strike,
                    'Credit': entry_credit,
                    'Cost': exit_cost,
                    'P&L': pnl,
                    'P&L_%': pnl_pct
                })
            
            if results:
                results_df = pd.DataFrame(results)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total = len(results_df)
                wins = len(results_df[results_df['P&L'] > 0])
                rate = (wins / total * 100) if total > 0 else 0
                
                col1.metric("Trades", total)
                col2.metric("Win Rate", f"{rate:.1f}%")
                col3.metric("Total P&L", f"₹{results_df['P&L'].sum():,.0f}")
                col4.metric("Avg W", f"₹{results_df[results_df['P&L']>0]['P&L'].mean():,.0f}")
                col5.metric("Avg L", f"₹{results_df[results_df['P&L']<0]['P&L'].mean():,.0f}")
                
                st.warning(f"⚠️ **Realistic: {rate:.1f}% win rate**")
                st.dataframe(results_df, use_container_width=True)
                st.session_state['strangle'] = results_df

# Export
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        if 'straddle' in st.session_state:
            csv = st.session_state['straddle'].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="straddle_{datetime.now().strftime("%Y%m%d")}.csv"><button>📥 Straddle</button></a>', unsafe_allow_html=True)
    with col2:
        if 'strangle' in st.session_state:
            csv = st.session_state['strangle'].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="strangle_{datetime.now().strftime("%Y%m%d")}.csv"><button>📥 Strangle</button></a>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94aace; font-size: 12px;'>
<p><strong>GANN SHORT Options - Realistic Pricing</strong></p>
<p>⚠️ Uses actual Black-Scholes + intrinsic at expiry + slippage. More realistic = lower win rate ✅</p>
</div>
""", unsafe_allow_html=True)
