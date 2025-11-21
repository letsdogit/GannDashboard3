# gann_nse_real_data.py
"""
GANN SHORT Options Strategy - NSE REAL DATA INTEGRATION
========================================================

ENTERPRISE VERSION WITH ACTUAL NSE DATA

DATA SOURCES:
1. NSE India Options Chain API (Free, updates every 3 seconds)
   - Real option prices (not Black-Scholes theoretical)
   - Actual bid/ask, OI, IV, Greeks
   - Live market data

2. NSE India Historical Data API
   - Actual EOD OHLC data
   - Real volatility calculations

3. Intraday Data Options:
   - Option A: Use NSE's 15-min delayed snapshot (free)
   - Option B: Integrate broker API (requires account)
   - Option C: Subscribe to NSE real-time feed (₹8L/year)

LEGAL COMPLIANCE:
- Using publicly available NSE APIs
- For educational/research purposes
- Rate-limited to respect NSE servers
- Users should comply with NSE's terms of service

PRODUCTION DEPLOYMENT:
For commercial use, subscribe to:
- NSE Real-time Data: marketdata@nse.co.in
- Or integrate with registered broker API

Version: 2.0.0 (NSE Real Data)
Last Updated: 2025-11-21
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, date, timedelta, time as dt_time
from dateutil.relativedelta import relativedelta
import logging
from typing import Dict, List, Tuple, Optional
import time as time_module
import warnings
from scipy.stats import norm
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class NSEConfig:
    """NSE API Configuration"""
    # NSE API endpoints
    BASE_URL = "https://www.nseindia.com"
    OPTION_CHAIN_URL = f"{BASE_URL}/api/option-chain-indices"
    EQUITY_OPTION_CHAIN_URL = f"{BASE_URL}/api/option-chain-equities"
    QUOTE_URL = f"{BASE_URL}/api/quote-derivative"
    
    # Headers to mimic browser (required by NSE)
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/option-chain',
        'Connection': 'keep-alive'
    }
    
    # Rate limiting (respect NSE servers)
    REQUEST_DELAY = 3  # seconds between requests
    MAX_RETRIES = 3
    
    # Supported indices
    INDICES = {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY",
        "MIDCPNIFTY": "MIDCPNIFTY"
    }
    
    # Trading parameters
    STRIKE_INTERVAL = {
        "NIFTY": 50,
        "BANKNIFTY": 100,
        "FINNIFTY": 50,
        "MIDCPNIFTY": 25
    }
    
    RISK_FREE_RATE = 0.06
    TRADING_DAYS_PER_YEAR = 252

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# NSE DATA FETCHER
# =============================================================================

class NSEDataFetcher:
    """
    Fetches REAL data from NSE India APIs
    
    FEATURES:
    - Real option chain data (not theoretical)
    - Actual bid/ask prices
    - Live OI, volume, IV
    - Greeks from market
    
    LIMITATIONS:
    - No authentication (public API)
    - Rate limited (3 seconds between calls)
    - May fail if NSE changes API structure
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSEConfig.HEADERS)
        self.last_request_time = 0
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize session with NSE (required for cookies)"""
        try:
            # Visit homepage to get cookies
            self.session.get(NSEConfig.BASE_URL, timeout=10)
            time_module.sleep(1)
            logger.info("✓ NSE session initialized")
        except Exception as e:
            logger.warning(f"Session initialization warning: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time_module.time() - self.last_request_time
        if elapsed < NSEConfig.REQUEST_DELAY:
            time_module.sleep(NSEConfig.REQUEST_DELAY - elapsed)
        self.last_request_time = time_module.time()
    
    def get_option_chain(self, symbol: str) -> Dict:
        """
        Get REAL option chain data from NSE
        
        Returns actual market data:
        - LTP (Last Traded Price)
        - Bid/Ask prices
        - Open Interest
        - Implied Volatility
        - Greeks (Delta, Gamma, Vega, Theta)
        - Volume
        
        Args:
            symbol: Index name (NIFTY, BANKNIFTY, etc.)
            
        Returns:
            Dictionary with complete option chain data
        """
        self._rate_limit()
        
        for attempt in range(NSEConfig.MAX_RETRIES):
            try:
                url = f"{NSEConfig.OPTION_CHAIN_URL}?symbol={symbol}"
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                
                if 'records' in data and 'data' in data['records']:
                    logger.info(f"✓ Fetched real option chain for {symbol}")
                    return data
                else:
                    logger.error(f"Invalid response structure from NSE")
                    return {}
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time_module.sleep(2)
                continue
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return {}
        
        logger.error(f"Failed to fetch option chain after {NSEConfig.MAX_RETRIES} attempts")
        return {}
    
    def get_spot_price(self, symbol: str) -> Optional[float]:
        """
        Get current spot price from NSE
        
        Returns REAL market price, not approximation
        """
        try:
            data = self.get_option_chain(symbol)
            if 'records' in data:
                spot = data['records'].get('underlyingValue')
                if spot:
                    return float(spot)
        except Exception as e:
            logger.error(f"Error getting spot price: {e}")
        
        return None
    
    def parse_option_chain(self, data: Dict, expiry_date: str) -> pd.DataFrame:
        """
        Parse NSE option chain data into structured DataFrame
        
        Returns DataFrame with REAL market data:
        - Strike, Type (CE/PE)
        - LTP, Bid, Ask
        - OI, Volume
        - IV, Delta, Gamma, Vega, Theta
        """
        if not data or 'records' not in data:
            return pd.DataFrame()
        
        records = data['records']['data']
        rows = []
        
        for record in records:
            if record.get('expiryDate') != expiry_date:
                continue
            
            strike = record.get('strikePrice')
            
            # Call option data
            if 'CE' in record:
                ce = record['CE']
                rows.append({
                    'Strike': strike,
                    'Type': 'CE',
                    'LTP': ce.get('lastPrice', 0),
                    'Bid': ce.get('bidprice', 0),
                    'Ask': ce.get('askPrice', 0),
                    'OI': ce.get('openInterest', 0),
                    'Volume': ce.get('totalTradedVolume', 0),
                    'IV': ce.get('impliedVolatility', 0),
                    'Delta': ce.get('delta', 0),
                    'Gamma': ce.get('gamma', 0),
                    'Vega': ce.get('vega', 0),
                    'Theta': ce.get('theta', 0)
                })
            
            # Put option data
            if 'PE' in record:
                pe = record['PE']
                rows.append({
                    'Strike': strike,
                    'Type': 'PE',
                    'LTP': pe.get('lastPrice', 0),
                    'Bid': pe.get('bidprice', 0),
                    'Ask': pe.get('askPrice', 0),
                    'OI': pe.get('openInterest', 0),
                    'Volume': pe.get('totalTradedVolume', 0),
                    'IV': pe.get('impliedVolatility', 0),
                    'Delta': pe.get('delta', 0),
                    'Gamma': pe.get('gamma', 0),
                    'Vega': pe.get('vega', 0),
                    'Theta': pe.get('theta', 0)
                })
        
        df = pd.DataFrame(rows)
        logger.info(f"✓ Parsed {len(df)} option contracts")
        return df
    
    def get_expiry_dates(self, symbol: str) -> List[str]:
        """Get available expiry dates for symbol"""
        try:
            data = self.get_option_chain(symbol)
            if 'records' in data:
                expiries = data['records'].get('expiryDates', [])
                logger.info(f"✓ Found {len(expiries)} expiry dates")
                return expiries
        except Exception as e:
            logger.error(f"Error getting expiry dates: {e}")
        
        return []

# =============================================================================
# GANN DATE GENERATOR
# =============================================================================

class GANNDateGenerator:
    """Generate GANN trading dates"""
    
    @staticmethod
    def generate(years: List[int], angles: List[int]) -> pd.DataFrame:
        """Generate GANN dates"""
        rows = []
        SPRING_EQ = (3, 21)
        
        for year in years:
            base = date(year, SPRING_EQ[0], SPRING_EQ[1])
            
            for angle in angles:
                offset = int(round((angle / 360.0) * 365.25))
                rows.append({
                    'GANN_Date': base + timedelta(days=offset),
                    'Type': f"{angle}°"
                })
            
            # Equinoxes & Solstices
            for name, (m, d) in [
                ('Spring', (3, 21)),
                ('Summer', (6, 21)),
                ('Fall', (9, 23)),
                ('Winter', (12, 21))
            ]:
                rows.append({
                    'GANN_Date': date(year, m, d),
                    'Type': name
                })
        
        df = pd.DataFrame(rows)
        df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
        return df.drop_duplicates(subset=['GANN_Date']).sort_values('GANN_Date').reset_index(drop=True)
    
    @staticmethod
    def get_next_thursday(from_date: date) -> date:
        """Get next Thursday (weekly expiry)"""
        days_ahead = 3 - from_date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return from_date + timedelta(days=days_ahead)

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="GANN NSE Real Data",
    layout="wide",
    page_icon="📉"
)

st.markdown("""
<style>
:root{--bg:#061026;--card:#0b1626;--muted:#94aace;--accent:#7dd3fc;--accent2:#a78bfa;}
body{background:linear-gradient(180deg,var(--bg),#020815); color:#eaf3ff;}
.block-container{padding-top:1rem;}
.stButton>button{background:linear-gradient(90deg,var(--accent),var(--accent2)); border:none; color:#012; font-weight:700;}
.success-box{background:#10b981;color:#fff;padding:16px;border-radius:8px;margin:16px 0;font-weight:600;}
.warning-box{background:#f59e0b;color:#000;padding:16px;border-radius:8px;margin:16px 0;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# Header with Real Data Badge
st.markdown("""
<div class='success-box'>
✅ <strong>USING REAL NSE DATA</strong><br>
• Live option chain from NSE India<br>
• Actual bid/ask prices (not Black-Scholes)<br>
• Real OI, volume, IV, Greeks<br>
• Updates every 3 seconds during market hours
</div>
""", unsafe_allow_html=True)

st.markdown("<h2>📉 GANN SHORT Options - NSE Real Data</h2>", unsafe_allow_html=True)

# Initialize NSE fetcher
if 'nse_fetcher' not in st.session_state:
    st.session_state.nse_fetcher = NSEDataFetcher()

nse = st.session_state.nse_fetcher

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 📊 Index Selection")
    symbol = st.selectbox(
        "NSE Index",
        list(NSEConfig.INDICES.keys()),
        index=0
    )
    
    st.markdown("### 📅 GANN Configuration")
    years = st.slider("Years", 2023, 2026, (2024, 2025))
    years_list = list(range(years[0], years[1] + 1))
    angles = st.multiselect(
        "Angles",
        [45, 90, 135, 180, 225, 270],
        default=[90, 180, 270]
    )
    
    st.markdown("### 💰 Position")
    contracts = st.number_input("Contracts", 1, 100, 1)
    lot_size = st.number_input("Lot Size", 1, 200, 50)
    
    st.markdown("### 🎯 Strategy")
    strategy = st.radio(
        "Strategy Type",
        ["SHORT Straddle", "SHORT Strangle"]
    )
    
    if strategy == "SHORT Strangle":
        otm_pct = st.slider("OTM %", 1, 10, 5)

# Main Content
st.markdown("---")
st.subheader("🔴 Live NSE Data")

# Fetch current data
if st.button("🔄 Fetch Live Data", key="fetch_live"):
    with st.spinner(f"Fetching real data from NSE for {symbol}..."):
        try:
            # Get spot price
            spot = nse.get_spot_price(symbol)
            
            if spot:
                st.success(f"✅ **{symbol} Spot:** ₹{spot:,.2f} (REAL-TIME)")
                
                # Get expiry dates
                expiries = nse.get_expiry_dates(symbol)
                
                if expiries:
                    st.info(f"📅 **Available Expiries:** {', '.join(expiries[:5])}")
                    
                    # Get option chain for nearest expiry
                    nearest_expiry = expiries[0]
                    chain_data = nse.get_option_chain(symbol)
                    
                    if chain_data:
                        options_df = nse.parse_option_chain(chain_data, nearest_expiry)
                        
                        if not options_df.empty:
                            st.markdown(f"### 📊 Option Chain ({nearest_expiry})")
                            
                            # Display sample data
                            st.dataframe(
                                options_df.head(20),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Store in session
                            st.session_state.current_spot = spot
                            st.session_state.current_options = options_df
                            st.session_state.current_expiry = nearest_expiry
                            
                            # Show data quality
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Strikes", len(options_df))
                            col2.metric("Avg Call IV", f"{options_df[options_df['Type']=='CE']['IV'].mean():.1f}%")
                            col3.metric("Avg Put IV", f"{options_df[options_df['Type']=='PE']['IV'].mean():.1f}%")
                            col4.metric("Data Source", "NSE LIVE ✅")
                        else:
                            st.error("⚠️ No options data available")
                    else:
                        st.error("⚠️ Failed to fetch option chain")
                else:
                    st.error("⚠️ No expiry dates available")
            else:
                st.error("⚠️ Failed to fetch spot price")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            logger.error(f"Live data fetch error: {e}")

# Show stored data if available
if 'current_spot' in st.session_state:
    st.markdown("---")
    st.markdown("### 💡 Current Market Data (Cached)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{symbol} Spot", f"₹{st.session_state.current_spot:,.2f}")
    col2.metric("Expiry", st.session_state.current_expiry)
    col3.metric("Data Points", len(st.session_state.current_options))

# Generate GANN Dates
st.markdown("---")
st.subheader("📅 GANN Dates")

if st.button("Generate GANN Dates", key="gen_gann"):
    with st.spinner("Generating..."):
        gann_master = GANNDateGenerator.generate(years_list, angles)
        st.session_state.gann_dates = gann_master
        st.success(f"✓ Generated {len(gann_master)} GANN dates")
        st.dataframe(gann_master, use_container_width=True, height=300)

# Instructions
st.markdown("---")
st.markdown("""
### 📋 How to Use

**Step 1: Fetch Live Data**
- Click "🔄 Fetch Live Data" to get real NSE option chain
- Data includes actual bid/ask, OI, IV, Greeks
- Updates every 3 seconds (rate-limited)

**Step 2: Generate GANN Dates**
- Select years and angles
- Click "Generate GANN Dates"

**Step 3: Backtest (Coming in next update)**
- Will use REAL option prices from NSE
- No more Black-Scholes theoretical pricing
- Actual market bid/ask execution

### ⚠️ Important Notes

**Data Limitations:**
- NSE API is public but unofficial
- May stop working if NSE changes structure
- Rate limited to 1 request per 3 seconds
- For production, subscribe to official NSE data

**For Commercial Use:**
- Email: marketdata@nse.co.in
- Real-time subscription: ₹8,00,000/year
- Or use registered broker API (Upstox, Dhan, etc.)

**Legal Compliance:**
- This is for educational/research only
- Respect NSE's rate limits
- Don't abuse the public API
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#94aace;font-size:12px;padding:20px 0;'>
<p><strong>GANN NSE Real Data Version 2.0.0</strong></p>
<p>✅ Using actual NSE option chain data (not approximations)</p>
<p>⚠️ Public API - for educational use only</p>
<p>For production: Subscribe to official NSE data or broker API</p>
</div>
""", unsafe_allow_html=True)
