# gann_options_strategy_dashboard.py
"""
GANN Pro — Options Strategy Dashboard
Focus: Long Straddle & Long Strangle strategies on GANN dates
Entry: 5 minutes after market open on GANN dates
Exit: 15 minutes before market close
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
st.set_page_config(page_title="GANN Options Strategy", layout="wide", page_icon="📈")
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

st.markdown("<h2>📈 GANN Options Strategy Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Long Straddle & Long Strangle strategies on GANN dates | Entry: 5min after open | Exit: 15min before close</div>", unsafe_allow_html=True)
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
def yf_downl
