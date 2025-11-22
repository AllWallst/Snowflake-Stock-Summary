import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Simply Clone", page_icon="❄️")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #1b222d; color: white; }
    h1, h2, h3 { color: #00d09c !important; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: white !important; }
    div[data-testid="stMetricLabel"] { color: #8c97a7 !important; }
    .css-1d391kg { background-color: #232b36; border-radius: 15px; padding: 20px; }
    hr { border-color: #36404e; }
    .stSelectbox label { color: #8c97a7; }
    </style>
""", unsafe_allow_html=True)

# --- URL & STATE MANAGEMENT ---
if "ticker" not in st.query_params:
    st.query_params["ticker"] = "AAPL"

url_ticker = st.query_params["ticker"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("❄️ Simply Clone")
    
    def update_url():
        st.query_params["ticker"] = st.session_state.ticker_input.upper()

    st.text_input("Enter Ticker", value=url_ticker, key="ticker_input", on_change=update_url)
    if st.button("Analyze"):
        update_url()
    
    st.divider()
    st.markdown("### ⚙️ Valuation Settings")
    val_method = st.radio("Select Valuation Method", ["Discounted Cash Flow (DCF)", "Graham Formula", "Analyst Target"])

ticker = st.query_params["ticker"]

# --- FETCH DATA ---
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    
    if not current_price:
        st.error("Ticker not found. Try a US stock like AAPL, MSFT, or GOOGL.")
        st.stop()
        
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- CALCULATION FUNCTIONS ---

# 1. GRAHAM NUMBER
def calc_graham(info):
    eps = info.get('trailingEps', 0)
    bv = info.get('bookValue', 0)
    if eps > 0 and bv > 0:
        return np.sqrt(22.5 * eps * bv)
    return 0

# 2. DISCOUNTED CASH FLOW (2-Stage)
def calc_dcf(stock, info):
    try:
        # Get Free Cash Flow (FCF)
        cashflow = stock.cash_flow
        if 'Free Cash Flow' in cashflow.index:
            fcf_latest = cashflow.loc['Free Cash Flow'].iloc[0]
        else:
            # Fallback: Operating Cash Flow - CapEx
            ocf = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
            capex = cashflow.loc['Capital Expenditures'].iloc[0]
            fcf_latest = ocf + capex # Capex is usually negative
            
        # Assumptions
        growth_rate = info.get('earningsGrowth', 0.10) # Default 10% if missing
        if growth_rate is None: growth_rate = 0.08
        
        # Cap unrealistic growth for safety (Simply Wall St does this too)
        growth_rate = min(growth_rate, 0.20) 
        
        discount_rate = 0.09  # 9% WACC (Standard assumption)
        terminal_growth = 0.025 # 2.5% (Inflation)
        shares = info.get('sharesOutstanding', 1)
        
        # Projection (5 Years)
        future_cash_flows = []
        for i in range(1, 6):
            fcf_val = fcf_latest * ((1 + growth_rate) ** i)
            discounted_val = fcf_val / ((1 + discount_rate) ** i)
            future_cash_flows.append(discounted_val)
            
        # Terminal Value
        last_fcf = fcf_latest * ((1 + growth_rate) ** 5)
        terminal_val = (last_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        discounted_terminal_val = terminal_val / ((1 + discount_rate) ** 5)
        
        total_value = sum(future_cash_flows) + discounted_terminal_val
        dcf_fair_value = total_value / shares
        
        return dcf_fair_value, growth_rate
    except:
        return 0, 0

# --- RUN CALCULATIONS ---
graham_fv = calc_graham(info)
dcf_fv, dcf_growth = calc_dcf(stock, info)
analyst_fv = info.get('targetMeanPrice', 0)

# Decide which to use based on Sidebar selection
if val_method == "Discounted Cash Flow (DCF)":
    fair_value = dcf_fv
    calc_desc = f"2-Stage DCF Model (Growth: {dcf_growth*100:.1f}%, Discount: 9%)"
elif val_method == "Graham Formula":
    fair_value = graham_fv
    calc_desc = "Graham Number (Sqrt(22.5 * EPS * Book Value))"
else:
    fair_value = analyst_fv
    calc_desc = "Average Analyst Price Target"

# Fallback if calculation fails (returns 0)
if fair_value == 0 or np.isnan(fair_value):
    fair_value = current_price
    calc_desc = "Data insufficient for calculation (Market Price used)"

# --- SNOWFLAKE SCORING (Updated with new data) ---
scores = {"Value": 0, "Future": 0, "Past": 0, "Health": 0, "Dividend": 0}

# Value (Using the selected Fair Value)
diff_percent = (fair_value - current_price) / current_price
if diff_percent > 0.20: scores["Value"] = 5    # >20% Undervalued
elif diff_percent > 0: scores["Value"] = 4     # Slightly Undervalued
elif diff_percent > -0.20: scores["Value"] = 3 # Fair
else: scores["Value"] = 1                      # Overvalued

# Future (PEG)
peg = info.get('pegRatio')
if peg is None:
    try: peg = info['trailingPE'] / (info['earningsGrowth']*100)
    except: peg = 0
if peg > 0 and peg < 1.5: scores["Future"] = 5
elif peg > 0 and peg < 3: scores["Future"] = 3
else: scores["Future"] = 2

# Past (ROE)
roe = info.get('returnOnEquity', 0)
if roe > 0.2: scores["Past"] = 5
elif roe > 0.1: scores["Past"] = 3
else: scores["Past"] = 1

# Health (Debt/Equity)
de = info.get('debtToEquity', 0)
if de < 50: scores["Health"] = 5
elif de < 100: scores["Health"] = 3
else: scores["Health"] = 1

# Dividend
dy = info.get('dividendYield', 0)
if dy and dy > 0.02: scores["Dividend"] = 5
elif dy: scores["Dividend"] = 3

# --- MAIN UI LAYOUT ---

col1, col2 = st.columns([2, 1])

with col1:
    # Ticker with Link Icon style
    st.markdown(f"""
        <h1>{info.get('shortName', ticker)} <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" style="text-decoration:none; font-size:0.5em;">🔗</a></h1>
        """, unsafe_allow_html=True)
    
    st.caption(f"{info.get('sector', 'Unknown')} | {info.get('industry', 'Unknown')}")
    st.write(info.get('longBusinessSummary', '')[:400] + "...")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${current_price:.2f}")
    m2.metric("Market Cap", f"${(info.get('marketCap',0)/1e9):.1f}B")
    m3.metric("Beta", f"{info.get('beta', 0):.2f}")
    m4.metric("PE Ratio", f"{info.get('trailingPE',0):.1f}")

with col2:
    # Snowflake Chart
    fig = go.Figure(data=go.Scatterpolar(
        r=[scores['Value'], scores['Future'], scores['Past'], scores['Health'], scores['Dividend']],
        theta=['Value', 'Future', 'Past', 'Health', 'Dividend'],
        fill='toself',
        line_color='#00d09c',
        fillcolor='rgba(0, 208, 156, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 5]), bgcolor='#232b36'),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        height=280
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 1. VALUATION SECTION ---
st.header("1. Valuation")

v_col_left, v_col_right = st.columns([2, 1])

with v_col_left:
    st.subheader("Share Price vs Fair Value")
    
    # Valuation Status Logic
    diff = ((current_price - fair_value) / fair_value) * 100
    status_color = "#ff6384" if diff > 20 else "#ffce56" if diff > -20 else "#00d09c"
    status_text = "Overvalued" if diff > 0 else "Undervalued"
    
    st.markdown(f"""
    The stock is trading at **${current_price}**. Our estimated Fair Value is **${fair_value:.2f}**.
    It appears to be <span style="color:{status_color}; font-weight:bold;">{abs(diff):.1f}% {status_text}</span>.
    <br><small style="color:#8c97a7">Calculation Method: {calc_desc}</small>
    """, unsafe_allow_html=True)
    
    # --- CUSTOM BAR VISUAL ---
    max_val = max(current_price, fair_value) * 1.3
    fig_val = go.Figure()

    # Zones
    fig_val.add_vrect(x0=0, x1=fair_value*0.8, fillcolor="#00d09c", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*0.8, x1=fair_value*1.2, fillcolor="#ffce56", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*1.2, x1=max_val*1.5, fillcolor="#ff6384", opacity=0.15, layer="below", line_width=0)

    # Current Price Bar
    fig_val.add_trace(go.Bar(
        y=[""], x=[current_price], name="Current Price", orientation='h',
        marker_color='#36a2eb', width=0.3,
        text=f"Current: ${current_price}", textposition='auto'
    ))

    # Fair Value Bar
    fig_val.add_trace(go.Bar(
        y=[""], x=[fair_value], name="Fair Value", orientation='h',
        marker_color='#232b36', marker_line_color='white', marker_line_width=2, width=0.3,
        text=f"Fair Value: ${fair_value:.2f}", textposition='auto'
    ))

    fig_val.update_layout(
        xaxis=dict(range=[0, max_val], visible=False),
        yaxis=dict(visible=False),
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=150,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False
    )
    st.plotly_chart(fig_val, use_container_width=True)
    
    st.caption("🟩 Undervalued (<20%) | 🟨 Fair Value | 🟥 Overvalued (>20%)")

with v_col_right:
    # Key Valuation Metrics
    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}")
    st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
    if graham_fv > 0:
        st.metric("Graham Number", f"${graham_fv:.2f}", help="Sqrt(22.5 * EPS * BookValue)")
    else:
        st.metric("Graham Number", "N/A")

st.divider()

# --- 2. FUTURE & PAST ---
c_fut, c_past = st.columns(2)

with c_fut:
    st.header("2. Future Growth")
    st.metric("Analyst Growth Est.", f"{info.get('earningsGrowth', 0)*100:.1f}%")
    st.write("Forecasted annual earnings growth.")

with c_past:
    st.header("3. Past Performance")
    st.metric("ROE", f"{roe*100:.1f}%")
    st.metric("ROA", f"{info.get('returnOnAssets', 0)*100:.1f}%")

# --- 4. HEALTH ---
st.divider()
st.header("4. Financial Health")
h1, h2 = st.columns([2, 1])

with h1:
    cash = info.get('totalCash', 0)
    debt = info.get('totalDebt', 0)
    
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(
        x=['Cash', 'Debt'], 
        y=[cash, debt], 
        marker_color=['#00d09c', '#ff6384'], 
        # Format the hover text to Billions
        text=[f"${cash/1e9:.1f}B", f"${debt/1e9:.1f}B"], 
        textposition='auto'
    ))
    fig_h.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='white'),
        margin=dict(t=0, b=0, l=0, r=0),
        height=200
    )
    st.plotly_chart(fig_h, use_container_width=True)

with h2:
    # Get the raw value (e.g., 150.5)
    de_ratio = info.get('debtToEquity', 0)
    
    # Display it with the % symbol
    st.metric("Debt to Equity", f"{de_ratio:.1f}%")
    
    # Add a helpful text explanation
    if de_ratio > 100:
        st.error(f"⚠️ High Debt. Liabilities are {de_ratio:.0f}% of Equity.")
    else:
        st.success(f"✅ Healthy. Debt is only {de_ratio:.0f}% of Equity.")
        
    if cash > debt: 
        st.caption("✅ Cash covers total debt.")
    else: 
        st.caption("⚠️ Debt exceeds total cash.")
        
# --- 5. DIVIDEND ---
st.divider()
st.header("5. Dividend")
d1, d2 = st.columns(2)
with d1:
    st.metric("Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
with d2:
    st.metric("Payout Ratio", f"{info.get('payoutRatio', 0)*100:.0f}%")

