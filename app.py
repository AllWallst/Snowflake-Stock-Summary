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
    </style>
""", unsafe_allow_html=True)

# --- URL & STATE MANAGEMENT ---
# 1. Read the URL to see if a ticker is already there (e.g. ?ticker=GOOGL)
if "ticker" not in st.query_params:
    st.query_params["ticker"] = "AAPL" # Default if empty

url_ticker = st.query_params["ticker"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("❄️ Simply Clone")
    
    # 2. Define a function that updates the URL when you type
    def update_url():
        # Get value from input, uppercase it, save to URL
        st.query_params["ticker"] = st.session_state.ticker_input.upper()

    # 3. The Input Box
    # We bind this to 'update_url' so it runs immediately when you hit Enter
    st.text_input(
        "Enter Ticker", 
        value=url_ticker, 
        key="ticker_input", 
        on_change=update_url
    )
    
    # Button acts as a backup trigger
    if st.button("Analyze"):
        update_url()

# 4. Set the variable for the rest of the app
ticker = st.query_params["ticker"]

# --- FETCH DATA (Rest of your code remains exactly the same below) ---
try:
    stock = yf.Ticker(ticker)
    # ... keep the rest of your existing code here ...

# --- FETCH DATA ---
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Check validity
    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    if not current_price:
        st.error("Ticker not found or data unavailable. Try a US stock like AAPL or MSFT.")
        st.stop()
        
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- DATA CLEANING & MANUAL CALCULATIONS ---
# yfinance often returns None for PEG/ROE, so we calculate manually if needed

# 1. Get/Calculate ROE
roe = info.get('returnOnEquity')
if roe is None:
    # Try calculating: Net Income / Total Equity
    try:
        bs = stock.balance_sheet
        inc = stock.financials
        if not bs.empty and not inc.empty:
            total_equity = bs.loc['Stockholders Equity'].iloc[0]
            net_income = inc.loc['Net Income'].iloc[0]
            roe = net_income / total_equity
        else:
            roe = 0
    except:
        roe = 0

# 2. Get/Calculate PEG
peg = info.get('pegRatio')
pe_ratio = info.get('trailingPE')
growth_est = info.get('earningsGrowth') # approx proxy
if peg is None and pe_ratio and growth_est:
    # PEG = PE / (Growth Rate * 100)
    try:
        peg = pe_ratio / (growth_est * 100)
    except:
        peg = 0
elif peg is None:
    peg = 0

# 3. Calculate "Fair Value" (Graham Number Proxy)
# Simply Wall St uses DCF. We will use Graham Number (Sqrt(22.5 * EPS * BookVal)) 
# or Analyst Target as a fallback for the visual.
eps = info.get('trailingEps')
book_val = info.get('bookValue')

fair_value = 0
fv_method = "Analyst Target"

# Try Graham Number first (Classic Value Formula)
if eps and book_val and eps > 0 and book_val > 0:
    fair_value = np.sqrt(22.5 * eps * book_val)
    fv_method = "Graham Formula"
    
# If Graham fails (negative earnings), use Analyst Target
if fair_value == 0 or np.isnan(fair_value):
    fair_value = info.get('targetMeanPrice', current_price)
    fv_method = "Analyst Target"

# --- SNOWFLAKE SCORE CALC ---
scores = {"Value": 0, "Future": 0, "Past": 0, "Health": 0, "Dividend": 0}

# Value
if pe_ratio:
    if pe_ratio < 15: scores["Value"] = 5
    elif pe_ratio < 30: scores["Value"] = 3
    else: scores["Value"] = 1

# Future (PEG)
if peg > 0 and peg < 1.5: scores["Future"] = 5
elif peg > 0 and peg < 3.0: scores["Future"] = 3
elif peg > 0: scores["Future"] = 2
else: scores["Future"] = 1 # Negative or N/A

# Past (ROE)
if roe > 0.20: scores["Past"] = 5
elif roe > 0.10: scores["Past"] = 3
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

# --- LAYOUT START ---

col1, col2 = st.columns([2, 1])

with col1:
    st.title(f"{info.get('shortName', ticker)}")
    st.caption(f"{info.get('sector', 'Unknown')} | {info.get('industry', 'Unknown')}")
    st.write(info.get('longBusinessSummary', '')[:350] + "...")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${current_price:.2f}")
    m2.metric("Market Cap", f"${(info.get('marketCap',0)/1e9):.1f}B")
    m3.metric("Beta", f"{info.get('beta', 0):.2f}")
    m4.metric("PE Ratio", f"{pe_ratio:.1f}" if pe_ratio else "N/A")

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
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 1. VALUATION (WITH NEW VISUAL) ---
st.header("1. Valuation")

# Create the "Share Price vs Fair Value" Chart
val_col1, val_col2 = st.columns([2,1])

with val_col1:
    st.subheader("Share Price vs Fair Value")
    
    # Logic for Undervalued/Overvalued text
    diff = ((current_price - fair_value) / fair_value) * 100
    status = "Overvalued" if diff > 0 else "Undervalued"
    color_status = "#ff6384" if diff > 0 else "#00d09c" # Red or Green
    
    st.markdown(f"""
    The stock is trading at **${current_price}**. Our estimated Fair Value is **${fair_value:.2f}**.
    It appears to be <span style="color:{color_status}; font-weight:bold;">{abs(diff):.1f}% {status}</span>.
    <small>(Calculation Method: {fv_method})</small>
    """, unsafe_allow_html=True)

    # BUILD THE CUSTOM VISUAL
    max_range = max(current_price, fair_value) * 1.4
    
    fig_fv = go.Figure()

    # 1. Background Zones (Green/Yellow/Red)
    # 20% Undervalued Zone
    fig_fv.add_vrect(x0=0, x1=fair_value * 0.8, fillcolor="#00d09c", opacity=0.2, layer="below", line_width=0)
    # Fair Zone
    fig_fv.add_vrect(x0=fair_value * 0.8, x1=fair_value * 1.2, fillcolor="#ffce56", opacity=0.2, layer="below", line_width=0)
    # Overvalued Zone
    fig_fv.add_vrect(x0=fair_value * 1.2, x1=max_range * 1.5, fillcolor="#ff6384", opacity=0.2, layer="below", line_width=0)

    # 2. Bar for Current Price
    fig_fv.add_trace(go.Bar(
        y=["Price"], x=[current_price], 
        name="Current Price", orientation='h',
        marker_color='#1f77b4', # Simply Wall St Blue
        text=f"Current: ${current_price}", textposition='auto'
    ))

    # 3. Bar for Fair Value
    fig_fv.add_trace(go.Bar(
        y=["Value"], x=[fair_value], 
        name="Fair Value", orientation='h',
        marker_color='#232b36', # Dark bar
        marker_line_color='white', marker_line_width=2,
        text=f"Fair Value: ${fair_value:.2f}", textposition='auto'
    ))

    fig_fv.update_layout(
        barmode='group',
        xaxis=dict(range=[0, max_range], showgrid=False, visible=False),
        yaxis=dict(showgrid=False, showticklabels=False), # Hide labels, bars explain themselves
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=0, r=0),
        height=180,
        showlegend=False
    )
    
    st.plotly_chart(fig_fv, use_container_width=True)
    
    # Legend for the zones
    st.caption("🟩 Undervalued (<20%) | 🟨 Fair Value | 🟥 Overvalued (>20%)")

with val_col2:
    st.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio else "N/A")
    st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A", delta_color="inverse")
    st.metric("Fair Value", f"${fair_value:.2f}")

st.divider()

# --- 2. FUTURE & PAST ---
col_fut, col_past = st.columns(2)

with col_fut:
    st.header("2. Future Growth")
    # Fix: PEG N/A Issue - Handled in calculation section above
    st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
    if peg and peg < 1:
        st.success("PEG < 1: Stock is potentially undervalued based on growth.")
    elif peg:
        st.warning("PEG > 1: Stock price assumes high growth.")
        
    st.metric("Analyst Growth Est.", f"{info.get('earningsGrowth', 0)*100:.1f}%")

with col_past:
    st.header("3. Past Performance")
    # Fix: ROE Missing - Handled in calculation section above
    st.metric("Return on Equity (ROE)", f"{roe*100:.1f}%")
    if roe > 0.20:
        st.success("High ROE (>20%) indicates efficient management.")
        
    st.metric("Return on Assets (ROA)", f"{info.get('returnOnAssets', 0)*100:.1f}%")

# --- 4. FINANCIAL HEALTH CHART ---
st.divider()
st.header("4. Financial Health")
st.subheader("Debt vs Equity")

h_chart_col, h_data_col = st.columns([2,1])

with h_chart_col:
    # Create Debt/Equity Bar Chart
    total_debt = info.get('totalDebt', 0)
    total_cash = info.get('totalCash', 0)
    
    # Handle case where debt/equity might be missing from info
    # using balance sheet fallback
    if not total_debt:
        try: total_debt = stock.balance_sheet.loc['Total Debt'].iloc[0]
        except: total_debt = 0
        
    # Visualize
    fig_health = go.Figure()
    fig_health.add_trace(go.Bar(
        x=['Cash', 'Debt'],
        y=[total_cash, total_debt],
        marker_color=['#00d09c', '#ff6384'],
        text=[f"${total_cash/1e9:.1f}B", f"${total_debt/1e9:.1f}B"],
        textposition='auto'
    ))
    fig_health.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='#36404e'),
        font=dict(color='white')
    )
    st.plotly_chart(fig_health, use_container_width=True)

with h_data_col:
    st.metric("Debt to Equity", f"{info.get('debtToEquity', 'N/A')}")
    if total_cash > total_debt:
        st.success("✅ More Cash than Debt")
    else:
        st.error("⚠️ More Debt than Cash")

st.divider()

# --- 5. MANAGEMENT & DIVIDENDS ---
mgt_col, div_col = st.columns(2)

with mgt_col:
    st.header("Management")
    officers = info.get('companyOfficers', [])
    if officers:
        st.write(f"**CEO:** {officers[0].get('name', 'N/A')}")
        st.write(f"**Pay:** ${officers[0].get('totalPay', 0):,}" if officers[0].get('totalPay') else "N/A")
    else:
        st.write("Data unavailable")

with div_col:
    st.header("Dividend")
    yield_val = info.get('dividendYield', 0)
    payout = info.get('payoutRatio', 0)
    
    st.metric("Yield", f"{yield_val*100:.2f}%" if yield_val else "0%")
    st.metric("Payout Ratio", f"{payout*100:.0f}%" if payout else "N/A")
    
    # Donut chart for payout
    if payout:
        fig_div = go.Figure(data=[go.Pie(
            labels=['Payout', 'Retained'], 
            values=[payout, 1-payout], 
            hole=.7,
            marker=dict(colors=['#00d09c', '#2c3542'])
        )])
        fig_div.update_layout(
            showlegend=False, 
            height=150, 
            margin=dict(t=0,b=0,l=0,r=0),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_div, use_container_width=True)

