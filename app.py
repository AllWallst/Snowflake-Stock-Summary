import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="MarketRadar", page_icon="📡")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #1b222d; color: white; }
    h1, h2, h3 { color: #00d09c !important; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: white !important; }
    div[data-testid="stMetricLabel"] { color: #8c97a7 !important; }
    .css-1d391kg { background-color: #232b36; border-radius: 15px; padding: 20px; }
    hr { border-color: #36404e; }
    .news-card { background-color: #232b36; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #00d09c; }
    .news-title { color: white; font-weight: bold; font-size: 1.1em; text-decoration: none; }
    .news-meta { color: #8c97a7; font-size: 0.85em; }
    </style>
""", unsafe_allow_html=True)

# --- URL & STATE MANAGEMENT ---
if "ticker" not in st.query_params:
    st.query_params["ticker"] = "AAPL"

url_ticker = st.query_params["ticker"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("📡 MarketRadar")
    
    def update_url():
        st.query_params["ticker"] = st.session_state.ticker_input.upper()

    st.text_input("Enter Ticker", value=url_ticker, key="ticker_input", on_change=update_url)
    if st.button("Analyze"):
        update_url()
    
    st.divider()
    st.markdown("### ⚙️ Valuation Settings")
    val_method = st.radio("Fair Value Method", ["Discounted Cash Flow (DCF)", "Graham Formula", "Analyst Target"])

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

def calc_graham(info):
    eps = info.get('trailingEps', 0)
    bv = info.get('bookValue', 0)
    if eps > 0 and bv > 0:
        return np.sqrt(22.5 * eps * bv)
    return 0

def calc_dcf(stock, info):
    try:
        cashflow = stock.cash_flow
        if 'Free Cash Flow' in cashflow.index:
            fcf_latest = cashflow.loc['Free Cash Flow'].iloc[0]
        else:
            ocf = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
            capex = cashflow.loc['Capital Expenditures'].iloc[0]
            fcf_latest = ocf + capex 
            
        growth_rate = info.get('earningsGrowth', 0.10) 
        if growth_rate is None: growth_rate = 0.08
        growth_rate = min(growth_rate, 0.20) 
        
        discount_rate = 0.09 
        terminal_growth = 0.025 
        shares = info.get('sharesOutstanding', 1)
        
        future_cash_flows = []
        for i in range(1, 6):
            fcf_val = fcf_latest * ((1 + growth_rate) ** i)
            discounted_val = fcf_val / ((1 + discount_rate) ** i)
            future_cash_flows.append(discounted_val)
            
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

if val_method == "Discounted Cash Flow (DCF)":
    fair_value = dcf_fv
    calc_desc = f"2-Stage DCF Model (Growth: {dcf_growth*100:.1f}%, Discount: 9%)"
elif val_method == "Graham Formula":
    fair_value = graham_fv
    calc_desc = "Graham Number (Sqrt(22.5 * EPS * Book Value))"
else:
    fair_value = analyst_fv
    calc_desc = "Average Analyst Price Target"

if fair_value == 0 or np.isnan(fair_value):
    fair_value = current_price
    calc_desc = "Data insufficient (Market Price used)"

# --- SNOWFLAKE SCORING ---
scores = {"Value": 0, "Future": 0, "Past": 0, "Health": 0, "Dividend": 0}

diff_percent = (fair_value - current_price) / current_price
if diff_percent > 0.20: scores["Value"] = 5
elif diff_percent > 0: scores["Value"] = 4
elif diff_percent > -0.20: scores["Value"] = 3
else: scores["Value"] = 1

peg = info.get('pegRatio')
if peg is None:
    try: peg = info['trailingPE'] / (info['earningsGrowth']*100)
    except: peg = 0
if peg > 0 and peg < 1.5: scores["Future"] = 5
elif peg > 0 and peg < 3: scores["Future"] = 3
else: scores["Future"] = 2

roe = info.get('returnOnEquity', 0)
if roe > 0.2: scores["Past"] = 5
elif roe > 0.1: scores["Past"] = 3
else: scores["Past"] = 1

de = info.get('debtToEquity', 0)
if de < 50: scores["Health"] = 5
elif de < 100: scores["Health"] = 3
else: scores["Health"] = 1

dy = info.get('dividendYield', 0)
if dy and dy > 0.02: scores["Dividend"] = 5
elif dy: scores["Dividend"] = 3

# --- HEADER UI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""<h1>{info.get('shortName', ticker)} <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" style="text-decoration:none; font-size:0.5em;">🔗</a></h1>""", unsafe_allow_html=True)
    st.caption(f"{info.get('sector', 'Unknown')} | {info.get('industry', 'Unknown')}")
    st.write(info.get('longBusinessSummary', '')[:400] + "...")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${current_price:.2f}")
    m2.metric("Market Cap", f"${(info.get('marketCap',0)/1e9):.1f}B")
    m3.metric("Beta", f"{info.get('beta', 0):.2f}")
    m4.metric("PE Ratio", f"{info.get('trailingPE',0):.1f}")

with col2:
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

# --- INTERACTIVE PRICE HISTORY CHART ---
st.header("Price History")
# Fetch historical data (Max period to support all zooms)
# Note: yfinance "max" is daily data.
hist_data = stock.history(period="max")

if not hist_data.empty:
    fig_price = go.Figure()
    
    # Main Line
    fig_price.add_trace(go.Scatter(
        x=hist_data.index, 
        y=hist_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#00d09c', width=2),
        fill='tozeroy', # Area chart style
        fillcolor='rgba(0, 208, 156, 0.1)' 
    ))

    # Range Selector Buttons
    fig_price.update_xaxes(
        rangeslider_visible=True, # The slider at the bottom
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all", label="MAX")
            ]),
            bgcolor="#2c3542",
            activecolor="#00d09c",
            font=dict(color="white")
        )
    )

    fig_price.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#36404e'),
        yaxis=dict(gridcolor='#36404e'),
        height=400,
        margin=dict(l=0, r=0)
    )
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.write("Historical price data unavailable.")

st.divider()

# --- 1. VALUATION SECTION ---
st.header("1. Valuation")
v_col_left, v_col_right = st.columns([2, 1])

with v_col_left:
    st.subheader("Share Price vs Fair Value")
    diff = ((current_price - fair_value) / fair_value) * 100
    status_color = "#ff6384" if diff > 20 else "#ffce56" if diff > -20 else "#00d09c"
    status_text = "Overvalued" if diff > 0 else "Undervalued"
    
    st.markdown(f"""
    The stock is trading at **${current_price}**. Our estimated Fair Value is **${fair_value:.2f}**.
    It appears to be <span style="color:{status_color}; font-weight:bold;">{abs(diff):.1f}% {status_text}</span>.
    <br><small style="color:#8c97a7">Calculation Method: {calc_desc}</small>
    """, unsafe_allow_html=True)
    
    max_val = max(current_price, fair_value) * 1.3
    fig_val = go.Figure()
    fig_val.add_vrect(x0=0, x1=fair_value*0.8, fillcolor="#00d09c", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*0.8, x1=fair_value*1.2, fillcolor="#ffce56", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*1.2, x1=max_val*1.5, fillcolor="#ff6384", opacity=0.15, layer="below", line_width=0)

    fig_val.add_trace(go.Bar(y=[""], x=[current_price], name="Current Price", orientation='h', marker_color='#36a2eb', width=0.3, text=f"Current: ${current_price}", textposition='auto'))
    fig_val.add_trace(go.Bar(y=[""], x=[fair_value], name="Fair Value", orientation='h', marker_color='#232b36', marker_line_color='white', marker_line_width=2, width=0.3, text=f"Fair Value: ${fair_value:.2f}", textposition='auto'))

    fig_val.update_layout(xaxis=dict(range=[0, max_val], visible=False), yaxis=dict(visible=False), barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=150, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig_val, use_container_width=True)
    st.caption("🟩 Undervalued (<20%) | 🟨 Fair Value | 🟥 Overvalued (>20%)")

with v_col_right:
    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}")
    st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
    if graham_fv > 0: st.metric("Graham Number", f"${graham_fv:.2f}")
    else: st.metric("Graham Number", "N/A")

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
    fig_h.add_trace(go.Bar(x=['Cash', 'Debt'], y=[cash, debt], marker_color=['#00d09c', '#ff6384'], text=[f"${cash/1e9:.1f}B", f"${debt/1e9:.1f}B"], textposition='auto'))
    fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(t=0, b=0, l=0, r=0), height=200)
    st.plotly_chart(fig_h, use_container_width=True)
with h2:
    de_ratio = info.get('debtToEquity', 0)
    st.metric("Debt to Equity", f"{de_ratio:.1f}%")
    if de_ratio > 100: st.error(f"⚠️ High Debt ({de_ratio:.0f}%)")
    else: st.success(f"✅ Healthy Debt ({de_ratio:.0f}%)")
    if cash > debt: st.caption("✅ Cash covers total debt.")
    else: st.caption("⚠️ Debt exceeds total cash.")

# --- 5. DIVIDEND ---
st.divider()
st.header("5. Dividend")
d1, d2 = st.columns(2)
with d1:
    st.metric("Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
with d2:
    st.metric("Payout Ratio", f"{info.get('payoutRatio', 0)*100:.0f}%")

st.divider()

# --- NEW: COMPANY NEWS ---
st.header(f"Latest News for {ticker}")

# Helper function to safely extract data from different news formats
def get_news_data(article):
    # 1. Extract Title
    title = article.get('title')
    if not title and 'content' in article:
        title = article['content'].get('title')
    if not title:
        title = article.get('headline', 'No Title Available')
        
    # 2. Extract Link
    link = article.get('link')
    if isinstance(link, dict): # Fix dictionary links
        link = link.get('url')
        
    if not link and 'content' in article:
        link = article['content'].get('link') or article['content'].get('canonicalUrl')
        if isinstance(link, dict):
            link = link.get('url')
            
    if not link:
        link = article.get('clickThroughUrl', 'https://finance.yahoo.com')
    
    # 3. Extract Publisher (The Fix)
    publisher = "Unknown"
    
    # Case A: 'provider' key (Common in new API)
    if 'provider' in article:
        provider = article['provider']
        if isinstance(provider, dict):
            publisher = provider.get('displayName') or provider.get('title') or "Unknown"
            
    # Case B: 'publisher' key (Old API)
    elif 'publisher' in article:
        pub = article['publisher']
        if isinstance(pub, dict):
            publisher = pub.get('title') or pub.get('name') or "Unknown"
        else:
            publisher = str(pub)
            
    return title, link, publisher, article.get('providerPublishTime', 0)

news_list = stock.news

if news_list:
    for article in news_list[:5]: # Show top 5
        title, link, publisher, pub_time = get_news_data(article)
        
        # Convert timestamp
        date_str = ""
        if pub_time:
            try:
                date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
            except:
                date_str = ""
        
        st.markdown(f"""
        <div class="news-card">
            <a href="{link}" target="_blank" class="news-title">{title}</a><br>
            <span class="news-meta">{publisher} | {date_str}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.write("No recent news found via API.")

