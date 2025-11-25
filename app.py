import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta, time
from urllib.parse import urlparse

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="MarketRadar", page_icon="📡")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #1b222d; color: white; }
    h1, h2, h3, h4 { color: #00d09c !important; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: white !important; }
    div[data-testid="stMetricLabel"] { color: #8c97a7 !important; }
    .css-1d391kg { background-color: #232b36; border-radius: 15px; padding: 20px; }
    hr { border-color: #36404e; }
    .news-card { background-color: #232b36; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #00d09c; }
    .news-title { color: white; font-weight: bold; font-size: 1.1em; text-decoration: none; }
    .news-meta { color: #8c97a7; font-size: 0.85em; }
    div[data-baseweb="select"] > div { background-color: #2c3542; color: white; border-color: #444; }
    
    /* Timeframe Buttons */
    div[data-testid="stRadio"] > div { display: flex; justify-content: center; gap: 5px; width: 100%; flex-wrap: wrap; }
    div[data-testid="stRadio"] label {
        background-color: #232b36; padding: 5px 10px; border-radius: 5px; border: 1px solid #36404e;
        cursor: pointer; flex-grow: 1; text-align: center; font-size: 0.9rem;
    }
    div[data-testid="stRadio"] label:hover { border-color: #00d09c; color: #00d09c; }
    
    .perf-container {
        display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px;
        margin-top: 10px; margin-bottom: 20px; background-color: #232b36;
        padding: 15px; border-radius: 10px; text-align: center;
    }
    .perf-item { display: flex; flex-direction: column; }
    .perf-label { color: #8c97a7; font-size: 0.8rem; margin-bottom: 5px; }
    .perf-val { font-weight: bold; font-size: 1rem; }
    .pos { color: #00d09c; }
    .neg { color: #ff6384; }
    
    .check-item { margin-bottom: 8px; font-size: 0.9rem; }
    .check-pass { color: #00d09c; margin-right: 8px; }
    .check-fail { color: #ff6384; margin-right: 8px; }
    
    @media (max-width: 800px) {
        .perf-container { grid-template-columns: repeat(4, 1fr); gap: 15px; }
    }
    </style>
""", unsafe_allow_html=True)

# --- SEARCH FUNCTION ---
@st.cache_data(ttl=3600)
def search_symbol(query):
    if not query: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US&region=US&quotesCount=6&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        data = r.json()
        results = []
        if 'quotes' in data:
            for q in data['quotes']:
                symbol = q.get('symbol')
                name = q.get('shortname') or q.get('longname')
                exch = q.get('exchange')
                label = f"{symbol} - {name} ({exch})"
                results.append((label, symbol))
        return results
    except:
        return []

# --- TOP SEARCH BAR ---
col_search1, col_search2 = st.columns([1, 3])
with col_search1:
    exchange = st.selectbox("Region", ["All / US", "Canada (TSX) .TO", "Canada (Venture) .V", "UK (London) .L", "Australia .AX", "India .NS"])
with col_search2:
    search_query = st.text_input("🔎 Search Stock (Company Name or Ticker)", placeholder="e.g. Apple, Shopify, RY.TO...")

# Handle Search Logic
if search_query:
    search_results = search_symbol(search_query)
    if "Canada (TSX)" in exchange: search_results = [x for x in search_results if ".TO" in x[1]]
    elif "Venture" in exchange: search_results = [x for x in search_results if ".V" in x[1]]
        
    if search_results:
        selected_option = st.selectbox("Select Match:", options=[x[0] for x in search_results], key="search_select")
        if st.button("Analyze Stock"):
            st.query_params["ticker"] = selected_option.split(" - ")[0]
            st.rerun()
    else:
        st.warning("No matching stocks found.")

# --- URL & STATE MANAGEMENT ---
if "ticker" not in st.query_params:
    st.query_params["ticker"] = "AAPL"
ticker = st.query_params["ticker"]

if 'val_method' not in st.session_state:
    st.session_state.val_method = "Discounted Cash Flow (DCF)"

# --- FETCH DATA ---
news_list = [] 
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    
    if not current_price:
        st.error(f"Ticker '{ticker}' not found. Please check the symbol.")
        st.stop()
        
    # Fetch Financials & History
    financials = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T
    div_history = stock.dividends
    news_list = stock.news
    
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def get_val(df, keys_list):
    for k in keys_list:
        if k in df.columns:
            val = df[k].iloc[0]
            if pd.notna(val): return val
    return 0

def get_debt(df):
    d = get_val(df, ['Total Debt', 'Total Financial Debt'])
    if d == 0:
        long_term = get_val(df, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'])
        short_term = get_val(df, ['Current Debt', 'Current Debt And Capital Lease Obligation', 'Commercial Paper'])
        d = long_term + short_term
    return d

def fmt_num(num):
    if num is None or num == 0: return "N/A"
    if abs(num) >= 1e9: return f"${num/1e9:.1f}B"
    if abs(num) >= 1e6: return f"${num/1e6:.1f}M"
    return f"${num:.2f}"

def get_news_data(article):
    title = article.get('title')
    if not title and 'content' in article: title = article['content'].get('title')
    if not title: title = article.get('headline', 'No Title Available')
    
    link = article.get('link')
    if isinstance(link, dict): link = link.get('url')
    if not link and 'content' in article:
        link = article['content'].get('link') or article['content'].get('canonicalUrl')
        if isinstance(link, dict): link = link.get('url')
    if not link: link = article.get('clickThroughUrl', 'https://finance.yahoo.com')
    
    publisher = "Unknown"
    if 'provider' in article:
        provider = article['provider']
        if isinstance(provider, dict): publisher = provider.get('displayName') or provider.get('title') or "Unknown"
    elif 'publisher' in article:
        pub = article['publisher']
        if isinstance(pub, dict): publisher = pub.get('title') or pub.get('name') or "Unknown"
        else: publisher = str(pub)
    
    if publisher == "Unknown" and link:
        try:
            domain = urlparse(link).netloc
            clean_domain = domain.replace('www.', '').split('.')[0]
            if clean_domain: publisher = clean_domain.capitalize()
        except: pass
    return title, link, publisher, article.get('providerPublishTime', 0)

def calc_graham(info):
    eps = info.get('trailingEps', 0)
    bv = info.get('bookValue', 0)
    if eps > 0 and bv > 0: return np.sqrt(22.5 * eps * bv)
    return 0

def calc_dcf(stock, info):
    try:
        if 'Free Cash Flow' in cash_flow.columns: fcf_latest = cash_flow['Free Cash Flow'].iloc[0]
        else:
            ocf = cash_flow['Total Cash From Operating Activities'].iloc[0]
            capex = cash_flow['Capital Expenditures'].iloc[0]
            fcf_latest = ocf + capex 
        growth_rate = min(info.get('earningsGrowth', 0.10) or 0.08, 0.20)
        discount_rate = 0.09 
        future_cash_flows = []
        for i in range(1, 6):
            future_cash_flows.append((fcf_latest * ((1 + growth_rate) ** i)) / ((1 + discount_rate) ** i))
        term_val = (fcf_latest * ((1 + growth_rate) ** 5) * 1.025) / (discount_rate - 0.025)
        dcf_val = (sum(future_cash_flows) + (term_val / ((1 + discount_rate) ** 5))) / info.get('sharesOutstanding', 1)
        return dcf_val, growth_rate
    except: return 0, 0

def create_gauge(val, min_v, max_v, title, color="#00d09c", suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14, 'color': '#8c97a7'}},
        number={'suffix': suffix, 'font': {'size': 20}},
        gauge={'axis': {'range': [min_v, max_v]}, 'bar': {'color': color}, 
               'bgcolor': "#2c3542", 'borderwidth': 0}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=170, margin=dict(t=50, b=10, l=20, r=20))
    return fig

# --- VARIABLE EXTRACTION & CALCULATION ---
div_rate = info.get('dividendRate', 0)
if div_rate and current_price and current_price > 0:
    dy = div_rate / current_price
else:
    dy = info.get('dividendYield', 0) or 0

roe = info.get('returnOnEquity', 0) or 0
de = info.get('debtToEquity', 0) or 0
pe = info.get('trailingPE', 0) or 0
beta = info.get('beta', 1.0) or 1.0

# --- SMART GROWTH & PEG LOGIC ---
# 1. Get raw PEG from API
raw_peg = info.get('pegRatio', 0)
f_eps = info.get('forwardEps', 0) or 0
t_eps = info.get('trailingEps', 0) or 0

if raw_peg and raw_peg > 0 and pe > 0:
    # If API PEG is valid, use it to derive "Implied Growth" (Market Consensus)
    # PEG = PE / Growth  ->  Growth = PE / PEG
    peg = raw_peg
    g_rate = (pe / peg) / 100
elif f_eps > 0 and t_eps > 0:
    # If API PEG missing, calculate Growth from Estimates, then Calc PEG
    g_rate = (f_eps - t_eps) / t_eps
    peg = pe / (g_rate * 100) if g_rate > 0 else 0
else:
    # Fallback to trailing growth
    g_rate = info.get('earningsGrowth', 0) or 0
    peg = pe / (g_rate * 100) if g_rate > 0 else 0

rev_g = info.get('revenueGrowth', 0) or 0

# --- RUN CALCULATIONS ---
graham_fv = calc_graham(info)
dcf_fv, dcf_growth = calc_dcf(stock, info)
analyst_fv = info.get('targetMeanPrice', 0)

# Use Session State for Valuation Method
val_method = st.session_state.val_method

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

# --- SCORING HELPER ---
def check(condition, text):
    return (1, f"✅ {text}") if condition else (0, f"❌ {text}")

# --- 6-POINT CHECKLIST SCORING ENGINE ---

# 1. VALUATION
v_score = 0
v_details = []
s, t = check(current_price < fair_value, f"Below Fair Value ({current_price:.2f} < {fair_value:.2f})"); v_score+=s; v_details.append(t)
s, t = check(current_price < fair_value * 0.8, f"Significantly Below Fair Value ({current_price:.2f} < {(fair_value*0.8):.2f})"); v_score+=s; v_details.append(t)
s, t = check(pe > 0 and pe < 25, f"P/E ({pe:.1f}x) < Market (25x)"); v_score+=s; v_details.append(t)
s, t = check(pe > 0 and pe < 35, f"P/E ({pe:.1f}x) < Peers (35x)"); v_score+=s; v_details.append(t)
s, t = check(peg > 0 and peg < 1.5, f"PEG Ratio within ideal range ({peg:.2f} < 1.5x)"); v_score+=s; v_details.append(t)
s, t = check(current_price < analyst_fv, f"Below Analyst Target ({current_price:.2f} < {analyst_fv:.2f})"); v_score+=s; v_details.append(t)

# 2. FUTURE GROWTH
f_score = 0
f_details = []
s, t = check(g_rate > 0.02, f"Earnings Growth ({g_rate*100:.1f}%) > Savings Rate (2%)"); f_score+=s; f_details.append(t)
s, t = check(g_rate > 0.10, f"Earnings Growth ({g_rate*100:.1f}%) > Market Avg (10%)"); f_score+=s; f_details.append(t)
s, t = check(g_rate > 0.20, f"High Growth Earnings > 20%"); f_score+=s; f_details.append(t)
s, t = check(rev_g > 0.10, f"Revenue Growth ({rev_g*100:.1f}%) > Market Avg (10%)"); f_score+=s; f_details.append(t)
s, t = check(rev_g > 0.20, f"High Growth Revenue > 20%"); f_score+=s; f_details.append(t)
s, t = check(roe > 0.20, f"High Future ROE ({roe*100:.1f}%) > 20%"); f_score+=s; f_details.append(t)

# 3. PAST PERFORMANCE
p_score = 0
p_details = []
eps_growth_1y = 0
try:
    hist_fin = financials.sort_index()
    hist_bs = balance_sheet.sort_index()
    if not hist_fin.empty and len(hist_fin) >= 2:
        if 'Basic EPS' in hist_fin.columns: eps_series = hist_fin['Basic EPS']
        elif 'Net Income' in hist_fin.columns: eps_series = hist_fin['Net Income'] / hist_fin.get('Basic Average Shares', 1)
        else: eps_series = pd.Series([0])
        eps_series = eps_series.dropna()
        if len(eps_series) >= 2:
            curr_eps = eps_series.iloc[-1]
            prev_eps = eps_series.iloc[-2]
            oldest_eps = eps_series.iloc[0]
            eps_growth_1y = (curr_eps - prev_eps) / abs(prev_eps) if prev_eps != 0 else 0
            s, t = check(eps_growth_1y > 0.12, f"EPS Growth ({eps_growth_1y*100:.1f}%) > Industry (12%)"); p_score+=s; p_details.append(t)
            s, t = check(curr_eps > oldest_eps, f"Long Term Growth (EPS: {curr_eps:.2f} > {oldest_eps:.2f})"); p_score+=s; p_details.append(t)
            years = len(eps_series) - 1
            if years > 0 and oldest_eps > 0 and curr_eps > 0:
                cagr = (curr_eps / oldest_eps) ** (1/years) - 1
                s, t = check(eps_growth_1y > cagr, f"Accelerating Growth > {cagr*100:.1f}% Avg"); p_score+=s; p_details.append(t)
            else: p_details.append("❌ Accelerated Growth (Data Gap)")
            s, t = check(roe > 0.20, f"High ROE ({roe*100:.1f}% > 20%)"); p_score+=s; p_details.append(t)
            def get_roce(idx):
                try: return hist_fin['EBIT'].iloc[idx] / (hist_bs['Total Assets'].iloc[idx] - hist_bs['Current Liabilities'].iloc[idx])
                except: return 0
            curr_roce = get_roce(-1)
            old_roce = get_roce(-3) if len(hist_fin) >= 3 else get_roce(0)
            s, t = check(curr_roce > old_roce, f"ROCE Trend ({curr_roce*100:.1f}% > {old_roce*100:.1f}%)"); p_score+=s; p_details.append(t)
            roa = info.get('returnOnAssets', 0)
            s, t = check(roa > 0.06, f"ROA ({roa*100:.1f}%) > Industry (6%)"); p_score+=s; p_details.append(t)
        else: p_details.append("❌ Insufficient Historical Data")
    else: p_details.append("❌ Insufficient Historical Data")
except Exception as e: p_details.append(f"❌ Error in Past: {str(e)}")

# 4. FINANCIAL HEALTH
h_score = 0
h_details = []
try:
    curr_assets = get_val(balance_sheet, ['Current Assets', 'Total Current Assets'])
    curr_liab = get_val(balance_sheet, ['Current Liabilities', 'Total Current Liabilities'])
    total_liab = get_val(balance_sheet, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
    total_debt = get_debt(balance_sheet)
    equity = get_val(balance_sheet, ['Stockholders Equity', 'Total Stockholder Equity'])
    cash_bs = get_val(balance_sheet, ['Cash And Cash Equivalents', 'Cash', 'Cash Financial'])
    ebit = get_val(financials, ['EBIT', 'Operating Income', 'Net Income'])
    interest = abs(get_val(financials, ['Interest Expense', 'Interest Expense Non Operating', 'Total Interest Expenses']))
    ocf = get_val(cash_flow, ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities'])

    if curr_assets > 0 and curr_liab > 0: s, t = check(curr_assets > curr_liab, "Short Term Assets > Short Term Liab"); h_score+=s; h_details.append(t)
    else: h_score+=0; h_details.append("❌ Short Term Check (Bank/N/A)")
    
    if curr_assets > 0: s, t = check(curr_assets > (total_liab - curr_liab), "Short Term Assets > Long Term Liab"); h_score+=s; h_details.append(t)
    else: h_score+=0; h_details.append("❌ Long Term Check (Bank/N/A)")

    de_ratio = total_debt / equity if equity != 0 else 999
    s, t = check((de_ratio < 0.40) or (cash_bs > total_debt), f"Safe Debt Level (D/E: {de_ratio*100:.0f}%)"); h_score+=s; h_details.append(t)
    
    if len(balance_sheet.columns) > 1:
        prev_df = pd.DataFrame(balance_sheet.iloc[:, 1])
        prev_de = get_debt(prev_df) / get_val(prev_df, ['Stockholders Equity', 'Total Stockholder Equity'])
        s, t = check(de_ratio < prev_de, "Reducing Debt vs Last Year"); h_score+=s; h_details.append(t)
    else: h_details.append("❌ Reducing Debt (Data Gap)")

    if total_debt > 0: s, t = check(ocf > (total_debt * 0.2), f"Debt Coverage (OCF > 20% Debt)")
    else: s, t = 1, "✅ Debt Coverage (No Debt)"
    h_score+=s; h_details.append(t)

    if interest > 0: s, t = check(ebit > (interest * 5), f"Interest Coverage (EBIT > 5x Int)")
    else: s, t = 1, "✅ Interest Coverage (No Int)"
    h_score+=s; h_details.append(t)
except Exception as e: h_score=3; h_details.append(f"❌ Health Data Error: {str(e)}")

# 5. DIVIDEND
d_score = 0
d_details = []
is_notable = dy > 0.015
s, t = check(is_notable, f"Notable Dividend ({dy*100:.2f}% > 1.5%)"); d_score+=s; d_details.append(t)
s, t = check(dy > 0.035, f"High Dividend ({dy*100:.2f}% > 3.5%)"); d_score+=s; d_details.append(t)
is_stable = False; is_growing = False
try:
    if not div_history.empty and len(div_history) > 20:
        curr_div = div_history.iloc[-1]; old_div = div_history.iloc[-20]
        if curr_div >= old_div: is_stable = True
        if curr_div > old_div: is_growing = True
except: pass
if is_notable:
    s, t = check(is_stable, "Stable Dividend (10y)"); d_score+=s; d_details.append(t)
    s, t = check(is_growing, "Growing Dividend (10y)"); d_score+=s; d_details.append(t)
else:
    d_details.append("❌ Stable Dividend (Yield too low)")
    d_details.append("❌ Growing Dividend (Yield too low)")
payout = info.get('payoutRatio', 0) or 0
s, t = check(payout < 0.90 and dy > 0, f"Earnings Coverage (Payout {payout*100:.0f}%)"); d_score+=s; d_details.append(t)
cf_cover = False
try:
    div_paid = abs(get_val(cash_flow, ['Cash Dividends Paid']))
    fcf = get_val(cash_flow, ['Free Cash Flow'])
    if div_paid < fcf and dy > 0: cf_cover = True
    s, t = check(cf_cover, "Cash Flow Coverage"); d_score+=s; d_details.append(t)
except: d_details.append("❌ Cash Flow Coverage (Data Gap)")

final_scores = [v_score, f_score, p_score, h_score, d_score]
total_raw = sum(final_scores)
if total_raw < 12: flake_color = "#ff4b4b"
elif total_raw < 20: flake_color = "#ffb300"
else: flake_color = "#00d09c"
def hex_to_rgba(h, alpha): return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)
fill_rgba = f"rgba{hex_to_rgba(flake_color, 0.4)}"

# --- MAIN LAYOUT ---

st.markdown(f"### {info.get('shortName', ticker)} ({ticker})")
st.write(info.get('longBusinessSummary', '')[:350] + "...")

col1, col2 = st.columns([2, 1])

with col1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${current_price:.2f}")
    m2.metric("Market Cap", f"${(info.get('marketCap',0)/1e9):.1f}B")
    m3.metric("Beta", f"{info.get('beta', 0):.2f}")
    m4.metric("PE Ratio", f"{info.get('trailingPE',0):.1f}")

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(create_gauge(beta, 0, 3, "Beta", suffix="x"), use_container_width=True)
    g2.plotly_chart(create_gauge(info.get('marketCap',0)/1e9, 0, 3000, "Market Cap ($B)", color="#36a2eb"), use_container_width=True)
    g3.plotly_chart(create_gauge(current_price, 0, current_price*1.5, "Price ($)"), use_container_width=True)

with col2:
    # --- SNOWFLAKE ---
    r_vals = final_scores + [final_scores[0]]
    theta_vals = ['Value', 'Future', 'Past', 'Health', 'Dividend', 'Value']
    fig = go.Figure(data=go.Scatterpolar(
        r=r_vals, theta=theta_vals, fill='toself', line_shape='spline', 
        line_color=flake_color, fillcolor=fill_rgba, hoverinfo='text', 
        text=[f"{s}/6" for s in r_vals], marker=dict(size=5)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 6], tickvals=[1, 2, 3, 4, 5, 6], showticklabels=False, gridcolor='#444', gridwidth=1.5, layer='below traces'),
            angularaxis=dict(direction='clockwise', rotation=90, gridcolor='rgba(0,0,0,0)', tickfont=dict(color='white', size=12)),
            bgcolor='#232b36'
        ),
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=20, l=40, r=40), showlegend=False, height=350
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Breakdown"):
        t1, t2, t3, t4, t5 = st.tabs(["Val", "Fut", "Pst", "Hlt", "Div"])
        with t1: 
            st.caption(f"Score: {v_score}/6")
            for x in v_details: st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True)
        with t2: 
            st.caption(f"Score: {f_score}/6")
            for x in f_details: st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True)
        with t3: 
            st.caption(f"Score: {p_score}/6")
            for x in p_details: st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True)
        with t4: 
            st.caption(f"Score: {h_score}/6")
            for x in h_details: st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True)
        with t5: 
            st.caption(f"Score: {d_score}/6")
            for x in d_details: st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True)

st.divider()

# --- PRICE HISTORY ---
st.header("Price History")

# Placeholder for the Graph
chart_placeholder = st.empty()

# 1. PRE-FETCH MAX HISTORY FOR LABELS
perf_data = stock.history(period="max", interval="1d")
if not perf_data.empty:
    if perf_data.index.tz is not None:
        perf_data.index = perf_data.index.tz_localize(None)
    curr_c = perf_data['Close'].iloc[-1]
else:
    curr_c = 0

def get_ret_fmt(days, fixed=None):
    try:
        if fixed: 
            idx = perf_data.index.get_indexer([pd.to_datetime(fixed)], method='nearest')[0]
            p = perf_data['Close'].iloc[idx]
        else: p = perf_data['Close'].iloc[-days]
        ret = ((curr_c - p)/p)*100
        sign = "+" if ret >=0 else ""
        return f"({sign}{ret:.1f}%)"
    except: return ""

# Labels for Buttons
tf_labels = {}
ytd_d = datetime(datetime.now().year, 1, 1)
ret_1d = "(-)"
if not perf_data.empty: ret_1d = get_ret_fmt(2)

tf_labels["1D"] = f"1D {ret_1d}"
tf_labels["5D"] = f"5D {get_ret_fmt(6)}"
tf_labels["1M"] = f"1M {get_ret_fmt(22)}"
tf_labels["6M"] = f"6M {get_ret_fmt(126)}"
tf_labels["YTD"] = f"YTD {get_ret_fmt(0, ytd_d)}"
tf_labels["1Y"] = f"1Y {get_ret_fmt(252)}"
tf_labels["5Y"] = f"5Y {get_ret_fmt(1260)}"
tf_labels["Max"] = f"Max {get_ret_fmt(len(perf_data)-1)}"

def format_func(option): return tf_labels.get(option, option)

# Buttons (Static Keys)
tf_keys = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
if 'tf_sel' not in st.session_state: st.session_state.tf_sel = '1D'
def update_tf(): pass

# Render Buttons Below the Placeholder spot
timeframe = st.radio("TF", tf_keys, format_func=format_func, horizontal=True, label_visibility="collapsed", key="tf_sel", on_change=update_tf)

# Performance Strip (Dynamic Data)
ytd_d = datetime(datetime.now().year, 1, 1)
ret_1d = "(-)"
if not perf_data.empty: ret_1d = get_ret_fmt(2)

def get_color(val_str):
    if "+" in val_str: return "pos"
    if "-" in val_str: return "neg"
    return ""

# Logic
df = pd.DataFrame()
y_rng = None; x_rng = None

if timeframe == '1D':
    df = stock.history(period='1d', interval='5m', prepost=True)
    if not df.empty:
        ldt = df.index[-1]
        x_rng = [ldt.replace(hour=7, minute=30), ldt.replace(hour=18, minute=0)]
elif timeframe == '5D': df = stock.history(period='5d', interval='15m', prepost=True)
elif timeframe == '1M': df = stock.history(period='1mo', interval='1d')
elif timeframe == '6M': df = stock.history(period='6mo', interval='1d')
elif timeframe == 'YTD': df = stock.history(period='ytd', interval='1d')
elif timeframe == '1Y': df = stock.history(period='1y', interval='1d')
elif timeframe == '5Y': df = stock.history(period='5y', interval='1d')
elif timeframe == 'Max': df = stock.history(period='max', interval='1d')

if not df.empty:
    ymin = df['Close'].min(); ymax = df['Close'].max()
    buff = (ymax - ymin)*0.05 if ymax!=ymin else ymax*0.01
    y_rng = [ymin - buff, ymax + buff]
    
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='#36a2eb' if timeframe in ['1D','5D'] else '#00d09c', width=2), fill='tozeroy', fillcolor=f"rgba(0,208,156,0.1)" if timeframe not in ['1D','5D'] else "rgba(54,162,235,0.1)", hovertemplate='<b>%{x|%b %d %H:%M}</b><br>$%{y:.2f}<extra></extra>'))
    
    xa = dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikecolor="white", spikethickness=1, gridcolor='#36404e')
    if timeframe == '1D' and x_rng: xa['range'] = x_rng
    
    fig_p.update_xaxes(**xa)
    fig_p.update_yaxes(range=y_rng, showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash', spikecolor="white", spikethickness=1, gridcolor='#36404e')
    fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified", hoverlabel=dict(bgcolor="#2c3542", font_size=14, font_family="Segoe UI"))
    
    chart_placeholder.plotly_chart(fig_p, use_container_width=True)
else:
    chart_placeholder.write("Price data unavailable for this timeframe.")

# Performance Strip Below Buttons
v_1d = ret_1d
v_5d = get_ret_fmt(6)
v_1m = get_ret_fmt(22)
v_6m = get_ret_fmt(126)
v_ytd = get_ret_fmt(0, ytd_d)
v_1y = get_ret_fmt(252)
v_5y = get_ret_fmt(1260)
v_max = get_ret_fmt(len(perf_data)-1)

st.markdown(f"""
<div class="perf-container">
    <div class="perf-item"><span class="perf-label">1 Day</span><span class="perf-val {get_color(v_1d)}">{v_1d}</span></div>
    <div class="perf-item"><span class="perf-label">5 Days</span><span class="perf-val {get_color(v_5d)}">{v_5d}</span></div>
    <div class="perf-item"><span class="perf-label">1 Month</span><span class="perf-val {get_color(v_1m)}">{v_1m}</span></div>
    <div class="perf-item"><span class="perf-label">6 Months</span><span class="perf-val {get_color(v_6m)}">{v_6m}</span></div>
    <div class="perf-item"><span class="perf-label">YTD</span><span class="perf-val {get_color(v_ytd)}">{v_ytd}</span></div>
    <div class="perf-item"><span class="perf-label">1 Year</span><span class="perf-val {get_color(v_1y)}">{v_1y}</span></div>
    <div class="perf-item"><span class="perf-label">5 Years</span><span class="perf-val {get_color(v_5y)}">{v_5y}</span></div>
    <div class="perf-item"><span class="perf-label">All Time</span><span class="perf-val {get_color(v_max)}">{v_max}</span></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- 1. VALUATION ---
st.header("1. Valuation")
c_val1, c_val2 = st.columns([2, 1])
with c_val1:
    st.subheader("Fair Value Analysis")
    def update_val_method(): pass
    st.radio("Method", ["Discounted Cash Flow (DCF)", "Graham Formula", "Analyst Target"], key="val_method", on_change=update_val_method, horizontal=True)
    max_v = max(current_price, fair_value) * 1.3
    fig_v = go.Figure()
    fig_v.add_vrect(x0=0, x1=fair_value*0.8, fillcolor="#00d09c", opacity=0.15, layer="below", line_width=0)
    fig_v.add_vrect(x0=fair_value*0.8, x1=fair_value*1.2, fillcolor="#ffce56", opacity=0.15, layer="below", line_width=0)
    fig_v.add_vrect(x0=fair_value*1.2, x1=max_v*1.5, fillcolor="#ff6384", opacity=0.15, layer="below", line_width=0)
    fig_v.add_trace(go.Bar(y=[""], x=[current_price], name="Price", orientation='h', marker_color='#36a2eb', width=0.3, text=f"${current_price}", textposition='auto'))
    fig_v.add_trace(go.Bar(y=[""], x=[fair_value], name="Fair Value", orientation='h', marker_color='#232b36', marker_line_color='white', marker_line_width=2, width=0.3, text=f"${fair_value:.2f}", textposition='auto'))
    fig_v.update_layout(xaxis=dict(range=[0, max_v], visible=False), yaxis=dict(visible=False), barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=180, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig_v, use_container_width=True)

with c_val2:
    g1 = create_gauge(pe, 0, 50, "P/E Ratio", suffix="x", color="#ff6384" if pe>30 else "#00d09c")
    st.plotly_chart(g1, use_container_width=True)
    g2 = create_gauge(peg, 0, 3, "PEG Ratio", color="#00d09c" if peg<1.5 else "#ff6384")
    st.plotly_chart(g2, use_container_width=True)

st.divider()

# --- 2. FUTURE ---
st.header("2. Future Growth")
f1, f2 = st.columns(2)
with f1:
    fig_f = go.Figure(data=[
        go.Bar(name='Company', x=['Growth'], y=[g_rate*100], marker_color='#36a2eb', text=[f"{g_rate*100:.1f}%"]),
        go.Bar(name='Market', x=['Growth'], y=[0.10*100], marker_color='#ff6384', text=["10.0%"]),
        go.Bar(name='Savings', x=['Growth'], y=[0.02*100], marker_color='#ffce56', text=["2.0%"])
    ])
    fig_f.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), title="Annual Forecast", height=300)
    st.plotly_chart(fig_f, use_container_width=True)
with f2:
    fig_roe = go.Figure(go.Indicator(
        mode = "number+gauge", value = roe*100, title = {'text': "Future ROE Est."},
        gauge = {'shape': "bullet", 'axis': {'range': [0, 100]}, 'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 20}}
    ))
    fig_roe.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250)
    st.plotly_chart(fig_roe, use_container_width=True)

st.divider()

# --- 3. PAST ---
st.header("3. Past Performance")
p1, p2 = st.columns(2)
with p1:
    st.plotly_chart(create_gauge(info.get('returnOnEquity',0)*100, 0, 50, "Return on Equity (ROE)", suffix="%"), use_container_width=True)
with p2:
    st.plotly_chart(create_gauge(info.get('returnOnAssets',0)*100, 0, 20, "Return on Assets (ROA)", suffix="%"), use_container_width=True)

st.divider()

# --- 4. HEALTH ---
st.header("4. Financial Health")
fin_freq = st.radio("Frequency:", ["Quarterly", "Annual"], horizontal=True)
if fin_freq == "Annual":
    f_data = stock.financials.T.iloc[::-1]; b_data = stock.balance_sheet.T.iloc[::-1]
    d_fmt = "%Y"
else:
    f_data = stock.quarterly_financials.T.iloc[::-1]; b_data = stock.quarterly_balance_sheet.T.iloc[::-1]
    d_fmt = "%Y-%m"
d_lbls = [d.strftime(d_fmt) for d in f_data.index]

h1, h2 = st.columns([2, 1])
with h1:
    if not b_data.empty:
        d_vals = [get_debt(pd.DataFrame(b_data.iloc[[i]])) for i in range(len(b_data))]
        c_vals = [get_val(pd.DataFrame(b_data.iloc[[i]]), ['Cash And Cash Equivalents', 'Cash']) for i in range(len(b_data))]
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=d_lbls, y=d_vals, name='Debt', marker_color='#ff6384', text=[f"${v/1e9:.1f}B" for v in d_vals], textposition='auto'))
        fig_h.add_trace(go.Bar(x=d_lbls, y=c_vals, name='Cash', marker_color='#36a2eb', text=[f"${v/1e9:.1f}B" for v in c_vals], textposition='auto'))
        fig_h.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=350)
        st.plotly_chart(fig_h, use_container_width=True)
    else: st.write("Data Unavailable")
with h2:
    st.plotly_chart(create_gauge(de, 0, 200, "Debt to Equity", suffix="%"), use_container_width=True)
    st.plotly_chart(create_gauge(info.get('currentRatio',0), 0, 5, "Current Ratio", suffix="x"), use_container_width=True)

st.divider()

# --- 5. DIVIDEND ---
st.header("5. Dividend")
d1, d2 = st.columns(2)
with d1:
    st.plotly_chart(create_gauge(dy*100, 0, max(6, dy*100), "Dividend Yield", suffix="%"), use_container_width=True)
with d2:
    payout = info.get('payoutRatio', 0) or 0
    if payout > 0:
        fig_pay = go.Figure(data=[go.Pie(labels=['Payout', 'Retained'], values=[payout, 1-payout], hole=.7, marker=dict(colors=['#36a2eb', '#232b36']))])
        fig_pay.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), annotations=[dict(text=f"{payout*100:.0f}%", x=0.5, y=0.5, font_size=24, showarrow=False)])
        st.plotly_chart(fig_pay, use_container_width=True)
    else: st.write("No Dividend Payout")

st.divider()

# --- NEWS ---
st.header(f"Latest News for {ticker}")
if news_list:
    for article in news_list[:5]:
        title, link, publisher, pub_time = get_news_data(article)
        date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M') if pub_time else ""
        meta_text = f"{publisher} | {date_str}" if publisher and publisher != "Unknown" else date_str
        st.markdown(f"""<div class="news-card"><a href="{link}" target="_blank" class="news-title">{title}</a><br><span class="news-meta">{meta_text}</span></div>""", unsafe_allow_html=True)
else: st.write("No recent news found.")

st.markdown("<br><br><br>", unsafe_allow_html=True)
