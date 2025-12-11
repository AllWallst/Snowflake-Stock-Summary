import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="MarketRadar", page_icon="📡")

# --- SESSION STATE INITIALIZATION ---
if 'val_method' not in st.session_state: st.session_state.val_method = "Discounted Cash Flow (DCF)"
if 'tf_sel' not in st.session_state: st.session_state.tf_sel = '1D'

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

# --- CACHED DATA FETCHING FUNCTIONS ---
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

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Accessing info triggers the request
        info = stock.info
        
        # Helper to safely get dataframes
        def get_df(attr):
            try: return getattr(stock, attr).T
            except: return pd.DataFrame()
            
        financials = get_df('financials')
        balance_sheet = get_df('balance_sheet')
        cash_flow = get_df('cashflow')
        
        # Quarterly
        q_financials = get_df('quarterly_financials')
        q_balance_sheet = get_df('quarterly_balance_sheet')
        
        div_history = stock.dividends
        news = stock.news
        
        # Fetch max history once to be sliced later for performance metrics and charts
        # This saves multiple API calls
        history = stock.history(period="max", interval="1d")
        if history.index.tz is not None:
            history.index = history.index.tz_localize(None)
        
        # IMPORTANT: Do NOT return the 'stock' object itself, it is not serializable.
        return info, financials, balance_sheet, cash_flow, q_financials, q_balance_sheet, div_history, news, history
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None, None, None, None, None, None, None

# --- TOP SEARCH BAR ---
col_search1, col_search2 = st.columns([1, 3])
with col_search1:
    exchange = st.selectbox("Region", ["All / US", "Canada (TSX) .TO", "Canada (Venture) .V", "UK (London) .L", "Australia .AX", "India .NS"])
with col_search2:
    search_query = st.text_input("🔎 Search Stock (Company Name or Ticker)", placeholder="e.g. Apple, Shopify, RY.TO...")

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

# --- URL & DATA LOADING ---
if "ticker" not in st.query_params:
    st.query_params["ticker"] = "AAPL"
ticker = st.query_params["ticker"]

# Unpacking the data (Removed 'stock' from the tuple)
info, financials, balance_sheet, cash_flow, q_financials, q_balance_sheet, div_history, news_list, full_history = get_stock_data(ticker)

if not info or 'currentPrice' not in info:
    st.error(f"Ticker '{ticker}' not found or data unavailable.")
    st.stop()

current_price = info.get('currentPrice') or info.get('regularMarketPrice')

# --- HELPER FUNCTIONS ---
def get_val(df, keys_list):
    if df.empty: return 0
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

# --- UPDATED MATH FUNCTIONS ---

def calc_graham(info):
    # Added safety check: Graham number fails if EPS or BV are negative
    eps = info.get('trailingEps', 0)
    bv = info.get('bookValue', 0)
    if eps > 0 and bv > 0: return np.sqrt(22.5 * eps * bv)
    return 0

def calc_dcf(info, cash_flow):
    try:
        # 1. Determine Base Free Cash Flow
        if not cash_flow.empty and 'Free Cash Flow' in cash_flow.columns:
            fcf_latest = cash_flow['Free Cash Flow'].iloc[0]
        elif not cash_flow.empty:
            # Fallback: OCF + CapEx (CapEx is usually negative)
            ocf = get_val(cash_flow, ['Total Cash From Operating Activities'])
            capex = get_val(cash_flow, ['Capital Expenditures'])
            fcf_latest = ocf + capex
        else:
            return 0, 0, 0

        # 2. Dynamic Discount Rate (WACC / Cost of Equity)
        # Risk Free Rate ~4.2% (10Y Treasury), Market Return ~10%
        risk_free = 0.042
        market_return = 0.10
        beta = info.get('beta', 1.0)
        if beta is None: beta = 1.0
        
        # Cost of Equity = Rf + Beta * (Rm - Rf)
        discount_rate = risk_free + beta * (market_return - risk_free)
        # Clamp discount rate for sanity (6% to 15%)
        discount_rate = max(0.06, min(discount_rate, 0.15))

        # 3. Growth Rate
        # Use analyst estimate, cap at 20% to be conservative
        growth_rate = info.get('earningsGrowth', 0.10)
        if growth_rate is None: growth_rate = 0.05
        growth_rate = min(growth_rate, 0.20)

        # 4. Projection
        future_cash_flows = []
        for i in range(1, 6):
            # FCF * (1+g)^i / (1+r)^i
            val = (fcf_latest * ((1 + growth_rate) ** i)) / ((1 + discount_rate) ** i)
            future_cash_flows.append(val)
        
        # 5. Terminal Value (Gordon Growth Method)
        # Terminal Growth 2.5% (Inflation)
        perp_growth = 0.025
        fcf_year_5 = fcf_latest * ((1 + growth_rate) ** 5)
        # Value at Year 5 = FCF5 * (1+g) / (r - g)
        term_val = (fcf_year_5 * (1 + perp_growth)) / (discount_rate - perp_growth)
        # Discount Terminal Value back to today
        term_val_discounted = term_val / ((1 + discount_rate) ** 5)
        
        total_equity_val = sum(future_cash_flows) + term_val_discounted
        shares = info.get('sharesOutstanding', 1)
        dcf_val = total_equity_val / shares
        
        return dcf_val, growth_rate, discount_rate
    except:
        return 0, 0, 0

def create_gauge(val, min_v, max_v, title, color="#00d09c", suffix=""):
    if val is None: val = 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14, 'color': '#8c97a7'}},
        number={'suffix': suffix, 'font': {'size': 20}},
        gauge={'axis': {'range': [min_v, max_v]}, 'bar': {'color': color}, 
               'bgcolor': "#2c3542", 'borderwidth': 0}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=170, margin=dict(t=50, b=10, l=20, r=20))
    return fig

# --- VARIABLE EXTRACTION ---
div_rate = info.get('dividendRate', 0)
if div_rate and current_price and current_price > 0:
    dy = div_rate / current_price
else:
    dy = info.get('dividendYield', 0) or 0

roe = info.get('returnOnEquity', 0) or 0
de = info.get('debtToEquity', 0) or 0
pe = info.get('trailingPE', 0) or 0
beta = info.get('beta', 1.0) or 1.0

# PEG Logic - fallback to "N/A" rather than 1-year calc which is volatile
peg = info.get('pegRatio', 0)
if peg is None: peg = 0 

# --- RUN CALCULATIONS ---
graham_fv = calc_graham(info)
dcf_fv, dcf_growth, dcf_disc = calc_dcf(info, cash_flow)
analyst_fv = info.get('targetMeanPrice', 0)

val_method = st.session_state.val_method

if val_method == "Discounted Cash Flow (DCF)":
    fair_value = dcf_fv
    calc_desc = f"5Y DCF Model (Growth: {dcf_growth*100:.1f}%, Disc: {dcf_disc*100:.1f}%)"
elif val_method == "Graham Formula":
    fair_value = graham_fv
    calc_desc = "Graham Number (Sqrt(22.5 * EPS * Book Value))"
else:
    fair_value = analyst_fv
    calc_desc = "Average Analyst Price Target"

if fair_value == 0 or np.isnan(fair_value) or fair_value < 0:
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
s, t = check(analyst_fv > 0 and current_price < analyst_fv, f"Below Analyst Target ({current_price:.2f} < {analyst_fv:.2f})"); v_score+=s; v_details.append(t)

# 2. FUTURE GROWTH
f_score = 0
f_details = []
# Prefer PEG implied growth, else fallback to analyst estimates
if peg > 0 and pe > 0: g_rate = (pe / peg) / 100
else: g_rate = info.get('earningsGrowth', 0) or 0
rev_g = info.get('revenueGrowth', 0) or 0

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
            
            # MATH FIX: Ensure oldest_eps > 0 for CAGR calculation
            if years > 0 and oldest_eps > 0 and curr_eps > 0:
                cagr = (curr_eps / oldest_eps) ** (1/years) - 1
                s, t = check(eps_growth_1y > cagr, f"Accelerating Growth > {cagr*100:.1f}% Avg"); p_score+=s; p_details.append(t)
            else: 
                s, t = 0, "❌ Accelerating Growth (N/A or Neg EPS)"
                p_score+=s; p_details.append(t)
                
            s, t = check(roe > 0.20, f"High ROE ({roe*100:.1f}% > 20%)"); p_score+=s; p_details.append(t)
            
            # ROCE Calculation
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
    div_paid = abs(get_val(cash_flow, ['Cash Dividends Paid', 'Common Stock Dividend Paid']))
    fcf = get_val(cash_flow, ['Free Cash Flow'])
    if fcf == 0: # Fallback calc if FCF missing
         fcf = get_val(cash_flow, ['Total Cash From Operating Activities']) + get_val(cash_flow, ['Capital Expenditures'])
    
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

# --- METRICS ROW (GAUGES + NUMBERS) ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"${current_price:.2f}")
m2.metric("Market Cap", f"${(info.get('marketCap',0)/1e9):.1f}B")
m3.metric("Beta", f"{info.get('beta', 0):.2f}")
m4.metric("PE Ratio", f"{info.get('trailingPE',0):.1f}")

g1, g2, g3 = st.columns(3)
g1.plotly_chart(create_gauge(beta, 0, 3, "Beta", suffix="x"), use_container_width=True)
g2.plotly_chart(create_gauge(info.get('marketCap',0)/1e9, 0, 3000, "Market Cap ($B)", color="#36a2eb"), use_container_width=True)
g3.plotly_chart(create_gauge(current_price, 0, current_price*1.5, "Price ($)"), use_container_width=True)

st.divider()

# --- SNOWFLAKE & ANALYSIS BREAKDOWN ---
st.header("Fundamental Analysis")

# Centered Snowflake with Columns
c_left, c_center, c_right = st.columns([1, 2, 1])

with c_center:
    r_vals = final_scores + [final_scores[0]]
    theta_vals = ['Value', 'Future', 'Past', 'Health', 'Dividend', 'Value']
    fig = go.Figure(data=go.Scatterpolar(r=r_vals, theta=theta_vals, fill='toself', line_shape='spline', line_color=flake_color, fillcolor=fill_rgba, hoverinfo='text', text=[f"{s}/6" for s in r_vals], marker=dict(size=5)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 6], tickvals=[1, 2, 3, 4, 5, 6], showticklabels=False, gridcolor='#444', gridwidth=1.5, layer='below traces'), angularaxis=dict(direction='clockwise', rotation=90, gridcolor='rgba(0,0,0,0)', tickfont=dict(color='white', size=12)), bgcolor='#232b36'), paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=20, l=40, r=40), showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# Analysis Breakdown
with st.expander("📊 See Analysis Breakdown", expanded=True):
    t1, t2, t3, t4, t5 = st.tabs(["Valuation", "Future Growth", "Past Performance", "Financial Health", "Dividend"])
    def print_list(items):
        for x in items: st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True)
    with t1: st.markdown(f"**Score: {v_score}/6**"); print_list(v_details)
    with t2: st.markdown(f"**Score: {f_score}/6**"); print_list(f_details)
    with t3: st.markdown(f"**Score: {p_score}/6**"); print_list(p_details)
    with t4: st.markdown(f"**Score: {h_score}/6**"); print_list(h_details)
    with t5: st.markdown(f"**Score: {d_score}/6**"); print_list(d_details)

st.divider()

# --- PRICE HISTORY & PERFORMANCE ---
st.header("Price History")

# Labels for Buttons
tf_keys = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
def update_tf(): pass
timeframe = st.radio("TF", tf_keys, horizontal=True, label_visibility="collapsed", key="tf_sel", on_change=update_tf)

# Helper to slice the cached full_history
def get_chart_data(full_history, tf):
    if full_history.empty: return pd.DataFrame()
    end_date = full_history.index[-1]
    
    # For 1D and 5D, yfinance history(period='max') usually returns daily data only.
    # To get intraday 1m/5m data, we must fetch fresh.
    # BUT to avoid API lag, we only do it if strictly necessary, otherwise we use daily for long views.
    
    if tf == '1D':
        # Must fetch fresh for intraday
        try: return yf.Ticker(ticker).history(period='1d', interval='5m', prepost=True)
        except: return full_history.iloc[-1:] # Fallback
    elif tf == '5D':
         try: return yf.Ticker(ticker).history(period='5d', interval='15m', prepost=True)
         except: return full_history.iloc[-5:] # Fallback
    
    # For others, slice the Daily max history
    start_date = None
    if tf == '1M': start_date = end_date - timedelta(days=30)
    elif tf == '6M': start_date = end_date - timedelta(days=180)
    elif tf == 'YTD': start_date = datetime(end_date.year, 1, 1)
    elif tf == '1Y': start_date = end_date - timedelta(days=365)
    elif tf == '5Y': start_date = end_date - timedelta(days=365*5)
    else: return full_history # Max
    
    return full_history[full_history.index >= pd.Timestamp(start_date)]

df = get_chart_data(full_history, timeframe)
chart_placeholder = st.empty()

if not df.empty:
    curr_c = df['Close'].iloc[-1]
    ymin = df['Close'].min(); ymax = df['Close'].max()
    buff = (ymax - ymin)*0.05 if ymax!=ymin else ymax*0.01
    y_rng = [ymin - buff, ymax + buff]
    
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='#36a2eb' if timeframe in ['1D','5D'] else '#00d09c', width=2), fill='tozeroy', fillcolor=f"rgba(0,208,156,0.1)" if timeframe not in ['1D','5D'] else "rgba(54,162,235,0.1)", hovertemplate='<b>%{x|%b %d %H:%M}</b><br>$%{y:.2f}<extra></extra>'))
    
    fig_p.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikecolor="white", spikethickness=1, gridcolor='#36404e')
    fig_p.update_yaxes(range=y_rng, showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash', spikecolor="white", spikethickness=1, gridcolor='#36404e')
    fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified", hoverlabel=dict(bgcolor="#2c3542", font_size=14, font_family="Segoe UI"))
    
    chart_placeholder.plotly_chart(fig_p, use_container_width=True)
else:
    chart_placeholder.write("Price data unavailable for this timeframe.")

# Performance Strip Calculation (using cached full_history)
def get_ret(days, fixed=None):
    if full_history.empty: return "N/A"
    try:
        curr = full_history['Close'].iloc[-1]
        if fixed:
            # Find closest index
            idx = full_history.index.get_indexer([pd.Timestamp(fixed)], method='nearest')[0]
            past = full_history['Close'].iloc[idx]
        else:
            if len(full_history) < days: return "N/A"
            past = full_history['Close'].iloc[-days]
        
        ret = ((curr - past)/past)*100
        sign = "+" if ret >=0 else ""
        return f"{sign}{ret:.1f}%"
    except: return "N/A"

def get_color(val_str):
    if "+" in val_str: return "pos"
    if "-" in val_str: return "neg"
    return ""

# Note: 1D perf here is from previous close of Max history, might differ slightly from realtime 5m data
v_1d = get_ret(2) 
v_5d = get_ret(6)
v_1m = get_ret(22)
v_6m = get_ret(126)
v_ytd = get_ret(0, datetime(datetime.now().year, 1, 1))
v_1y = get_ret(252)
v_5y = get_ret(1260)
v_max = get_ret(len(full_history)-1)

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

# --- 1. VALUATION UI ---
st.header("1. Valuation")
c_val1, c_val2 = st.columns([2, 1])
with c_val1:
    st.subheader(f"Fair Value: {calc_desc}")
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
        go.Bar(name='Company', x=['Growth'], y=[g_rate*100], marker_color='#36a2eb', text=[f"{g_rate*100:.1f}%"], textposition='auto'),
        go.Bar(name='Market', x=['Growth'], y=[10.0], marker_color='#ff6384', text=["10.0%"], textposition='auto'),
        go.Bar(name='Savings', x=['Growth'], y=[2.0], marker_color='#ffce56', text=["2.0%"], textposition='auto')
    ])
    fig_f.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), title="Annual Forecast", height=250, showlegend=True, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_f, use_container_width=True)
with f2:
    st.plotly_chart(create_gauge(roe*100, 0, max(30, roe*100), "Future Return on Equity (ROE)", suffix="%"), use_container_width=True)

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
    f_data = financials.iloc[::-1]; b_data = balance_sheet.iloc[::-1]
    d_fmt = "%Y"
else:
    f_data = q_financials.iloc[::-1]; b_data = q_balance_sheet.iloc[::-1]
    d_fmt = "%Y-%m"
    
if not f_data.empty: d_lbls = [d.strftime(d_fmt) for d in f_data.index]
else: d_lbls = []

h1, h2 = st.columns([2, 1])
with h1:
    if not b_data.empty and len(b_data) > 0:
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
        fig_pay = go.Figure(data=[go.Pie(labels=['Payout', 'Retained'], values=[payout, 1-payout], hole=.7, marker=dict(colors=['#36a2eb', '#232b36']), textinfo='none', hoverinfo='label+percent')])
        fig_pay.update_layout(showlegend=False, height=170, margin=dict(t=50, b=10, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), title={'text': "Payout Ratio", 'x': 0.5, 'y': 0.85, 'xanchor': 'center', 'font': {'size': 14, 'color': '#8c97a7'}}, annotations=[dict(text=f"<span style='font-size:20px; font-weight:bold'>{payout*100:.0f}%</span>", x=0.5, y=0.5, showarrow=False)])
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
