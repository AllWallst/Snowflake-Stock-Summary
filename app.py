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
    
    /* News Card */
    .news-card { background-color: #232b36; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #00d09c; }
    .news-title { color: white; font-weight: bold; font-size: 1.1em; text-decoration: none; }
    .news-meta { color: #8c97a7; font-size: 0.85em; }
    div[data-baseweb="select"] > div { background-color: #2c3542; color: white; border-color: #444; }
    
    /* Timeframe Buttons */
    div[data-testid="stRadio"] > div { 
        display: flex; justify-content: center; flex-wrap: wrap; gap: 5px; width: 100%; 
    }
    div[data-testid="stRadio"] label {
        background-color: #232b36; padding: 8px 16px; border-radius: 5px; border: 1px solid #36404e;
        cursor: pointer; flex-grow: 1; text-align: center; font-size: 0.95rem;
    }
    div[data-testid="stRadio"] label:hover { border-color: #00d09c; color: #00d09c; }
    
    /* Performance Grid */
    .perf-container {
        display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px;
        margin-top: 15px; margin-bottom: 30px; background-color: #232b36;
        padding: 20px; border-radius: 10px; text-align: center;
    }
    .perf-item { display: flex; flex-direction: column; }
    .perf-label { color: #8c97a7; font-size: 0.85rem; margin-bottom: 5px; }
    .perf-val { font-weight: bold; font-size: 1.1rem; }
    .pos { color: #00d09c; }
    .neg { color: #ff6384; }
    
    /* Checklist Styles */
    .check-item { margin-bottom: 8px; font-size: 0.9rem; }
    
    /* Section Container */
    .section-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #36404e;
        margin-bottom: 20px;
    }
    
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

if search_query:
    search_results = search_symbol(search_query)
    if "Canada (TSX)" in exchange: search_results = [x for x in search_results if ".TO" in x[1]]
    elif "Venture" in exchange: search_results = [x for x in search_results if ".V" in x[1]]
    if search_results:
        selected_option = st.selectbox("Select Match:", options=[x[0] for x in search_results], key="search_select")
        if st.button("Analyze Stock"):
            st.query_params["ticker"] = selected_option.split(" - ")[0]
            st.rerun()
    else: st.warning("No matching stocks found.")

# --- URL & STATE ---
if "ticker" not in st.query_params: st.query_params["ticker"] = "AAPL"
ticker = st.query_params["ticker"]
if 'val_method' not in st.session_state: st.session_state.val_method = "Discounted Cash Flow (DCF)"

# --- FETCH DATA ---
news_list = [] 
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    if not current_price:
        st.error(f"Ticker '{ticker}' not found.")
        st.stop()
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

def create_gauge(val, min_v, max_v, title, color="#00d09c", suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14, 'color': '#8c97a7'}},
        number={'suffix': suffix, 'font': {'size': 20}},
        gauge={'axis': {'range': [min_v, max_v]}, 'bar': {'color': color}, 
               'bgcolor': "#2c3542", 'borderwidth': 0}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=160, margin=dict(t=30, b=10, l=20, r=20))
    return fig

def calc_graham(info):
    eps = info.get('trailingEps', 0); bv = info.get('bookValue', 0)
    if eps > 0 and bv > 0: return np.sqrt(22.5 * eps * bv)
    return 0

def calc_dcf(stock, info):
    try:
        if 'Free Cash Flow' in cash_flow.columns: fcf = cash_flow['Free Cash Flow'].iloc[0]
        else: fcf = cash_flow['Total Cash From Operating Activities'].iloc[0] + cash_flow['Capital Expenditures'].iloc[0]
        gr = min(info.get('earningsGrowth', 0.10) or 0.08, 0.20)
        dr = 0.09; flows = []
        for i in range(1, 6): flows.append((fcf * ((1 + gr) ** i)) / ((1 + dr) ** i))
        term = (fcf * ((1 + gr) ** 5) * 1.025) / (dr - 0.025)
        dcf = (sum(flows) + (term / ((1 + dr) ** 5))) / info.get('sharesOutstanding', 1)
        return dcf, gr
    except: return 0, 0

# --- VARIABLES ---
div_rate = info.get('dividendRate', 0)
dy = (div_rate / current_price) if (div_rate and current_price) else (info.get('dividendYield', 0) or 0)
roe = info.get('returnOnEquity', 0) or 0
peg = info.get('pegRatio', 0) or 0
de = info.get('debtToEquity', 0) or 0
pe = info.get('trailingPE', 0) or 0
beta = info.get('beta', 1.0) or 1.0

graham_fv = calc_graham(info)
dcf_fv, dcf_growth = calc_dcf(stock, info)
analyst_fv = info.get('targetMeanPrice', 0)
val_method = st.session_state.val_method

if val_method == "Discounted Cash Flow (DCF)":
    fair_value = dcf_fv; calc_desc = f"2-Stage DCF (Gr: {dcf_growth*100:.1f}%)"
elif val_method == "Graham Formula":
    fair_value = graham_fv; calc_desc = "Graham Number"
else:
    fair_value = analyst_fv; calc_desc = "Analyst Target"
if fair_value == 0 or np.isnan(fair_value): fair_value = current_price

# --- SCORING (STRICT 6-POINT) ---
def check(condition, text): return (1, f"✅ {text}") if condition else (0, f"❌ {text}")

# 1. VALUATION
v_score = 0; v_details = []
s,t = check(current_price < fair_value, "Below Fair Value"); v_score+=s; v_details.append(t)
s,t = check(current_price < fair_value*0.8, "Significantly Below (>20%)"); v_score+=s; v_details.append(t)
s,t = check(pe > 0 and pe < 25, f"P/E ({pe:.1f}x) < 25x"); v_score+=s; v_details.append(t)
s,t = check(pe > 0 and pe < 35, f"P/E ({pe:.1f}x) < Peers (35x)"); v_score+=s; v_details.append(t)
s,t = check(peg > 0 and peg < 1.5, f"PEG ({peg:.2f}) < 1.5"); v_score+=s; v_details.append(t)
s,t = check(current_price < analyst_fv, "Below Analyst Target"); v_score+=s; v_details.append(t)

# 2. FUTURE
f_score = 0; f_details = []
f_eps = info.get('forwardEps', 0) or 0; t_eps = info.get('trailingEps', 0) or 0
if peg > 0 and pe > 0: g_rate = (pe / peg) / 100
elif f_eps > 0 and t_eps > 0: g_rate = (f_eps - t_eps) / t_eps
else: g_rate = info.get('earningsGrowth', 0) or 0
rev_g = info.get('revenueGrowth', 0) or 0

s,t = check(g_rate > 0.02, f"Earnings ({g_rate*100:.1f}%) > Savings"); f_score+=s; f_details.append(t)
s,t = check(g_rate > 0.10, "Earnings > Market (10%)"); f_score+=s; f_details.append(t)
s,t = check(g_rate > 0.20, "High Growth Earnings (>20%)"); f_score+=s; f_details.append(t)
s,t = check(rev_g > 0.10, f"Revenue ({rev_g*100:.1f}%) > Market"); f_score+=s; f_details.append(t)
s,t = check(rev_g > 0.20, "High Growth Revenue (>20%)"); f_score+=s; f_details.append(t)
s,t = check(roe > 0.20, f"High ROE ({roe*100:.1f}%)"); f_score+=s; f_details.append(t)

# 3. PAST
p_score = 0; p_details = []
eps_growth_1y = 0
try:
    hfin = financials.sort_index(); hbs = balance_sheet.sort_index()
    if not hfin.empty and len(hfin)>=2:
        curr_eps = hfin['Basic EPS'].iloc[-1] if 'Basic EPS' in hfin.columns else hfin['Net Income'].iloc[-1]/info.get('sharesOutstanding',1)
        prev_eps = hfin['Basic EPS'].iloc[-2] if 'Basic EPS' in hfin.columns else hfin['Net Income'].iloc[-2]/info.get('sharesOutstanding',1)
        old_eps = hfin['Basic EPS'].iloc[0] if 'Basic EPS' in hfin.columns else hfin['Net Income'].iloc[0]/info.get('sharesOutstanding',1)
        eps_growth_1y = (curr_eps - prev_eps) / abs(prev_eps) if prev_eps!=0 else 0
        
        s,t = check(eps_growth_1y > 0.12, "EPS Growth > 12%"); p_score+=s; p_details.append(t)
        s,t = check(curr_eps > old_eps, "Long Term Growth"); p_score+=s; p_details.append(t)
        cagr = (curr_eps/old_eps)**(1/(len(hfin)-1)) - 1 if old_eps>0 else 0
        s,t = check(eps_growth_1y > cagr, "Accelerating Growth"); p_score+=s; p_details.append(t)
        s,t = check(roe > 0.20, "High ROE (>20%)"); p_score+=s; p_details.append(t)
        
        curr_roce = hfin['EBIT'].iloc[-1] / (hbs['Total Assets'].iloc[-1] - hbs['Current Liabilities'].iloc[-1]) if 'EBIT' in hfin.columns else 0
        old_roce = hfin['EBIT'].iloc[0] / (hbs['Total Assets'].iloc[0] - hbs['Current Liabilities'].iloc[0]) if 'EBIT' in hfin.columns else 0
        s,t = check(curr_roce > old_roce, "ROCE Improving"); p_score+=s; p_details.append(t)
        s,t = check(info.get('returnOnAssets', 0) > 0.06, "ROA > 6%"); p_score+=s; p_details.append(t)
    else: p_details.append("❌ Insufficient Data")
except: p_details.append("❌ Data Error")

# 4. HEALTH
h_score = 0; h_details = []
try:
    ca = get_val(balance_sheet, ['Current Assets', 'Total Current Assets'])
    cl = get_val(balance_sheet, ['Current Liabilities', 'Total Current Liabilities'])
    tl = get_val(balance_sheet, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
    td = get_debt(balance_sheet)
    eq = get_val(balance_sheet, ['Stockholders Equity', 'Total Stockholder Equity'])
    c = get_val(balance_sheet, ['Cash And Cash Equivalents', 'Cash'])
    ebit = get_val(financials, ['EBIT', 'Operating Income'])
    int_exp = abs(get_val(financials, ['Interest Expense']))
    ocf = get_val(cash_flow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
    
    if ca > 0 and cl > 0: s,t=check(ca > cl, "Short Term Assets > Liab"); h_score+=s; h_details.append(t)
    else: h_details.append("❌ Short Term (N/A)")
    if ca > 0: s,t=check(ca > (tl - cl), "Short Term > Long Term Liab"); h_score+=s; h_details.append(t)
    else: h_details.append("❌ Long Term (N/A)")
    
    der = td/eq if eq!=0 else 999
    s,t=check((der < 0.40) or (c > td), f"Safe Debt (D/E: {der*100:.0f}%)"); h_score+=s; h_details.append(t)
    
    if len(balance_sheet.columns)>1:
        p_td = get_debt(pd.DataFrame(balance_sheet.iloc[:,1]))
        p_eq = get_val(pd.DataFrame(balance_sheet.iloc[:,1]), ['Stockholders Equity'])
        p_der = p_td/p_eq if p_eq!=0 else 999
        s,t=check(der < p_der, "Reducing Debt"); h_score+=s; h_details.append(t)
    else: h_details.append("❌ Reducing Debt (N/A)")
    
    if td>0: s,t=check(ocf > (td*0.2), "Debt Coverage"); h_score+=s; h_details.append(t)
    else: s,t=1,"✅ Debt Coverage (No Debt)"; h_score+=s; h_details.append(t)
    
    if int_exp>0: s,t=check(ebit > (int_exp*5), "Interest Coverage"); h_score+=s; h_details.append(t)
    else: s,t=1,"✅ Interest Coverage (No Int)"; h_score+=s; h_details.append(t)
except: h_score=3; h_details.append("❌ Data Error")

# 5. DIVIDEND
d_score = 0; d_details = []
notable = dy > 0.015
s,t=check(notable, f"Notable >1.5% ({dy*100:.2f}%)"); d_score+=s; d_details.append(t)
s,t=check(dy > 0.035, "High >3.5%"); d_score+=s; d_details.append(t)
stable=False; growing=False
try:
    if len(div_history)>20:
        if div_history.iloc[-1] >= div_history.iloc[-20]: stable=True
        if div_history.iloc[-1] > div_history.iloc[-20]: growing=True
except: pass
if notable:
    s,t=check(stable, "Stable (10y)"); d_score+=s; d_details.append(t)
    s,t=check(growing, "Growing (10y)"); d_score+=s; d_details.append(t)
else:
    d_details.append("❌ Stable (Yield too low)"); d_details.append("❌ Growing (Yield too low)")
payout = info.get('payoutRatio', 0) or 0
s,t=check(payout < 0.90 and dy > 0, f"Earnings Coverage ({payout*100:.0f}%)"); d_score+=s; d_details.append(t)
try:
    dp = abs(get_val(cash_flow, ['Cash Dividends Paid']))
    fcf_val = get_val(cash_flow, ['Free Cash Flow'])
    s,t=check(dp < fcf_val and dy > 0, "Cash Flow Coverage"); d_score+=s; d_details.append(t)
except: d_details.append("❌ Cash Flow Coverage (N/A)")

scores = [v_score, f_score, p_score, h_score, d_score]
total_score = sum(scores)
if total_score < 12: color = "#ff4b4b"
elif total_score < 20: color = "#ffb300"
else: color = "#00d09c"
def hex_to_rgba(h, a): return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (a,)
fill_rgba = f"rgba{hex_to_rgba(color, 0.4)}"

# --- UI LAYOUT ---
st.markdown(f"### {info.get('shortName', ticker)} ({ticker})")
st.write(info.get('longBusinessSummary', '')[:400] + "...")

# --- HERO ROW (Metrics + Snowflake) ---
c1, c2 = st.columns([1, 1])

with c1:
    # New Gauge Row
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(create_gauge(beta, 0, 3, "Beta (Volatility)", suffix="x"), use_container_width=True)
    g2.plotly_chart(create_gauge(info.get('marketCap',0)/1e9, 0, 3000, "Market Cap", suffix="B", color="#36a2eb"), use_container_width=True)
    g3.plotly_chart(create_gauge(current_price, 0, current_price*1.5, "Current Price"), use_container_width=True)
    
    # Key Stats
    st.markdown("#### Key Metrics")
    k1, k2 = st.columns(2)
    k1.metric("P/E Ratio", f"{pe:.1f}x")
    k1.metric("PEG Ratio", f"{peg:.2f}")
    k2.metric("EPS (TTM)", f"${info.get('trailingEps',0)}")
    k2.metric("Dividend Yield", f"{dy*100:.2f}%")

with c2:
    # Snowflake
    r_vals = scores + [scores[0]]
    t_vals = ['Value', 'Future', 'Past', 'Health', 'Dividend', 'Value']
    fig = go.Figure(data=go.Scatterpolar(r=r_vals, theta=t_vals, fill='toself', line_shape='spline', line_color=color, fillcolor=fill_rgba, hoverinfo='text', text=[f"{s}/6" for s in r_vals], marker=dict(size=5)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 6], tickvals=[1,2,3,4,5,6], showticklabels=False, gridcolor='#444', gridwidth=1.5), angularaxis=dict(direction='clockwise', rotation=90, gridcolor='rgba(0,0,0,0)', tickfont=dict(color='white')), bgcolor='#232b36'), paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=30, l=30, r=30), showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📊 Analysis Breakdown"):
        t1,t2,t3,t4,t5=st.tabs(["Val","Fut","Pst","Hlt","Div"])
        with t1: [st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True) for x in v_details]
        with t2: [st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True) for x in f_details]
        with t3: [st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True) for x in p_details]
        with t4: [st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True) for x in h_details]
        with t5: [st.markdown(f"<div class='check-item'>{x}</div>", unsafe_allow_html=True) for x in d_details]

st.divider()

# --- PRICE HISTORY (EXPANDED) ---
with st.container(border=True):
    st.subheader("Price History")
    chart_ph = st.empty()
    
    # Static Keys
    tf_keys = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
    
    # Calc Returns
    ph_data = stock.history(period="max", interval="1d")
    if not ph_data.empty:
        if ph_data.index.tz is not None: ph_data.index = ph_data.index.tz_localize(None)
        curr_c = ph_data['Close'].iloc[-1]
    else: curr_c = 0
    
    def get_ret_fmt(days, fixed=None):
        try:
            if fixed: 
                idx = ph_data.index.get_indexer([pd.to_datetime(fixed)], method='nearest')[0]
                p = ph_data['Close'].iloc[idx]
            else: p = ph_data['Close'].iloc[-days]
            ret = ((curr_c - p)/p)*100
            sign = "+" if ret >=0 else ""
            return f"({sign}{ret:.1f}%)"
        except: return ""
    
    # Labels
    tf_labels = {}
    ytd_d = datetime(datetime.now().year, 1, 1)
    ret_1d = "(-)"
    if not ph_data.empty: ret_1d = get_ret_fmt(2)
    
    tf_labels["1D"] = f"1D {ret_1d}"
    tf_labels["5D"] = f"5D {get_ret_fmt(6)}"
    tf_labels["1M"] = f"1M {get_ret_fmt(22)}"
    tf_labels["6M"] = f"6M {get_ret_fmt(126)}"
    tf_labels["YTD"] = f"YTD {get_ret_fmt(0, ytd_d)}"
    tf_labels["1Y"] = f"1Y {get_ret_fmt(252)}"
    tf_labels["5Y"] = f"5Y {get_ret_fmt(1260)}"
    tf_labels["Max"] = f"Max {get_ret_fmt(len(ph_data)-1)}"
    
    def fmt_func(opt): return tf_labels.get(opt, opt)
    if 'tf_sel' not in st.session_state: st.session_state.tf_sel = '1D'
    def update_tf(): pass
    
    timeframe = st.radio("TF", tf_keys, format_func=fmt_func, horizontal=True, label_visibility="collapsed", key="tf_sel", on_change=update_tf)
    
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
        fig_p.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='#36a2eb' if timeframe in ['1D','5D'] else '#00d09c', width=2), fill='tozeroy', fillcolor=f"rgba(0,208,156,0.1)" if timeframe not in ['1D','5D'] else "rgba(54,162,235,0.1)", hovertemplate='<b>%{x|%b %d %H:%M}</b><br>$%{y:.2f}<extra></extra>'))
        
        xa = dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikecolor="white", spikethickness=1, gridcolor='#36404e')
        if x_rng: xa['range'] = x_rng
        fig_p.update_xaxes(**xa)
        fig_p.update_yaxes(range=y_rng, showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash', spikecolor="white", spikethickness=1, gridcolor='#36404e')
        fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified", hoverlabel=dict(bgcolor="#2c3542"))
        chart_ph.plotly_chart(fig_p, use_container_width=True)
    else: chart_ph.write("Chart Data Unavailable")

st.divider()

# --- 1. VALUATION ---
st.header("1. Valuation")
c_val1, c_val2 = st.columns([2, 1])
with c_val1:
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
    # Bullet chart for ROE
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
        fig_pay.update_layout(showlegend=False, height=250, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), annotations=[dict(text=f"{payout*100:.0f}%", x=0.5, y=0.5, font_size=20, showarrow=False)])
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

# Footer Spacer
st.markdown("<br><br><br>", unsafe_allow_html=True)
