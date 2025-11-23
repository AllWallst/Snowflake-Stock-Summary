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
    
    .perf-container {
        display: grid;
        grid-template-columns: repeat(8, 1fr);
        gap: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
        background-color: #232b36;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .perf-item { display: flex; flex-direction: column; }
    .perf-label { color: #8c97a7; font-size: 0.8rem; margin-bottom: 5px; }
    .perf-val { font-weight: bold; font-size: 1rem; }
    .pos { color: #00d09c; }
    .neg { color: #ff6384; }
    
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

# --- URL & STATE MANAGEMENT ---
if "ticker" not in st.query_params:
    st.query_params["ticker"] = "AAPL"

current_ticker = st.query_params["ticker"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("📡 MarketRadar")
    
    st.markdown("### 🔎 Symbol Lookup")
    exchange = st.selectbox("Market / Region", ["All / US", "Canada (TSX) .TO", "Canada (Venture) .V", "UK (London) .L", "Australia .AX", "India .NS"])
    search_query = st.text_input("Search Company or Ticker", placeholder="e.g. Shopify, Apple...")
    
    if search_query:
        search_results = search_symbol(search_query)
        if "Canada (TSX)" in exchange: search_results = [x for x in search_results if ".TO" in x[1]]
        elif "Venture" in exchange: search_results = [x for x in search_results if ".V" in x[1]]
            
        if search_results:
            selected_option = st.selectbox("Select Stock:", options=[x[0] for x in search_results], key="search_select")
            if st.button("Go"):
                st.query_params["ticker"] = selected_option.split(" - ")[0]
                st.rerun()
        else:
            st.caption("No matching stocks found.")

    st.divider()
    st.markdown("### ⚙️ Settings")
    val_method = st.radio("Fair Value Method", ["Discounted Cash Flow (DCF)", "Graham Formula", "Analyst Target"])

ticker = st.query_params["ticker"]

# --- FETCH DATA ---
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    
    if not current_price:
        st.error(f"Ticker '{ticker}' not found. Try searching for the company name in the sidebar.")
        st.stop()
        
    # Fetch Financials Early for Scoring
    financials = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T
    
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- CALCULATION FUNCTIONS ---
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

# --- DATA PREP: SAFE DIVIDEND YIELD ---
div_rate = info.get('dividendRate', 0)
if div_rate and current_price and current_price > 0:
    dy = div_rate / current_price
else:
    dy = info.get('dividendYield', 0)

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

# --- SWS-STYLE CHECKLIST SCORING ENGINE ---
# Each pillar calculates a score out of 6 based on specific boolean checks.
# The final score is normalized to 5 for the chart.

# 1. VALUATION (0-6)
v_score = 0
if current_price < fair_value: v_score += 1       # Below Fair Value
if current_price < fair_value * 0.8: v_score += 1 # Significantly Below
if info.get('trailingPE', 99) < 25: v_score += 1  # vs Market (Approx 25)
if info.get('trailingPE', 99) < 35: v_score += 1  # vs Peers/Industry (Proxy)
if info.get('pegRatio', 5) < 1.5: v_score += 1    # PEG (Fair Ratio)
if current_price < analyst_fv: v_score += 1       # Analyst Forecast

# 2. FUTURE (0-6)
f_score = 0
g_rate = info.get('earningsGrowth', 0)
if g_rate > 0.02: f_score += 1  # vs Savings Rate
if g_rate > 0.10: f_score += 1  # vs Market
if g_rate > 0.20: f_score += 1  # High Growth
rev_g = info.get('revenueGrowth', 0)
if rev_g > 0.10: f_score += 1   # Rev vs Market
if rev_g > 0.20: f_score += 1   # High Rev Growth
if info.get('returnOnEquity', 0) > 0.20: f_score += 1 # Future ROE (Proxy with current)

# 3. PAST (0-6)
p_score = 0
try:
    ni = financials['Net Income'].iloc[0] if not financials.empty else 0
    ocf = cash_flow['Total Cash From Operating Activities'].iloc[0] if not cash_flow.empty else 0
    # Check 1: Quality Earnings
    if ocf > ni: p_score += 1
    # Check 2: Profit Margin Growth
    curr_rev = financials['Total Revenue'].iloc[0]
    prev_rev = financials['Total Revenue'].iloc[1]
    curr_ni = financials['Net Income'].iloc[0]
    prev_ni = financials['Net Income'].iloc[1]
    if (curr_ni/curr_rev) > (prev_ni/prev_rev): p_score += 1
    # Check 3: Earnings Trend (Positive last year)
    if curr_ni > prev_ni: p_score += 1
    # Check 4: Growth Acceleration (Current > 3yr avg - simplified)
    if curr_ni > prev_ni * 1.1: p_score += 1 
except: pass
if info.get('returnOnEquity', 0) > 0.20: p_score += 1 # High ROE
if info.get('returnOnAssets', 0) > 0.10: p_score += 1 # ROA

# 4. HEALTH (0-6)
h_score = 0
try:
    curr_assets = balance_sheet['Current Assets'].iloc[0]
    curr_liab = balance_sheet['Current Liabilities'].iloc[0]
    total_liab = balance_sheet['Total Liabilities Net Minority Interest'].iloc[0]
    total_debt = balance_sheet['Total Debt'].iloc[0]
    equity = balance_sheet['Stockholders Equity'].iloc[0]
    ebit = financials['EBIT'].iloc[0]
    interest = abs(financials['Interest Expense'].iloc[0])
    
    if curr_assets > curr_liab: h_score += 1 # Short term
    if curr_assets > total_liab * 0.5: h_score += 1 # Long term proxy
    if (total_debt/equity) < 0.40: h_score += 1 # Debt Level
    if ebit > (interest * 3): h_score += 1 # Interest Coverage
    # Debt Coverage (OCF > 20% of Debt)
    ocf = cash_flow['Total Cash From Operating Activities'].iloc[0]
    if ocf > (total_debt * 0.2): h_score += 1
    # Reducing Debt (Current < Prev)
    prev_debt = balance_sheet['Total Debt'].iloc[1]
    if total_debt < prev_debt: h_score += 1
except: h_score = 3 # Default if data missing

# 5. DIVIDEND (0-6)
d_score = 0
if dy > 0: d_score += 1         # Payer
if dy > 0.015: d_score += 1     # Notable (>1.5%)
if dy > 0.035: d_score += 1     # High (>3.5%)
if info.get('payoutRatio', 1) < 0.90: d_score += 1 # Coverage
try:
    div_paid = abs(cash_flow['Cash Dividends Paid'].iloc[0])
    fcf = cash_flow['Free Cash Flow'].iloc[0]
    if div_paid < fcf: d_score += 1 # Cash Flow Coverage
except: pass
if dy > 0.01: d_score += 1 # Stability (Proxy: just giving points if > 1%)

# Normalize to 5 for Chart
final_scores = [
    (v_score / 6) * 5,
    (f_score / 6) * 5,
    (p_score / 6) * 5,
    (h_score / 6) * 5,
    (d_score / 6) * 5
]

# --- HEADER UI ---
col1, col2 = st.columns([2, 1])

with col1:
    currency_symbol = info.get('currency', 'USD')
    st.markdown(f"""<h1>{info.get('shortName', ticker)} <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" style="text-decoration:none; font-size:0.5em;">🔗</a></h1>""", unsafe_allow_html=True)
    st.caption(f"{info.get('sector', 'Unknown')} | {info.get('industry', 'Unknown')} | {currency_symbol}")
    st.write(info.get('longBusinessSummary', '')[:400] + "...")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"{current_price:.2f}")
    m2.metric("Market Cap", f"{(info.get('marketCap',0)/1e9):.1f}B")
    m3.metric("Beta", f"{info.get('beta', 0):.2f}")
    m4.metric("PE Ratio", f"{info.get('trailingPE',0):.1f}")

with col2:
    fig = go.Figure(data=go.Scatterpolar(
        r=final_scores,
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

# --- PRICE HISTORY & RETURNS ---
st.header("Price History")

# 1. GRAPH CONTROLS
chart_type = st.radio("Select Timeframe:", ["Long Term (Daily)", "Short Term (Intraday)"], horizontal=True, label_visibility="collapsed")

start_range = None
end_range = None

if chart_type == "Short Term (Intraday)":
    hist_data = stock.history(period="5d", interval="15m", prepost=True)
    if not hist_data.empty:
        last_dt = hist_data.index[-1]
        start_range = last_dt.replace(hour=7, minute=30, second=0, microsecond=0)
        end_range = last_dt.replace(hour=18, minute=0, second=0, microsecond=0)

    buttons = list([
        dict(count=1, label="1d", step="day", stepmode="backward"),
        dict(count=5, label="5d", step="day", stepmode="backward"),
        dict(step="all", label="Show All")
    ])
    line_color = '#36a2eb'
else:
    hist_data = stock.history(period="max", interval="1d")
    buttons = list([
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(step="all", label="MAX")
    ])
    line_color = '#00d09c'

# 2. GRAPH RENDER
if not hist_data.empty:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=hist_data.index, y=hist_data['Close'], mode='lines', name='Close',
        line=dict(color=line_color, width=2),
        fill='tozeroy', fillcolor=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)",
        hovertemplate = '<b>Date:</b> %{x|%b %d, %H:%M}<br><b>Price:</b> %{y:.2f}<extra></extra>'
    ))
    xaxis_args = dict(
        rangeslider_visible=True,
        rangeselector=dict(buttons=buttons, bgcolor="#2c3542", activecolor=line_color, font=dict(color="white")),
        showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikecolor="#ffffff", spikethickness=1,
        gridcolor='#36404e'
    )
    if start_range and end_range: xaxis_args['range'] = [start_range, end_range]
    fig_price.update_xaxes(**xaxis_args)
    fig_price.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dash', spikecolor="#ffffff", spikethickness=1, gridcolor='#36404e')
    fig_price.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400, margin=dict(l=0, r=0), hovermode="x unified", hoverlabel=dict(bgcolor="#2c3542", font_size=14, font_family="Segoe UI"))
    st.plotly_chart(fig_price, use_container_width=True)
else: st.write("Historical price data unavailable.")

# 3. PERCENTAGE RETURNS
if chart_type == "Short Term (Intraday)": hist_max = stock.history(period="max", interval="1d")
else: hist_max = hist_data

if not hist_max.empty and len(hist_max) > 1:
    curr = hist_max['Close'].iloc[-1]
    hist_max.index = hist_max.index.tz_localize(None)
    def get_ret(df, days_back=None, fixed_date=None):
        try:
            if fixed_date:
                idx = df.index.get_indexer([pd.to_datetime(fixed_date)], method='nearest')[0]
                past_price = df['Close'].iloc[idx]
            else:
                if len(df) < days_back: return "N/A"
                past_price = df['Close'].iloc[-days_back]
            val = ((curr - past_price) / past_price) * 100
            color = "pos" if val >= 0 else "neg"
            return f'<span class="{color}">{val:+.2f}%</span>'
        except: return "N/A"
    ytd_date = datetime(datetime.now().year, 1, 1)
    st.markdown(f"""<div class="perf-container"><div class="perf-item"><span class="perf-label">1 Day</span><span class="perf-val">{get_ret(hist_max, 2)}</span></div><div class="perf-item"><span class="perf-label">5 Days</span><span class="perf-val">{get_ret(hist_max, 6)}</span></div><div class="perf-item"><span class="perf-label">1 Month</span><span class="perf-val">{get_ret(hist_max, 22)}</span></div><div class="perf-item"><span class="perf-label">6 Months</span><span class="perf-val">{get_ret(hist_max, 126)}</span></div><div class="perf-item"><span class="perf-label">YTD</span><span class="perf-val">{get_ret(hist_max, fixed_date=ytd_date)}</span></div><div class="perf-item"><span class="perf-label">1 Year</span><span class="perf-val">{get_ret(hist_max, 252)}</span></div><div class="perf-item"><span class="perf-label">5 Years</span><span class="perf-val">{get_ret(hist_max, 1260)}</span></div><div class="perf-item"><span class="perf-label">All Time</span><span class="perf-val">{get_ret(hist_max, days_back=len(hist_max)-1)}</span></div></div>""", unsafe_allow_html=True)

st.divider()

# --- 1. VALUATION ---
st.header("1. Valuation")
v_col_left, v_col_right = st.columns([2, 1])
with v_col_left:
    st.subheader("Share Price vs Fair Value")
    diff = ((current_price - fair_value) / fair_value) * 100
    status_color = "#ff6384" if diff > 20 else "#ffce56" if diff > -20 else "#00d09c"
    status_text = "Overvalued" if diff > 0 else "Undervalued"
    st.markdown(f"The stock is trading at **{current_price}**. Fair Value: **{fair_value:.2f}**. It is <span style='color:{status_color}; font-weight:bold;'>{abs(diff):.1f}% {status_text}</span>.<br><small style='color:#8c97a7'>Method: {calc_desc}</small>", unsafe_allow_html=True)
    max_val = max(current_price, fair_value) * 1.3
    fig_val = go.Figure()
    fig_val.add_vrect(x0=0, x1=fair_value*0.8, fillcolor="#00d09c", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*0.8, x1=fair_value*1.2, fillcolor="#ffce56", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*1.2, x1=max_val*1.5, fillcolor="#ff6384", opacity=0.15, layer="below", line_width=0)
    fig_val.add_trace(go.Bar(y=[""], x=[current_price], name="Current", orientation='h', marker_color='#36a2eb', width=0.3, text=f"{current_price}", textposition='auto'))
    fig_val.add_trace(go.Bar(y=[""], x=[fair_value], name="Fair Value", orientation='h', marker_color='#232b36', marker_line_color='white', marker_line_width=2, width=0.3, text=f"{fair_value:.2f}", textposition='auto'))
    fig_val.update_layout(xaxis=dict(range=[0, max_val], visible=False), yaxis=dict(visible=False), barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=150, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig_val, use_container_width=True)
with v_col_right:
    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}")
    st.metric("PEG Ratio", f"{info.get('pegRatio', 0):.2f}")
    st.metric("Graham Number", f"{graham_fv:.2f}" if graham_fv else "N/A")

st.divider()

# --- FINANCIALS ---
st.header("Financials")
fin_period = st.radio("Frequency:", ["Annual", "Quarterly"], horizontal=True)
if fin_period == "Annual":
    financials = financials.iloc[::-1]
    balance_sheet = balance_sheet.iloc[::-1]
    cash_flow = cash_flow.iloc[::-1]
    date_fmt = "%Y"
else:
    financials = stock.quarterly_financials.T.iloc[::-1]
    balance_sheet = stock.quarterly_balance_sheet.T.iloc[::-1]
    cash_flow = stock.quarterly_cashflow.T.iloc[::-1]
    date_fmt = "%Y-%m"
dates = [d.strftime(date_fmt) for d in financials.index]

st.subheader("Performance")
if not financials.empty:
    rev = financials.get('Total Revenue', financials.get('Revenue', []))
    net_inc = financials.get('Net Income', [])
    if len(rev) > 0:
        fig_perf = go.Figure()
        text_rev = [f"{x/1e9:.1f}B" for x in rev]
        text_inc = [f"{x/1e9:.1f}B" for x in net_inc]
        fig_perf.add_trace(go.Bar(x=dates, y=rev, name='Revenue', marker_color='#36a2eb', text=text_rev, textposition='auto'))
        fig_perf.add_trace(go.Bar(x=dates, y=net_inc, name='Net Income', marker_color='#00d09c', text=text_inc, textposition='auto'))
        fig_perf.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_perf, use_container_width=True)
        
# 2. Revenue to Profit (Waterfall)
st.subheader("Revenue to Profit Conversion (Latest)")
if not financials.empty:
    latest = financials.iloc[-1]
    rev_val = latest.get('Total Revenue', latest.get('Revenue', 0))
    cost_rev = latest.get('Cost Of Revenue', 0)
    gross_profit = latest.get('Gross Profit', 0)
    op_exp = latest.get('Operating Expense', 0)
    net_val = latest.get('Net Income', 0)
    other_exp = rev_val - cost_rev - op_exp - net_val
    fig_water = go.Figure(go.Waterfall(orientation = "v", measure = ["relative", "relative", "total", "relative", "relative", "total"], x = ["Revenue", "COGS", "Gross Profit", "Op Expenses", "Other", "Net Income"], textposition = "auto", text = [f"{rev_val/1e9:.1f}B", f"-{cost_rev/1e9:.1f}B", f"{gross_profit/1e9:.1f}B", f"-{op_exp/1e9:.1f}B", f"-{other_exp/1e9:.1f}B", f"{net_val/1e9:.1f}B"], y = [rev_val, -cost_rev, gross_profit, -op_exp, -other_exp, net_val], connector = {"line":{"color":"white"}}, decreasing = {"marker":{"color":"#ff6384"}}, increasing = {"marker":{"color":"#00d09c"}}, totals = {"marker":{"color":"#36a2eb"}}))
    fig_water.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis=dict(showgrid=True, gridcolor='#36404e'))
    st.plotly_chart(fig_water, use_container_width=True)

# 3. Debt Level
st.subheader("Debt Level and Coverage")
if not balance_sheet.empty and not cash_flow.empty:
    common_idx = balance_sheet.index.intersection(cash_flow.index)
    bs_align, cf_align = balance_sheet.loc[common_idx], cash_flow.loc[common_idx]
    d_dates = [d.strftime(date_fmt) for d in common_idx]
    debt, cash, fcf = bs_align.get('Total Debt', []), bs_align.get('Cash And Cash Equivalents', []), cf_align.get('Free Cash Flow', [])
    t_d = [f"{x/1e9:.1f}B" for x in debt]
    t_c = [f"{x/1e9:.1f}B" for x in cash]
    t_f = [f"{x/1e9:.1f}B" for x in fcf]
    fig_debt = go.Figure()
    fig_debt.add_trace(go.Bar(x=d_dates, y=debt, name='Total Debt', marker_color='#ff6384', text=t_d, textposition='auto'))
    fig_debt.add_trace(go.Bar(x=d_dates, y=fcf, name='Free Cash Flow', marker_color='#00d09c', text=t_f, textposition='auto'))
    fig_debt.add_trace(go.Bar(x=d_dates, y=cash, name='Cash', marker_color='#36a2eb', text=t_c, textposition='auto'))
    fig_debt.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis=dict(showgrid=True, gridcolor='#36404e'), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_debt, use_container_width=True)

# 4. Earnings
st.subheader("Earnings (EPS): Actual vs Estimate")
try:
    earn_hist = stock.earnings_history
    if earn_hist is not None and not earn_hist.empty:
        earn_hist = earn_hist.sort_values(by="startdatetime")
        e_dates, actual, estimate = [d.strftime("%Y-%m-%d") for d in earn_hist["startdatetime"]], earn_hist["epsactual"], earn_hist["epsestimate"]
        fig_earn = go.Figure()
        fig_earn.add_trace(go.Scatter(x=e_dates, y=estimate, mode='markers', name='Estimate', marker=dict(color='gray', size=10, symbol='circle-open')))
        fig_earn.add_trace(go.Scatter(x=e_dates, y=actual, mode='markers', name='Actual', marker=dict(color='#00d09c', size=10)))
        fig_earn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#36404e'), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_earn, use_container_width=True)
except: st.write("Earnings history data unavailable.")

st.divider()

# --- 2. FUTURE & PAST ---
c_fut, c_past = st.columns(2)
with c_fut:
    st.header("2. Future Growth")
    st.metric("Analyst Growth Est.", f"{info.get('earningsGrowth', 0)*100:.1f}%")
with c_past:
    st.header("3. Past Performance")
    st.metric("ROE", f"{roe*100:.1f}%")
    st.metric("ROA", f"{info.get('returnOnAssets', 0)*100:.1f}%")

st.divider()

# --- 4. HEALTH ---
st.header("4. Financial Health")
h1, h2 = st.columns([2, 1])
with h1:
    cash = info.get('totalCash', 0)
    debt_total = info.get('totalDebt', 0)
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(x=['Cash', 'Debt'], y=[cash, debt_total], marker_color=['#00d09c', '#ff6384'], text=[f"${cash/1e9:.1f}B", f"${debt_total/1e9:.1f}B"], textposition='auto'))
    fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(t=0, b=0, l=0, r=0), height=200)
    st.plotly_chart(fig_h, use_container_width=True)
with h2:
    st.metric("Debt to Equity", f"{de:.1f}%")
    if de > 100: st.error(f"⚠️ High Debt")
    else: st.success(f"✅ Healthy")

st.divider()

# --- 5. DIVIDEND ---
st.header("5. Dividend")
d1, d2 = st.columns(2)
with d1:
    st.metric("Yield", f"{dy*100:.2f}%")
with d2:
    st.metric("Payout Ratio", f"{info.get('payoutRatio', 0)*100:.0f}%")

st.divider()

# --- NEWS ---
st.header(f"Latest News for {ticker}")
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

news_list = stock.news
if news_list:
    for article in news_list[:5]:
        title, link, publisher, pub_time = get_news_data(article)
        date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M') if pub_time else ""
        meta_text = f"{publisher} | {date_str}" if publisher and publisher != "Unknown" else date_str
        st.markdown(f"""<div class="news-card"><a href="{link}" target="_blank" class="news-title">{title}</a><br><span class="news-meta">{meta_text}</span></div>""", unsafe_allow_html=True)
else:
    st.write("No recent news found.")
