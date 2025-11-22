import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
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
    .stRadio > div { display: flex; justify-content: flex-start; gap: 10px; }
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

# --- PRICE HISTORY ---
st.header("Price History")
chart_type = st.radio("Select Timeframe:", ["Long Term (Daily)", "Short Term (Intraday)"], horizontal=True, label_visibility="collapsed")

if chart_type == "Short Term (Intraday)":
    hist_data = stock.history(period="5d", interval="15m")
    buttons = list([dict(count=1, label="1d", step="day", stepmode="backward"), dict(count=5, label="5d", step="day", stepmode="backward"), dict(step="all", label="Show All")])
    line_color = '#36a2eb'
else:
    hist_data = stock.history(period="max", interval="1d")
    buttons = list([dict(count=1, label="1m", step="month", stepmode="backward"), dict(count=6, label="6m", step="month", stepmode="backward"), dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label="1y", step="year", stepmode="backward"), dict(count=5, label="5y", step="year", stepmode="backward"), dict(step="all", label="MAX")])
    line_color = '#00d09c'

if not hist_data.empty:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], mode='lines', name='Close', line=dict(color=line_color, width=2), fill='tozeroy', fillcolor=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)"))
    fig_price.update_xaxes(rangeslider_visible=True, rangeselector=dict(buttons=buttons, bgcolor="#2c3542", activecolor=line_color, font=dict(color="white")))
    fig_price.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis=dict(gridcolor='#36404e'), yaxis=dict(gridcolor='#36404e'), height=400, margin=dict(l=0, r=0))
    st.plotly_chart(fig_price, use_container_width=True)

st.divider()

# --- 1. VALUATION ---
st.header("1. Valuation")
v_col_left, v_col_right = st.columns([2, 1])
with v_col_left:
    st.subheader("Share Price vs Fair Value")
    diff = ((current_price - fair_value) / fair_value) * 100
    status_color = "#ff6384" if diff > 20 else "#ffce56" if diff > -20 else "#00d09c"
    status_text = "Overvalued" if diff > 0 else "Undervalued"
    st.markdown(f"The stock is trading at **${current_price}**. Fair Value: **${fair_value:.2f}**. It is <span style='color:{status_color}; font-weight:bold;'>{abs(diff):.1f}% {status_text}</span>.<br><small style='color:#8c97a7'>Method: {calc_desc}</small>", unsafe_allow_html=True)
    max_val = max(current_price, fair_value) * 1.3
    fig_val = go.Figure()
    fig_val.add_vrect(x0=0, x1=fair_value*0.8, fillcolor="#00d09c", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*0.8, x1=fair_value*1.2, fillcolor="#ffce56", opacity=0.15, layer="below", line_width=0)
    fig_val.add_vrect(x0=fair_value*1.2, x1=max_val*1.5, fillcolor="#ff6384", opacity=0.15, layer="below", line_width=0)
    fig_val.add_trace(go.Bar(y=[""], x=[current_price], name="Current", orientation='h', marker_color='#36a2eb', width=0.3, text=f"${current_price}", textposition='auto'))
    fig_val.add_trace(go.Bar(y=[""], x=[fair_value], name="Fair Value", orientation='h', marker_color='#232b36', marker_line_color='white', marker_line_width=2, width=0.3, text=f"${fair_value:.2f}", textposition='auto'))
    fig_val.update_layout(xaxis=dict(range=[0, max_val], visible=False), yaxis=dict(visible=False), barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=150, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig_val, use_container_width=True)
with v_col_right:
    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}")
    st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
    st.metric("Graham Number", f"${graham_fv:.2f}" if graham_fv else "N/A")

st.divider()

# --- NEW: FINANCIALS SECTION ---
st.header("Financials")

# Toggle for Annual vs Quarterly
fin_period = st.radio("Frequency:", ["Annual", "Quarterly"], horizontal=True)

# Fetch Data based on selection
if fin_period == "Annual":
    financials = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T
    date_fmt = "%Y"
else:
    financials = stock.quarterly_financials.T
    balance_sheet = stock.quarterly_balance_sheet.T
    cash_flow = stock.quarterly_cashflow.T
    date_fmt = "%Y-%m"

# Reverse to show oldest to newest
financials = financials.iloc[::-1]
balance_sheet = balance_sheet.iloc[::-1]
cash_flow = cash_flow.iloc[::-1]
dates = [d.strftime(date_fmt) for d in financials.index]

# 1. Performance (Rev vs Net Income vs Margin)
st.subheader("Performance")
if not financials.empty:
    rev = financials.get('Total Revenue', financials.get('Revenue', []))
    net_inc = financials.get('Net Income', [])
    
    if len(rev) > 0 and len(net_inc) > 0:
        margin = (net_inc / rev) * 100
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(x=dates, y=rev, name='Revenue', marker_color='#36a2eb'))
        fig_perf.add_trace(go.Bar(x=dates, y=net_inc, name='Net Income', marker_color='#00d09c'))
        fig_perf.add_trace(go.Scatter(x=dates, y=margin, name='Net Margin %', yaxis='y2', mode='lines+markers', line=dict(color='white', width=2)))
        
        fig_perf.update_layout(
            yaxis=dict(showgrid=True, gridcolor='#36404e', title="Amount"),
            yaxis2=dict(title="Margin %", overlaying='y', side='right', showgrid=False),
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

# 2. Revenue to Profit (Waterfall) - Latest Period
st.subheader("Revenue to Profit Conversion (Latest)")
if not financials.empty:
    latest = financials.iloc[-1]
    rev_val = latest.get('Total Revenue', latest.get('Revenue', 0))
    cost_rev = latest.get('Cost Of Revenue', 0)
    gross_profit = latest.get('Gross Profit', 0)
    op_exp = latest.get('Operating Expense', 0)
    net_val = latest.get('Net Income', 0)
    
    # Calculate implicit "Other" to make waterfall balance
    # Revenue - COGS - OpEx - Other = Net Income
    # Other = Revenue - COGS - OpEx - Net Income
    other_exp = rev_val - cost_rev - op_exp - net_val
    
    fig_water = go.Figure(go.Waterfall(
        orientation = "v",
        measure = ["relative", "relative", "total", "relative", "relative", "total"],
        x = ["Revenue", "COGS", "Gross Profit", "Op Expenses", "Other/Tax", "Net Income"],
        textposition = "outside",
        text = [f"{rev_val/1e9:.1f}B", f"-{cost_rev/1e9:.1f}B", f"{gross_profit/1e9:.1f}B", f"-{op_exp/1e9:.1f}B", f"-{other_exp/1e9:.1f}B", f"{net_val/1e9:.1f}B"],
        y = [rev_val, -cost_rev, gross_profit, -op_exp, -other_exp, net_val],
        connector = {"line":{"color":"white"}},
        decreasing = {"marker":{"color":"#ff6384"}},
        increasing = {"marker":{"color":"#00d09c"}},
        totals = {"marker":{"color":"#36a2eb"}}
    ))
    fig_water.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        yaxis=dict(showgrid=True, gridcolor='#36404e')
    )
    st.plotly_chart(fig_water, use_container_width=True)

# 3. Debt Level & Coverage
st.subheader("Debt Level and Coverage")
if not balance_sheet.empty and not cash_flow.empty:
    # Align dates
    common_idx = balance_sheet.index.intersection(cash_flow.index)
    bs_align = balance_sheet.loc[common_idx]
    cf_align = cash_flow.loc[common_idx]
    d_dates = [d.strftime(date_fmt) for d in common_idx]
    
    debt = bs_align.get('Total Debt', [])
    cash = bs_align.get('Cash And Cash Equivalents', [])
    fcf = cf_align.get('Free Cash Flow', [])
    
    fig_debt = go.Figure()
    fig_debt.add_trace(go.Bar(x=d_dates, y=debt, name='Total Debt', marker_color='#ff6384'))
    fig_debt.add_trace(go.Bar(x=d_dates, y=fcf, name='Free Cash Flow', marker_color='#00d09c'))
    fig_debt.add_trace(go.Bar(x=d_dates, y=cash, name='Cash', marker_color='#36a2eb'))
    
    fig_debt.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        yaxis=dict(showgrid=True, gridcolor='#36404e'),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_debt, use_container_width=True)

# 4. Earnings (Actual vs Estimate)
st.subheader("Earnings (EPS): Actual vs Estimate")
try:
    earn_hist = stock.earnings_history
    if earn_hist is not None and not earn_hist.empty:
        # Sort by date
        earn_hist = earn_hist.sort_values(by="startdatetime")
        e_dates = [d.strftime("%Y-%m-%d") for d in earn_hist["startdatetime"]]
        actual = earn_hist["epsactual"]
        estimate = earn_hist["epsestimate"]
        
        fig_earn = go.Figure()
        fig_earn.add_trace(go.Scatter(x=e_dates, y=estimate, mode='markers', name='Estimate', marker=dict(color='gray', size=10, symbol='circle-open')))
        fig_earn.add_trace(go.Scatter(x=e_dates, y=actual, mode='markers', name='Actual', marker=dict(color='#00d09c', size=10)))
        
        fig_earn.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#36404e'),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_earn, use_container_width=True)
    else:
        st.write("Earnings history data unavailable.")
except:
    st.write("Earnings history data unavailable.")

st.divider()

# --- 2. FUTURE & PAST (Remaining sections) ---
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
    st.metric("Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
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
