import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Simply Clone", page_icon="❄️")

# Custom CSS to mimic Simply Wall St (Dark Mode & Green Accents)
st.markdown("""
    <style>
    .stApp { background-color: #1b222d; color: white; }
    h1, h2, h3 { color: #00d09c !important; }
    .stMetricValue { font-size: 1.5rem !important; color: white !important; }
    .css-1d391kg { background-color: #232b36; border-radius: 15px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR SEARCH ---
with st.sidebar:
    st.title("❄️ Simply Clone")
    ticker_input = st.text_input("Enter Ticker", value="AAPL").upper()
    if st.button("Analyze"):
        st.session_state.ticker = ticker_input

ticker = st.session_state.get("ticker", "AAPL")

# --- FETCH DATA ---
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # If ticker is invalid, info is usually empty or throws error
    if 'regularMarketPrice' not in info and 'currentPrice' not in info:
        st.error("Ticker not found. Please try again.")
        st.stop()
        
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- HELPER: CALCULATE SCORES (0-5) ---
def calculate_snowflake(info):
    scores = {"Value": 0, "Future": 0, "Past": 0, "Health": 0, "Dividend": 0}
    
    # 1. Value (PE)
    pe = info.get('trailingPE', 0)
    if pe > 0 and pe < 15: scores["Value"] = 5
    elif pe < 25: scores["Value"] = 4
    elif pe < 35: scores["Value"] = 3
    elif pe < 50: scores["Value"] = 2
    else: scores["Value"] = 1
    
    # 2. Future (PEG)
    peg = info.get('pegRatio', 0)
    if peg > 0 and peg < 1: scores["Future"] = 5
    elif peg < 2: scores["Future"] = 4
    elif peg < 4: scores["Future"] = 2
    
    # 3. Past (ROE)
    roe = info.get('returnOnEquity', 0)
    if roe > 0.2: scores["Past"] = 5
    elif roe > 0.1: scores["Past"] = 3
    else: scores["Past"] = 1
    
    # 4. Health (Debt to Equity)
    de = info.get('debtToEquity', 0)
    if de < 50: scores["Health"] = 5 # < 0.5 ratio
    elif de < 100: scores["Health"] = 3
    else: scores["Health"] = 1
    
    # 5. Dividend (Yield)
    dy = info.get('dividendYield', 0)
    if dy and dy > 0.03: scores["Dividend"] = 5
    elif dy and dy > 0.01: scores["Dividend"] = 3
    
    return scores

scores = calculate_snowflake(info)

# --- HEADER SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title(f"{info.get('shortName', ticker)}")
    st.markdown(f"**{info.get('sector', 'Unknown Sector')}** | {info.get('industry', '')}")
    st.write(info.get('longBusinessSummary', 'No description available.')[:400] + "...")
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${info.get('currentPrice', 0)}")
    m2.metric("Market Cap", f"${(info.get('marketCap', 0)/1e9):.1f}B")
    m3.metric("Beta", f"{info.get('beta', 0):.2f}")
    m4.metric("EPS (TTM)", f"{info.get('trailingEps', 0)}")

with col2:
    # --- SNOWFLAKE CHART (PLOTLY) ---
    fig = go.Figure(data=go.Scatterpolar(
        r=[scores['Value'], scores['Future'], scores['Past'], scores['Health'], scores['Dividend']],
        theta=['Value', 'Future', 'Past', 'Health', 'Dividend'],
        fill='toself',
        line_color='#00d09c',
        fillcolor='rgba(0, 208, 156, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 5]),
            bgcolor='#232b36'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(color='white'),
        showlegend=False,
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- DETAILED SECTIONS ---

# 1. VALUATION
st.header("1. Valuation")
v1, v2 = st.columns(2)
with v1:
    st.info(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    st.write(f"Compare this to the industry average or the US market (~20x). A P/E of {info.get('trailingPE',0)} suggests the stock is {'expensive' if info.get('trailingPE',0) > 25 else 'good value'} based on earnings.")
with v2:
    if info.get('targetMeanPrice'):
        st.metric("Analyst Target Price", f"${info.get('targetMeanPrice')}")
        target_diff = ((info['targetMeanPrice'] - info['currentPrice']) / info['currentPrice']) * 100
        st.progress(min(max(target_diff + 50, 0), 100) / 100)
        st.caption(f"Analysts see {target_diff:.1f}% upside/downside")

# 2. FUTURE GROWTH
st.header("2. Future Growth")
f1, f2 = st.columns(2)
with f1:
    st.metric("PEG Ratio", info.get('pegRatio', 'N/A'), help="Below 1.0 is considered undervalued based on growth.")
    st.write("The PEG ratio accounts for the growth rate. A low PEG suggests you are paying less for future growth.")
with f2:
    st.metric("Revenue Growth (YoY)", f"{info.get('revenueGrowth', 0)*100:.1f}%")

# 3. PAST PERFORMANCE
st.header("3. Past Performance")
try:
    fin = stock.financials.T
    if not fin.empty:
        # Sort by date ascending for the chart
        fin = fin.sort_index()
        st.subheader("Revenue vs Earnings (Last 4 Years)")
        
        # Simple Bar Chart
        fig_past = go.Figure()
        fig_past.add_trace(go.Bar(x=fin.index, y=fin['Total Revenue'], name='Revenue', marker_color='#36a2eb'))
        fig_past.add_trace(go.Bar(x=fin.index, y=fin['Net Income'], name='Earnings', marker_color='#00d09c'))
        fig_past.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_past, use_container_width=True)
    else:
        st.write("No historical financial data available.")
except:
    st.write("Could not load historical charts.")

# 4. FINANCIAL HEALTH
st.header("4. Financial Health")
h1, h2 = st.columns(2)
with h1:
    cash = info.get('totalCash', 0)
    debt = info.get('totalDebt', 0)
    st.metric("Total Cash", f"${cash/1e9:.2f}B")
    st.metric("Total Debt", f"${debt/1e9:.2f}B")
with h2:
    debt_equity = info.get('debtToEquity', 'N/A')
    st.write(f"**Debt to Equity Ratio:** {debt_equity}")
    if debt > cash:
        st.warning("⚠️ This company has more debt than cash.")
    else:
        st.success("✅ This company has more cash than debt.")

# 5. DIVIDEND
st.header("5. Dividend")
d1, d2 = st.columns(2)
with d1:
    yld = info.get('dividendYield', 0)
    st.metric("Dividend Yield", f"{yld*100:.2f}%" if yld else "0%")
with d2:
    payout = info.get('payoutRatio', 0)
    st.metric("Payout Ratio", f"{payout*100:.2f}%" if payout else "N/A")
    st.write("A high payout ratio (>90%) may indicate the dividend is not sustainable.")

# 6. MANAGEMENT
st.header("6. Management")
officers = stock.info.get('companyOfficers', [])
if officers:
    # Create a clean dataframe
    mgmt_data = []
    for o in officers[:5]: # Top 5
        mgmt_data.append({
            "Name": o.get('name'),
            "Title": o.get('title'),
            "Pay": f"${o.get('totalPay', 0):,}" if o.get('totalPay') else "N/A"
        })
    st.table(pd.DataFrame(mgmt_data))
else:
    st.write("Management data not available.")

# 7. OWNERSHIP
st.header("7. Ownership")
o1, o2 = st.columns(2)
with o1:
    insider = info.get('heldPercentInsiders', 0)
    st.metric("Insider Ownership", f"{insider*100:.2f}%" if insider else "N/A")
with o2:
    inst = info.get('heldPercentInstitutions', 0)
    st.metric("Institutional Ownership", f"{inst*100:.2f}%" if inst else "N/A")

holders = stock.major_holders
if holders is not None:
    st.write("Top Holders breakdown available in backend (dataframe hidden for cleanliness).")