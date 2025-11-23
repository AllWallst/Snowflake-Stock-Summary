# 📡 MarketRadar

**MarketRadar** is a visual stock analysis tool built with Python and Streamlit. Inspired by the popular "Simply Wall St" platform, it transforms complex financial data into easy-to-understand visuals.

It uses **yfinance** to fetch live market data, meaning it is completely free to use with no API keys required.

See it live here
https://allwallst.com/market-radar/

## 🌟 Features

*   **The "Snowflake" Analysis:** A 5-axis radar chart visualizing Value, Future Growth, Past Performance, Financial Health, and Dividends.
*   **Fair Value Calculator:**
    *   **Discounted Cash Flow (DCF):** A 2-stage model projecting 5 years of growth + terminal value.
    *   **Graham Formula:** Benjamin Graham's classic intrinsic value formula.
    *   **Analyst Targets:** Aggregated price targets from Wall St analysts.
*   **Interactive Price Charts:**
    *   **Short Term:** 1-Day and 5-Day Intraday (15min intervals).
    *   **Long Term:** History up to 20+ years (Daily intervals).
*   **Visual Valuation:** Green/Yellow/Red zones showing if a stock is Undervalued or Overvalued.
*   **News Aggregator:** Live news feed with smart publisher parsing and formatting.
*   **Deep Linking:** Share specific analysis via URL (e.g., `?ticker=NVDA`).

## 🛠️ Technologies Used

*   **Python 3.8+**
*   **Streamlit:** For the web interface and UI components.
*   **yfinance:** For fetching stock market data, financials, and news.
*   **Plotly:** For interactive charts (Radar, Bar, and Line charts).
*   **Pandas & NumPy:** For financial calculations and data manipulation.

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/market-radar.git
    cd market-radar
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

4.  **Open your browser:** The app will typically launch at `http://localhost:8501`.

## ☁️ How to Deploy to Streamlit Cloud

You can host this app for free on Streamlit Community Cloud.

1.  **Push to GitHub:**
    *   Upload `app.py` and `requirements.txt` to a new GitHub repository.
2.  **Sign up for Streamlit:**
    *   Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3.  **Deploy:**
    *   Click **"New App"**.
    *   Select your GitHub repository from the list.
    *   Select the branch (usually `main`) and the file path (`app.py`).
    *   Click **"Deploy"**.
4.  **Done!**
    *   Wait 1-2 minutes for the build to finish. You will get a URL (e.g., `https://market-radar.streamlit.app`) to share with the world.

## ⚠️ Disclaimer

**Not Financial Advice.** This tool is for educational and informational purposes only. The calculations (DCF, Graham Number) depend on data provided by third-party APIs which may be delayed or inaccurate. Always do your own due diligence before investing.

---
*Created with ❤️ using Python.*
