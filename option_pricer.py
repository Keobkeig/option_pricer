import streamlit as st
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, dividend_yield=0.0, option_type='call'):
    """
    Calculate Black-Scholes option price for European options
    
    S: current stock price
    K: strike price
    T: time to maturity (in years)
    r: risk-free rate
    sigma: volatility
    dividend_yield: continuous dividend yield
    """
    
    d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * np.exp(-dividend_yield * T) * norm.cdf(d1) 
                 - K * np.exp(-r * T) * norm.cdf(d2))
        delta = norm.cdf(d1) 
        gamma = delta / (S * sigma * np.sqrt(T)) 
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2) 
                 - S * np.exp(-dividend_yield * T) * norm.cdf(-d1))
        delta = -norm.cdf(-d1)
        gamma = delta / (S * sigma * np.sqrt(T))
    return (price, delta, gamma)

# Streamlit
st.set_page_config(page_title="Option Pricing Calculator", layout="wide")

# Sidebar 
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Current Stock Price ($)", 50.0, 500.0, 100.0)
K = st.sidebar.number_input("Strike Price ($)", 50.0, 500.0, 100.0)
T = st.sidebar.number_input("Time to Expiration (Years)", 0.1, 3.0, 1.0)
r = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
sigma = st.sidebar.number_input("Volatility (%)", 1.0, 100.0, 20.0) / 100
dividend_yield = st.sidebar.number_input("Dividend Yield (%)", 0.0, 10.0, 0.0) / 100
st.sidebar.markdown("---")  
st.sidebar.header("PnL Heatmap Settings")
price_range_pct = st.sidebar.slider(
    "Price Range (%)",
    min_value=10,
    max_value=100,
    value=30,
    help="Percentage above and below current price"
)
min_time = st.sidebar.slider(
    "Minimum Time (Years)",
    min_value=0.1,
    max_value=float(T),
    value=0.1,
    step=0.1
)

# Calculate prices
call_price = black_scholes(S, K, T, r, sigma, dividend_yield, 'call')
put_price = black_scholes(S, K, T, r, sigma, dividend_yield, 'put')

# Main display
st.title("Black-Scholes Option Pricing Calculator")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option Price")
    st.success(f"${call_price[0]:.2f}")

with col2:
    st.subheader("Put Option Price")
    st.success(f"${put_price[0]:.2f}")


st.header("Price Sensitivity Analysis")
stocks = np.linspace(0.5*S, 1.5*S, 100)
call_prices = [black_scholes(s, K, T, r, sigma, dividend_yield, 'call') for s in stocks]
put_prices = [black_scholes(s, K, T, r, sigma, dividend_yield, 'put') for s in stocks]

chart_data = {
    "Stock Price": stocks,
    "Call Option Price": [p[0] for p in call_prices],
    "Put Option Price": [p[0] for p in put_prices],
}

st.line_chart(
    chart_data,
    x="Stock Price",
    y=["Call Option Price", "Put Option Price"],
    color=["#FF0000", "#0000FF"]
)

st.header("Delta and Gamma Sensitivity Analysis")
st.write("Delta and Gamma values for Call Options")

greeks_chart_data = {
    "Stock Price": stocks,
    "Delta": [p[1] for p in call_prices],
    "Gamma": [p[2] for p in call_prices],
}

st.line_chart(
    greeks_chart_data,
    x="Stock Price",
    y=["Delta", "Gamma"],
    color=["#FF0000", "#0000FF"]
)

st.header("Option PnL Analysis")


# Update price and time ranges based on slider values
price_range = np.linspace((1-price_range_pct/100)*S, (1+price_range_pct/100)*S, 50)
time_range = np.linspace(min_time, T, 50)

price_mesh, time_mesh = np.meshgrid(price_range, time_range)
pnl_matrix_call = np.zeros_like(price_mesh)
pnl_matrix_put = np.zeros_like(price_mesh)

# Calculate PnL for each price-time 
initial_call_price = call_price[0]
initial_put_price = put_price[0]

for i in range(len(time_range)):
    for j in range(len(price_range)):
        future_call_price = black_scholes(price_range[j], K, time_range[i], r, sigma, dividend_yield, 'call')[0]
        pnl_matrix_call[i, j] = future_call_price - initial_call_price
        
        future_put_price = black_scholes(price_range[j], K, time_range[i], r, sigma, dividend_yield, 'put')[0]
        pnl_matrix_put[i, j] = future_put_price - initial_put_price

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=('Call Option PnL', 'Put Option PnL'))

fig.add_trace(
    go.Heatmap(
        x=price_range,
        y=time_range,
        z=pnl_matrix_call,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
        colorbar=dict(title='PnL ($)'),
        showscale=False
    ),
    row=1, col=1
)

fig.add_trace(
    go.Heatmap(
        x=price_range,
        y=time_range,
        z=pnl_matrix_put,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
    ),
    row=1, col=2
)

fig.update_layout(
    title_text="Option PnL Heatmap",
    xaxis_title="Stock Price ($)",
    yaxis_title="Time to Expiration (Years)",
    xaxis2_title="Stock Price ($)",
    yaxis2_title="Time to Expiration (Years)",
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

st.write("Note: This calculator uses the Black-Scholes model for European-style Options.")