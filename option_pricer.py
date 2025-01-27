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

# Streamlit GUI
st.set_page_config(page_title="Option Pricing Calculator", layout="wide")

# Sidebar 
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Current Stock Price ($)", 50.0, 500.0, 100.0)
K = st.sidebar.number_input("Strike Price ($)", 50.0, 500.0, 100.0)
T = st.sidebar.number_input("Time to Expiration (Years)", 0.1, 3.0, 1.0)
r = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
sigma = st.sidebar.number_input("Volatility (%)", 1.0, 100.0, 20.0) / 100
dividend_yield = st.sidebar.number_input("Dividend Yield (%)", 0.0, 10.0, 0.0) / 100


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


# Price sensitivity visualization
st.header("Price Sensitivity Analysis")
stocks = np.linspace(0.5*S, 1.5*S, 100)
call_prices = [black_scholes(s, K, T, r, sigma, dividend_yield, 'call') for s in stocks]
put_prices = [black_scholes(s, K, T, r, sigma, dividend_yield, 'put') for s in stocks]

chart_data = {
    "Stock Price": stocks,
    "Call Option Price": [p[0] for p in call_prices],
    "Put Option Price": [p[0] for p in put_prices],
    # "Delta": [p[1] for p in call_prices],
    # "Gamma": [p[2] for p in call_prices],
}
st.line_chart(
    chart_data,
    x="Stock Price",
    y=["Call Option Price", "Put Option Price"],
    color=["#FF0000", "#0000FF"]
)

# Display Delta and Gamma values using heatmap
st.header("Delta and Gamma Sensitivity Analysis")
st.write("Delta and Gamma values for Call Options")

import pandas as pd
# Create DataFrame for Heatmap
heatmap_data = pd.DataFrame({
    "Stock Price": stocks,
    "Delta": [p[1] for p in call_prices],
    "Gamma": [p[2] for p in call_prices],
}).set_index("Stock Price")

import matplotlib.pyplot as plt
import seaborn as sns
fig , ax = plt.subplots()
sns.heatmap(heatmap_data, fmt=".2f", ax=ax)
st.write(fig)

st.write("Note: This calculator uses the Black-Scholes model for European-style Options.")