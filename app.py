import streamlit as st
from market_model import marketPredict
from stock_model import get_stock_forecast 

st.set_page_config(page_title="Finance Assistant Demo", layout="wide")

st.title("📈 Finance Assistant Demo")

# Predicting Market Health, whether or not to invest based on S&P 500 data
st.header("📊 S&P 500 Market Prediction")
if st.button("Check Market Health"):
    market_result = marketPredict()
    st.success(f"Market Prediction for Tomorrow: {market_result}")

# Stock prediction section
st.header("💹 Stock Price Prediction")

# User inputs ticker symbol
stock_code = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):")
days_ahead = st.slider("Days Ahead to Predict", 1, 14, 5)

if st.button("Predict Stock Price") and stock_code:
    with st.spinner(f"Predicting {stock_code} for {days_ahead} days..."):
        result = get_stock_forecast(stock_code, future_days=days_ahead)

        if "error" in result:
            st.error(result["error"])
        else:
            # Display results
            st.success(f"Predictions for {stock_code} starting {result['last_date']}:")
            
            # Show future forecast in table
            forecast_df = (
                st.dataframe(
                    result["future_forecast"],
                    use_container_width=True
                )
            )

            # Plot chart
            import pandas as pd
            import matplotlib.pyplot as plt

            forecast_data = pd.DataFrame(result["future_forecast"])
            forecast_data["date"] = pd.to_datetime(forecast_data["date"])

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast_data["date"], forecast_data["predicted_price"], marker="o", label="Forecast")
            ax.set_title(f"{stock_code} Stock Price Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Price")
            ax.legend()
            st.pyplot(fig)
