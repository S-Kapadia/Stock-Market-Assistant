import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Defining model
class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Defining forecast
def get_stock_forecast(ticker: str, start_date="2020-01-01", seq_length=30, 
                       num_epochs=50, hidden_dim=32, future_days=5):
    #training done based on past stock history et. 200 onwards

    
    df = yf.download(ticker, start=start_date)
    if df.empty:
        return {"error": f"No data found for ticker {ticker}"}

    scaler = StandardScaler()
    df['Close_scaled'] = scaler.fit_transform(df[['Close']])

    #Sequence building
    data = []
    for i in range(len(df) - seq_length):
        data.append(df.Close_scaled.values[i:i+seq_length])
    data = np.array(data)

    X = data[:, :-1]
    y = data[:, -1]

    X = torch.from_numpy(X).unsqueeze(-1).float().to(device)  # (samples, seq_len-1, 1)
    y = torch.from_numpy(y).unsqueeze(-1).float().to(device)  # (samples, 1)

    # Train/Test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Model
    model = PredictionModel(input_dim=1, hidden_dim=hidden_dim, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Train
    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # future predictions
    model.eval()
    last_sequence = df['Close_scaled'].values[-seq_length:]
    current_seq = torch.tensor(last_sequence, dtype=torch.float32).view(1, -1, 1).to(device)

    future_preds = []
    with torch.no_grad():
        for _ in range(future_days):
            next_pred = model(current_seq)
            future_preds.append(next_pred.item())
            current_seq = torch.cat((current_seq[:, 1:, :], next_pred.view(1, 1, 1)), dim=1)

    # trnsform to real prices
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Generate future dates
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)

    #return dict
    return {
        "ticker": ticker,
        "last_date": str(df.index[-1].date()),
        "future_forecast": [
            {"date": str(date.date()), "predicted_price": float(price[0])}
            for date, price in zip(future_dates, future_preds)
        ]
    }


