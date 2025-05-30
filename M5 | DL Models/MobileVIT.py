import torch
import torch.nn as nn
import requests
import time

# Define a transformer model for real-time financial prediction
class RealTimeFinPredictor(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(RealTimeFinPredictor, self).__init__()
        self.input_layer = nn.Linear(1, d_model)  # Transform 1D time series input into d_model dimension
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.output_layer = nn.Linear(d_model, 1)  # Transform output back to 1D prediction

    def forward(self, src, tgt):
        src = self.input_layer(src)
        tgt = self.input_layer(tgt)
        transformer_output = self.transformer(src, tgt)
        return self.output_layer(transformer_output)

# Fetch real-time stock data (e.g., from Alpha Vantage)
def fetch_stock_data(api_key, symbol, interval="1min", length=10):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}"
    response = requests.get(url).json()
    
    # Extract closing prices and prepare for inference
    data = [float(v['4. close']) for k, v in response['Time Series (1min)'].items()]
    if len(data) < length:
        raise ValueError("Insufficient data for the required time window")
    
    return torch.tensor(data[:length]).unsqueeze(-1).float()  # Shape (length, 1)

# Real-time inference loop for stock prediction
def real_time_inference(api_key, symbol, model, input_seq_len=10, output_seq_len=5, interval=60):
    model.eval()  # Set model to evaluation mode
    
    while True:
        try:
            # Fetch the latest stock prices
            src = fetch_stock_data(api_key, symbol, length=input_seq_len).unsqueeze(1)  # Add batch dimension
            tgt = torch.zeros((output_seq_len, 1, 1))  # For predicting next 5 steps

            # Perform autoregressive inference
            with torch.no_grad():
                for i in range(output_seq_len):
                    prediction = model(src, tgt[:i+1])
                    tgt[i] = prediction[-1]  # Autoregressive prediction

            print(f"Predicted next {output_seq_len} stock prices: {tgt.squeeze().tolist()}")
        except Exception as e:
            print(f"Error during inference: {e}")

        time.sleep(interval)  # Wait before fetching new data

if __name__ == "__main__":
    # API details and stock symbol
    API_KEY = "your_alpha_vantage_api_key"
    SYMBOL = "AAPL"  # Stock symbol, e.g., Apple Inc.
    
    # Initialize the model
    model = RealTimeFinPredictor(d_model=64, nhead=4, num_layers=2, dropout=0.1)
    
    # Load your pretrained model weights if applicable
    # model.load_state_dict(torch.load("path_to_your_model_weights.pth"))

    # Start real-time inference
    real_time_inference(api_key=API_KEY, symbol=SYMBOL, model=model, input_seq_len=10, output_seq_len=5)