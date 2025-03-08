import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime as dt
from typing import Dict, List, Callable

class StockTradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Trading Signal Application")
        self.root.geometry("1200x800")
        
        # Data storage
        self.watchlist = []  # List of tickers
        self.stock_data = {}  # Dictionary to store fetched data
        self.signal_algorithms = {
            "Moving Average Crossover": self.compute_ma_crossover,
            "RSI": self.compute_rsi,
            "MACD": self.compute_macd,
            "Bollinger Bands": self.compute_bollinger
        }
        self.current_algorithm = "Moving Average Crossover"
        
        # Create main layout
        self.create_layout()
        
    def create_layout(self):
        # Create frames
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame components (watchlist and controls)
        ttk.Label(left_frame, text="Your Watchlist", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Watchlist display
        self.watchlist_frame = ttk.Frame(left_frame)
        self.watchlist_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for watchlist
        scrollbar = ttk.Scrollbar(self.watchlist_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Watchlist listbox
        self.watchlist_box = tk.Listbox(self.watchlist_frame, height=15, width=30, yscrollcommand=scrollbar.set)
        self.watchlist_box.pack(fill=tk.BOTH, expand=True)
        self.watchlist_box.bind('<<ListboxSelect>>', self.on_watchlist_select)
        scrollbar.config(command=self.watchlist_box.yview)
        
        # Button controls
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Add Stock/Fund", command=self.add_ticker).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Remove Selected", command=self.remove_ticker).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh Data", command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        
        # Algorithm selection
        algo_frame = ttk.Frame(left_frame)
        algo_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(algo_frame, text="Signal Algorithm:").pack(side=tk.LEFT, padx=5)
        self.algo_var = tk.StringVar(value=self.current_algorithm)
        algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.algo_var, 
                                     values=list(self.signal_algorithms.keys()),
                                     state="readonly", width=20)
        algo_dropdown.pack(side=tk.LEFT, padx=5)
        algo_dropdown.bind("<<ComboboxSelected>>", self.change_algorithm)
        
        # Right frame components (charts and data)
        self.chart_frame = ttk.Frame(right_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.signal_frame = ttk.LabelFrame(right_frame, text="Trading Signal", padding=10)
        self.signal_frame.pack(fill=tk.X, pady=10)
        
        self.signal_label = ttk.Label(self.signal_frame, text="No stock selected", font=("Arial", 16))
        self.signal_label.pack(pady=10)
        
        self.signal_strength = ttk.Progressbar(self.signal_frame, orient="horizontal", length=300, mode="determinate")
        self.signal_strength.pack(pady=10)
        
        self.signal_text = ttk.Label(self.signal_frame, text="")
        self.signal_text.pack(pady=5)
        
    def add_ticker(self):
        ticker = simpledialog.askstring("Add Stock/Fund", "Enter stock symbol (e.g., AAPL):")
        if ticker:
            ticker = ticker.upper().strip()
            # Validate ticker exists
            try:
                stock = yf.Ticker(ticker)
                # Try to get info to validate the ticker
                info = stock.info
                if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                    messagebox.showerror("Invalid Ticker", f"Could not find data for {ticker}")
                    return
                
                if ticker not in self.watchlist:
                    self.watchlist.append(ticker)
                    self.watchlist_box.insert(tk.END, ticker)
                    # Fetch initial data
                    self.fetch_data(ticker)
                    messagebox.showinfo("Success", f"{ticker} added to watchlist")
                else:
                    messagebox.showinfo("Info", f"{ticker} is already in your watchlist")
            except Exception as e:
                messagebox.showerror("Error", f"Error adding {ticker}: {str(e)}")
    
    def remove_ticker(self):
        selection = self.watchlist_box.curselection()
        if selection:
            index = selection[0]
            ticker = self.watchlist_box.get(index)
            self.watchlist.remove(ticker)
            self.watchlist_box.delete(index)
            if ticker in self.stock_data:
                del self.stock_data[ticker]
            
            # Clear chart if the removed ticker was displayed
            self.clear_chart()
    
    def fetch_data(self, ticker):
        try:
            # Get data from Yahoo Finance (you could replace this with broker API data)
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=365)  # 1 year of data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                self.stock_data[ticker] = data
                return True
            else:
                messagebox.showerror("Data Error", f"No data available for {ticker}")
                return False
        except Exception as e:
            messagebox.showerror("Fetch Error", f"Error fetching data for {ticker}: {str(e)}")
            return False
    
    def refresh_data(self):
        for ticker in self.watchlist:
            self.fetch_data(ticker)
        
        # Refresh the selected ticker display if there is one
        selection = self.watchlist_box.curselection()
        if selection:
            self.on_watchlist_select(None)
    
    def on_watchlist_select(self, event):
        selection = self.watchlist_box.curselection()
        if selection:
            index = selection[0]
            ticker = self.watchlist_box.get(index)
            
            # Make sure we have data for this ticker
            if ticker not in self.stock_data:
                success = self.fetch_data(ticker)
                if not success:
                    return
            
            # Display chart and calculate signal
            self.display_chart(ticker)
            self.calculate_signal(ticker)
    
    def change_algorithm(self, event):
        self.current_algorithm = self.algo_var.get()
        
        # Recalculate signal for currently selected ticker
        selection = self.watchlist_box.curselection()
        if selection:
            self.on_watchlist_select(None)  # Refresh the current selection
    
    def display_chart(self, ticker):
        # Clear previous chart
        self.clear_chart()
        
        data = self.stock_data[ticker]
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data
        ax1.plot(data.index, data['Close'], label='Close Price')
        ax1.set_title(f"{ticker} - Price Chart")
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot volume - FIXED: Convert to list to avoid array conversion issues
        x_values = list(range(len(data.index)))
        ax2.bar(x_values, data['Volume'].values)
        # Set x-axis ticks to match dates
        ax2.set_xticks(x_values[::len(x_values)//5])  # Show only 5 dates for readability
        ax2.set_xticklabels([data.index[i].strftime('%Y-%m-%d') for i in ax2.get_xticks()])
        ax2.set_title('Volume')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
    
    # Add indicators based on selected algorithm
    # [Rest of the function remains the same]
        
    def clear_chart(self):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
    
    def calculate_signal(self, ticker):
        # Get the selected algorithm
        algorithm_func = self.signal_algorithms[self.current_algorithm]
        
        # Calculate signal
        signal, strength = algorithm_func(ticker)
        
        # Update signal display
        self.signal_label.config(text=f"{ticker}: {signal}")
        
        # Convert strength to percentage (0-100) for progress bar
        strength_pct = abs(int(strength * 100))
        self.signal_strength['value'] = strength_pct
        
        # Set color based on signal direction (buy=green, sell=red, neutral=yellow)
        if signal == "Strong Buy":
            color = "#00CC00"  # Green
            text = f"Strong Buy Signal - Confidence: {strength_pct}%"
        elif signal == "Buy":
            color = "#88CC88"  # Light green
            text = f"Buy Signal - Confidence: {strength_pct}%"
        elif signal == "Neutral":
            color = "#CCCC00"  # Yellow
            text = "Neutral Signal - No clear direction"
        elif signal == "Sell":
            color = "#CC8888"  # Light red
            text = f"Sell Signal - Confidence: {strength_pct}%"
        else:  # Strong Sell
            color = "#CC0000"  # Red
            text = f"Strong Sell Signal - Confidence: {strength_pct}%"
        
        self.signal_frame.configure(style='')
        style = ttk.Style()
        style.configure('SignalFrame.TLabelframe', background=color)
        self.signal_frame.configure(style='SignalFrame.TLabelframe')
        
        self.signal_text.config(text=text)
    
    # Signal calculation algorithms
    def compute_ma_crossover(self, ticker):
        data = self.stock_data[ticker]
        
        # Calculate short and long term moving averages
        short_window = 20
        long_window = 50
        
        data['SMA20'] = data['Close'].rolling(window=short_window).mean()
        data['SMA50'] = data['Close'].rolling(window=long_window).mean()
        
        # Get latest values
        latest_short_ma = data['SMA20'].iloc[-1]
        latest_long_ma = data['SMA50'].iloc[-1]
        
        # Calculate previous values
        prev_short_ma = data['SMA20'].iloc[-2]
        prev_long_ma = data['SMA50'].iloc[-2]
        
        # Calculate current difference and strength
        current_diff = latest_short_ma - latest_long_ma
        prev_diff = prev_short_ma - prev_long_ma
        
        # Normalize strength between 0 and 1
        max_diff = data['Close'].max() * 0.1  # 10% of max price as reference
        strength = min(abs(current_diff) / max_diff, 1.0)
        
        # Determine signal based on moving average crossover
        if current_diff > 0 and prev_diff <= 0:
            return "Strong Buy", strength
        elif current_diff > 0 and current_diff > prev_diff:
            return "Buy", strength
        elif current_diff < 0 and prev_diff >= 0:
            return "Strong Sell", -strength
        elif current_diff < 0 and current_diff < prev_diff:
            return "Sell", -strength
        else:
            return "Neutral", 0
    
    def compute_rsi(self, ticker):
        data = self.stock_data[ticker]
        
        # Calculate RSI
        rsi = self.compute_rsi_data(data['Close'])
        latest_rsi = rsi.iloc[-1]
        
        # Determine signal strength
        if latest_rsi <= 30:
            # Oversold - Buy signal
            strength = (30 - latest_rsi) / 30
            if latest_rsi < 20:
                return "Strong Buy", strength
            else:
                return "Buy", strength
        elif latest_rsi >= 70:
            # Overbought - Sell signal
            strength = (latest_rsi - 70) / 30
            if latest_rsi > 80:
                return "Strong Sell", strength
            else:
                return "Sell", strength
        else:
            # Neutral zone
            if latest_rsi < 45:
                # Leaning towards buy
                return "Neutral", (45 - latest_rsi) / 15
            elif latest_rsi > 55:
                # Leaning towards sell
                return "Neutral", (latest_rsi - 55) / 15
            else:
                return "Neutral", 0
    
    def compute_rsi_data(self, price_series, window=14):
        # Calculate price changes
        delta = price_series.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_macd(self, ticker):
        data = self.stock_data[ticker]
        
        # Calculate MACD
        macd, signal, histogram = self.compute_macd_data(data['Close'])
        
        # Get latest values
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]
        latest_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # Determine signal strength
        if latest_macd > latest_signal and latest_hist > 0 and latest_hist > prev_hist:
            # Strong bullish momentum
            strength = min(abs(latest_hist) / 2, 1.0)
            return "Strong Buy", strength
        elif latest_macd > latest_signal:
            # Bullish
            strength = min(abs(latest_macd - latest_signal) / 2, 1.0)
            return "Buy", strength
        elif latest_macd < latest_signal and latest_hist < 0 and latest_hist < prev_hist:
            # Strong bearish momentum
            strength = min(abs(latest_hist) / 2, 1.0)
            return "Strong Sell", strength
        elif latest_macd < latest_signal:
            # Bearish
            strength = min(abs(latest_macd - latest_signal) / 2, 1.0)
            return "Sell", strength
        else:
            return "Neutral", 0
    
    def compute_macd_data(self, price_series, fast=12, slow=26, signal=9):
        # Calculate EMAs
        ema_fast = price_series.ewm(span=fast, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogrampy
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def compute_bollinger(self, ticker):
        data = self.stock_data[ticker]
        
        # Calculate Bollinger Bands (20-period SMA with 2 standard deviations)
        window = 20
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        # Get latest values
        latest_price = data['Close'].iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]
        latest_sma = sma.iloc[-1]
        
        # Calculate percentage distance from bands
        band_width = latest_upper - latest_lower
        
        if band_width == 0:  # Avoid division by zero
            return "Neutral", 0
            
        # Calculate distance from upper and lower bands in percentage
        upper_distance = (latest_upper - latest_price) / band_width
        lower_distance = (latest_price - latest_lower) / band_width
        
        # Determine signal
        if latest_price < latest_lower:
            # Price below lower band - oversold
            strength = min(abs(lower_distance) * 2, 1.0)
            return "Strong Buy", strength
        elif latest_price < latest_sma and lower_distance < 0.3:
            # Price close to lower band - potential buy
            strength = min(0.3 - lower_distance, 1.0)
            return "Buy", strength
        elif latest_price > latest_upper:
            # Price above upper band - overbought
            strength = min(abs(upper_distance) * 2, 1.0)
            return "Strong Sell", strength
        elif latest_price > latest_sma and upper_distance < 0.3:
            # Price close to upper band - potential sell
            strength = min(0.3 - upper_distance, 1.0)
            return "Sell", strength
        else:
            # Price in the middle - neutral
            return "Neutral", 0


def main():
    root = tk.Tk()
    app = StockTradingApp(root)
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')  # You can use 'clam', 'alt', 'default', 'classic'
    
    root.mainloop()


if __name__ == "__main__":
    main()