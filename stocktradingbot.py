from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
from alpaca_trade_api import REST
import pandas as pd
import random
import logging
import yfinance as yf
import numpy as np

logging.basicConfig(level=logging.INFO)

API_KEY = "Add API Key"
API_SECRET = "Add Unique Secret Key"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}


sp500_symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "V", "JPM", "JNJ",
    "NVDA", "PG", "UNH", "MA", "VZ", "HD", "PYPL", "DIS", "BAC", "CRM", 
    "CMCSA", "INTC", "NFLX", "ORCL", "CSCO", "ADBE", "XOM", "T", "KO", "PFE",
    "NKE", "ACN", "ABT", "WMT", "TMO", "CVX", "MCD", "MRK", "IBM", "QCOM",
    "LMT", "C", "COST", "AVGO", "TXN", "SBUX", "PM", "AMGN", "AMD", "AAP", 
    "WFC", "LIN", "MO", "DHR", "MDLZ", "RTX", "HON", "BA", "MMM", "GE", 
    "NOW", "NEE", "UNP", "CHTR", "LOW", "UPS", "INTU", "GILD", "SPGI", 
    "GS", "MS", "CAT", "LLY", "BKNG", "FDX", "CI", "ISRG", "AXP", "COP", 
    "BIIB", "CVS", "TJX", "USB", "LRCX", "PEP", "TGT", "MET", "SYK", "ADP", 
    "DUK", "ADI", "BLK", "MMC", "ICE", "PNC", "TFC", "FIS", "SO", "DD", 
    "NEE", "EMR", "LUV", "DOW", "GM", "HUM", "AON", "KMB", "PRU", 
    "MPC", "NEM", "DELL", "AIG", "EW", "ALL", "EQIX", "GD", "CME", 
    "ROST", "HPQ", "TROW", "REGN", "VRTX", "SBAC", "MDT", "ZTS", "HCA", 
    "APD", "COO", "ORLY", "ECL", "GPN", "XEL", "CHD", "BK", "MSCI", "MAR", 
    "CDNS", "PGR", "EA", "ANSS", "EXC", "WY", "CMG", "STE", "IDXX", "MKC", 
    "CINF", "ROP", "IQV", "ZBH", "PH", "SPG", "DXCM", "NOC", "MCO",
    "SNPS", "ALK", "TDOC", "MNST", "ALGN", "CTSH", "YUM", "HLT", "LHX", 
    "PAYX", "MKTX", "ALLE", "CNC", "APTV", "ZBRA", "HSY", "TRV", 
    "CTAS", "ETSY", "CDW", "CTVA", "ANET", "WAT", "KEYS", 
    "TYL", "LNT", "FAST", "VTRS", "AWK", "O", "GRMN", "VRSK", "WST", "EXR", 
     "RMD", "RSG", "DPZ", "VRSN", "CPRT", "JKHY", "ODFL", 
    "TTWO", "WAB", "ENPH", "TSCO", "WRB", "RCL", "TT", "AZO", "PEG", "IPG", 
    "LBRDK", "CHRW", "IEX", "KLAC", "JBHT", "NCLH", "NVR", "STX", "LEN", 
    "LBRDA", "URI", "LVS", "GWW", "ADSK", "PAYC", "CHTR", "MCHP", "BDX", 
    "PKG", "DRI", "IPGP", "DLTR", "WHR", "WRK", "PNR", "LYV", 
    "MTB", "DLR", "MTD", "ALB", "KMX", "CBOE", "PTC", "ZION", "VNT", "WDC", 
    "WH", "AAL", "RHI", "UNM", "FLS", "TPR", "HII", "UDR", "MPWR", 
    "FMC", "NTAP", "EFX", "EXR", "LYB", "TXT", "TRIP", "HOG", "HST", "COTY", 
    "PVH", "XRAY", "BEN", "LUMN", "KSS", "MHK", "GPS", "BWA", 
    "IPG", "QRVO", "KIM", "APA", "HOG", "LEG", "L", "NOV", "HWM", 
    "HBI", "FOX", "UA", "UAA", "J", "HES", "COF", "NCLH", "R", "KIM", 
    "VNT", "H", "LEG", "COF"
]


def adjust_symbol_for_yahoo(symbol):
    return symbol.replace(".", "-")

def validate_symbols(symbols):
    valid_symbols = []
    for symbol in symbols:
        yahoo_symbol = adjust_symbol_for_yahoo(symbol)
        try:
            data = yf.Ticker(yahoo_symbol).history(period="1d")
            if not data.empty:
                valid_symbols.append(symbol)
        except Exception as e:
            logging.warning(f"Symbol {symbol} is invalid or has no data: {e}")
    return valid_symbols

valid_symbols = validate_symbols(sp500_symbols)

if not valid_symbols:
    logging.error("No valid symbols found. Exiting.")
    exit()

class MLTrader(Strategy):
    def initialize(self, cash_at_risk: float = .5):
        self.sleeptime = "10s"
        self.last_trade = {}
        self.cash_at_risk = cash_at_risk
        self.api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL)
        self.bought_price = {}
        self.sold_price = {}
        self.iteration_count = 0
        self.data = {symbol: self.get_price_data(symbol) for symbol in valid_symbols}
        self.support_levels = {symbol: None for symbol in valid_symbols}
        self.price_hits = {symbol: [] for symbol in valid_symbols}

    def get_price_data(self, symbol):
        yahoo_symbol = adjust_symbol_for_yahoo(symbol)
        data = yf.Ticker(yahoo_symbol).history(period="1y")
        return data['Close']

    def calculate_trama(self, prices, length=10):
        """
        Calculate the TRAMA indicator.
        """
        w = np.arange(1, length + 1)
        numerator = sum(w * prices[-length:])
        denominator = sum(w)
        return numerator / denominator

    def calculate_sma(self, prices, window):
        return prices.rolling(window=window).mean()

    def get_trama(self, symbol):
        prices = self.data[symbol]
        trama = self.calculate_trama(prices)
        return trama

    def get_sma(self, symbol, window):
        prices = self.data[symbol]
        sma = self.calculate_sma(prices, window)
        return sma

    def position_sizing(self, symbol):
        cash = self.get_cash()
        last_price = self.get_last_price(symbol)
        
        if last_price is None:
            logging.error(f"No price data available for {symbol}")
            return cash, None, 0
        
        max_cash = cash * 0.2
        quantity = round(max_cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def buy_stock(self, symbol):
        cash, last_price, quantity = self.position_sizing(symbol)
        if last_price is None:
            logging.info(f"Skipping buy for {symbol} due to missing price data.")
            return

        trama = self.get_trama(symbol)
        sma_5 = self.get_sma(symbol, 5)
        sma_20 = self.get_sma(symbol, 20)
        sma_200 = self.get_sma(symbol, 200)
        sma_50 = self.get_sma(symbol, 50)
        support_level = self.support_levels[symbol]

        logging.info(f"Checking buy conditions for {symbol}: last_price={last_price}, TRAMA={trama}, SMA_5={sma_5.iloc[-1]}, SMA_20={sma_20.iloc[-1]}, SMA_200={sma_200.iloc[-1]}, SMA_50={sma_50.iloc[-1]}, Support Level={support_level}")

        # Ensure that the most recent moving averages are calculated correctly
        if pd.isna(sma_5.iloc[-1]) or pd.isna(sma_20.iloc[-1]) or pd.isna(sma_200.iloc[-1]) or pd.isna(sma_50.iloc[-1]):
            logging.info(f"Skipping buy for {symbol} due to incomplete SMA data.")
            return

        # Check the updated buy conditions
        if (sma_5.iloc[-1] > sma_20.iloc[-1] and
                sma_50.iloc[-1] > sma_200.iloc[-1] * 1.1 and
                last_price - sma_5.iloc[-1] <= 1 and
                support_level is not None and
                abs(last_price - support_level) <= 0.1):
            logging.info(f"Buy conditions met for {symbol}: Quantity={quantity}, Last Price={last_price}")
            order = self.create_order(
                symbol, 
                quantity, 
                "buy", 
                type="market"
            )
            self.submit_order(order)
            self.bought_price[symbol] = last_price
            self.last_trade[symbol] = "buy"
            logging.info(f"Bought {quantity} shares of {symbol} at {last_price}")

            # Create a limit order to sell at 0.2 above the bought price
            sell_price = last_price + 0.2
            sell_order = self.create_order(
                symbol,
                quantity,
                "sell",
                type="limit",
                limit_price=sell_price,
                time_in_force="gtc"
            )
            self.submit_order(sell_order)
            logging.info(f"Created a limit order to sell {quantity} shares of {symbol} at {sell_price}")
        else:
            logging.info(f"Buy conditions not met for {symbol}")

    def sell_stock(self, symbol):
        position = self.get_position(symbol)
        if position and symbol in self.bought_price:
            current_price = self.get_last_price(symbol)
            logging.info(f"Checking sell conditions for {symbol}: current_price={current_price}, bought_price={self.bought_price[symbol]}")

            if current_price >= 0.2 + self.bought_price[symbol]:
                order = self.create_order(
                    symbol,
                    position.quantity,
                    "sell",
                    type="market"
                )
                self.submit_order(order)
                del self.bought_price[symbol]
                self.last_trade[symbol] = "sell"
                logging.info(f"Sold {position.quantity} shares of {symbol} at {current_price}")

    def short_stock(self, symbol):
        cash, last_price, quantity = self.position_sizing(symbol)
        if last_price is None:
            logging.info(f"Skipping short for {symbol} due to missing price data.")
            return

        trama = self.get_trama(symbol)
        sma_5 = self.get_sma(symbol, 5)
        sma_20 = self.get_sma(symbol, 20)
        sma_200 = self.get_sma(symbol, 200)
        sma_50 = self.get_sma(symbol, 50)
        support_level = self.support_levels[symbol]

        logging.info(f"Checking short conditions for {symbol}: last_price={last_price}, TRAMA={trama}, SMA_5={sma_5.iloc[-1]}, SMA_20={sma_20.iloc[-1]}, SMA_200={sma_200.iloc[-1]}, SMA_50={sma_50.iloc[-1]}, Support Level={support_level}")

        # Ensure that the most recent moving averages are calculated correctly
        if pd.isna(sma_5.iloc[-1]) or pd.isna(sma_20.iloc[-1]) or pd.isna(sma_200.iloc[-1]) or pd.isna(sma_50.iloc[-1]):
            logging.info(f"Skipping short for {symbol} due to incomplete SMA data.")
            return

        # Check the updated short conditions
        if (sma_5.iloc[-1] < sma_20.iloc[-1] and
                sma_50.iloc[-1] < sma_200.iloc[-1] * 0.9 and
                sma_5.iloc[-1] - last_price <= 1 and
                support_level is not None and
                abs(last_price - support_level) > 0.1):
            logging.info(f"Short conditions met for {symbol}: Quantity={quantity}, Last Price={last_price}")
            order = self.create_order(
                symbol, 
                quantity, 
                "sell", 
                type="market"
            )
            self.submit_order(order)
            self.sold_price[symbol] = last_price
            self.last_trade[symbol] = "short"
            logging.info(f"Sold short {quantity} shares of {symbol} at {last_price}")

            # Create a limit order to cover the short position at 0.2 below the sold price
            cover_price = last_price - 0.2
            cover_order = self.create_order(
                symbol,
                quantity,
                "buy",
                type="limit",
                limit_price=cover_price,
                time_in_force="gtc"
            )
            self.submit_order(cover_order)
            logging.info(f"Created a limit order to cover short for {quantity} shares of {symbol} at {cover_price}")
        else:
            logging.info(f"Short conditions not met for {symbol}")

    def cover_stock(self, symbol):
        position = self.get_position(symbol)
        if position and symbol in self.sold_price:
            current_price = self.get_last_price(symbol)
            logging.info(f"Checking cover conditions for {symbol}: current_price={current_price}, sold_price={self.sold_price[symbol]}")

            if current_price <= self.sold_price[symbol] - 0.2:
                order = self.create_order(
                    symbol,
                    position.quantity,
                    "buy",
                    type="market"
                )
                self.submit_order(order)
                del self.sold_price[symbol]
                self.last_trade[symbol] = "cover"
                logging.info(f"Covered short for {position.quantity} shares of {symbol} at {current_price}")

    def check_support_level(self, symbol, current_price):
        self.price_hits[symbol].append(current_price)
        hits = [price for price in self.price_hits[symbol] if abs(price - current_price) <= 0.1]

        if len(hits) >= 4:
            self.support_levels[symbol] = current_price
            logging.info(f"Support level set for {symbol} at {current_price}")

    def on_trading_iteration(self):
        for symbol in valid_symbols:
            current_price = self.get_last_price(symbol)
            if current_price is None:
                continue

            if self.support_levels[symbol] is None:
                self.check_support_level(symbol, current_price)

            logging.info(f"Selected symbol: {symbol}")
            if self.last_trade.get(symbol) != "buy":
                self.buy_stock(symbol)
            elif self.last_trade.get(symbol) != "short":
                self.short_stock(symbol)
            elif self.last_trade.get(symbol) == "buy":
                self.sell_stock(symbol)
            elif self.last_trade.get(symbol) == "short":
                self.cover_stock(symbol)
        
        self.iteration_count += 1

    def on_finish(self):
        print("Backtesting finished.")

start_date = datetime(2024, 5, 10)
end_date = datetime(2024, 5, 13)
broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(name='mlstrat', broker=broker, parameters={"cash_at_risk": .5})

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"cash_at_risk": .5}
)

trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
