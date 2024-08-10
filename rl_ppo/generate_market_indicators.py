import pandas as pd
import numpy as np
import talib as ta

RAW_DATA_PATH = "data/raw_data.csv"

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def add_momentum_indicators(self):
        self.data['RSI'] = ta.RSI(self.data['Close'], period=14)
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = ta.MACD(self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['Stoch_k'], self.data['Stoch_d'] = ta.STOCH(self.data,
                                                              fastk_period=14, slowk_period=3, slowd_period=3)

    def add_volume_indicators(self):
        self.data['OBV'] = ta.OBV(self.data['Close'], self.data['Volume'])

    def add_volatility_indicators(self):
        self.data['Upper_BB'], self.data['Middle_BB'], self.data['Lower_BB'] = ta.BBANDS(self.data['Close'], period=20)
        self.data['ATR_1'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], period=1)
        self.data['ATR_2'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], period=2)
        self.data['ATR_5'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], period=5)
        self.data['ATR_10'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], period=10)
        self.data['ATR_20'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], period=20)

    def add_trend_indicators(self):
        self.data['ADX'], self.data['+DI'], self.data['-DI'] = ta.ADX(self.data['High'], self.data['Low'], self.data['Close'], period=14)
        self.data['CCI'] = ta.CCI(self.data['High'], self.data['Low'], self.data['Close'], period=5)

    def add_other_indicators(self):
        self.data['DLR'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['TWAP'] = self.data['Close'].expanding().mean()
        self.data['VWAP'] = (self.data['Volume'] * (self.data['High'] + self.data['Low']) / 2).cumsum() / self.data['Volume'].cumsum()

    def add_all_indicators(self):
        self.add_momentum_indicators()
        self.add_volume_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()
        self.add_other_indicators()
        return self.data
    
    
def main():
    
    data = pd.read_csv(RAW_DATA_PATH)
    
    # Preprocessing to create necessary columns
    data['price']=data['price']/1e9
    data['bid_px_00']=data['bid_px_00']/1e9
    data['ask_px_00']=data['ask_px_00']/1e9

    data['Close'] = data['price']
    data['Volume'] = data['size']
    data['High'] = data[['bid_px_00', 'ask_px_00']].max(axis=1)
    data['Low'] = data[['bid_px_00', 'ask_px_00']].min(axis=1)
    data['Open'] = data['Close'].shift(1).fillna(data['Close'])


    ti = TechnicalIndicators(data)
    df_with_indicators = ti.add_all_indicators()
    market_features_df = df_with_indicators[35:]

    market_features_df.to_csv("data/market_indicators.csv", index=False)



if __name__ == "__main__":
    main()
