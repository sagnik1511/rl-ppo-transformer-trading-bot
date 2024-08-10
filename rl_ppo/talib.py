"""I was unable to use TA-Lib package, so I asked chatGPT to implement them for me.
There was some inconsistency with division operation where I used numerical stability of 1e-7
These helped me to generate the market_indicators"""

import numpy as np

def RSI(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]  # the diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def EMA(prices, period):
    ema = np.zeros_like(prices)
    multiplier = 2 / (period + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = EMA(prices, fastperiod)
    ema_slow = EMA(prices, slowperiod)
    macd = ema_fast - ema_slow
    signal = EMA(macd, signalperiod)
    hist = macd - signal
    return macd, signal, hist

def STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3):
    # Calculate Fast %K
    lowest_low = df['Low'].rolling(window=fastk_period).min()
    highest_high = df['High'].rolling(window=fastk_period).max()
    fastk = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # Calculate Slow %K (SMA of Fast %K)
    slowk = fastk.rolling(window=slowk_period).mean()
    
    # Calculate Slow %D (SMA of Slow %K)
    slowd = slowk.rolling(window=slowd_period).mean()
    
    return slowk, slowd


def OBV(close, volume):
    obv = np.zeros_like(close)
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    return obv


def ATR(high, low, close, period=14):
    tr = np.zeros_like(close)
    atr = np.zeros_like(close)

    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr

def TR(high, low, close):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    true_range[0] = 0  # The first TR value is undefined, set to 0
    return true_range

def DM(high, low):
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = down_move[0] = 0  # The first DM values are undefined, set to 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    return plus_dm, minus_dm

def SMMA(values, period):
    smma = np.zeros_like(values)
    smma[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        smma[i] = (smma[i-1] * (period - 1) + values[i]) / period
    return smma

def ADX(high, low, close, period=14):
    tr = TR(high, low, close)
    plus_dm, minus_dm = DM(high, low)
    
    smoothed_tr = SMMA(tr, period)
    smoothed_plus_dm = SMMA(plus_dm, period)
    smoothed_minus_dm = SMMA(minus_dm, period)
    
    plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + 1e-7))
    minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + 1e-7))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-7)
    
    adx = SMMA(dx, period)
    return adx, plus_di, minus_di

def BBANDS(close, period=20, num_std_dev=2):
    sma = np.convolve(close, np.ones((period,)) / period, mode='valid')
    sma = np.concatenate((np.zeros(period-1) + np.nan, sma))  # Align with original array length
    std_dev = np.zeros_like(close)
    
    for i in range(period - 1, len(close)):
        std_dev[i] = np.std(close[i - period + 1:i + 1])
        
    upper_band = sma + num_std_dev * std_dev
    lower_band = sma - num_std_dev * std_dev
    
    return upper_band, sma, lower_band


def CCI(high, low, close, period=20):
    typical_price = (high + low + close) / 3
    sma = np.convolve(typical_price, np.ones((period,)) / period, mode='valid')
    mean_deviation = np.zeros_like(typical_price)
    
    for i in range(period - 1, len(typical_price)):
        mean_deviation[i] = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma[i - period + 1]))

    cci = np.zeros_like(typical_price)
    for i in range(period - 1, len(typical_price)):
        cci[i] = (typical_price[i] - sma[i - period + 1]) / (0.015 * mean_deviation[i] + 1e-7)
    
    return cci
