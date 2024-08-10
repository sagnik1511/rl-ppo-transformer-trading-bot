import gym
from gym import spaces
import numpy as np
import pandas as pd
from utils import *


class TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, daily_trading_limit):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.daily_trading_limit = daily_trading_limit
        self.current_step = 0

        # Extract state columns
        self.state_columns = [
            "Close",
            "Volume",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "Stoch_k",
            "Stoch_d",
            "OBV",
            "Upper_BB",
            "Middle_BB",
            "Lower_BB",
            "ATR_1",
            "ADX",
            "+DI",
            "-DI",
            "CCI",
        ]

        # Initialize balance, shares held, and total shares traded
        self.balance = 10_000_000.0  # $10 million
        self.shares_held = 0
        self.total_shares_traded = 0

        # Define action space: [Hold, Buy, Sell]
        self.action_space = spaces.Discrete(3)

        # Define observation space based on state columns
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.state_columns),), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = 10_000_000.0  # $10 million
        self.shares_held = 0
        self.total_shares_traded = 0
        self.cumulative_reward = 0
        self.trades = []
        return self._next_observation()

    def _next_observation(self):
        return self.data[self.state_columns].iloc[self.current_step].values

    def step(self, action):
        expected_price = self.data.iloc[self.current_step]["ask_px_00"]
        actual_price = self.data.iloc[self.current_step]["price"]
        transaction_time = self.data.iloc[self.current_step]["ts_in_delta"]
        self._take_action(action)
        reward = 0

        if self.current_step >= len(self.data) - 1:
            self.current_step = 0
        if action != 0:
            transaction_cost = self._calculate_transaction_cost(
                self.data.iloc[self.current_step]["Volume"],
                0.3,
                self.data["Volume"].mean(),
            )
            reward = self._calculate_reward(
                expected_price, actual_price, transaction_time, transaction_cost
            )
            self.cumulative_reward += reward
            if self.trades:
                self.trades[-1]["reward"] = reward
                self.trades[-1]["transaction_cost"] = transaction_cost
                self.trades[-1]["slippage"] = expected_price - actual_price
                self.trades[-1]["time_penalty"] = 100 * transaction_time / 1e9
        done = self.current_step == len(self.data) - 1
        obs = self._next_observation()
        info = {
            "step": self.current_step,
            "action": action,
            "price": actual_price,
            "shares": self.trades[-1]["shares"] if self.trades else 0,
        }
        self.current_step += 1

        return obs, reward, done, info

    def _take_action(self, action):
        current_price = self.data.iloc[self.current_step]["Close"]
        current_time = pd.to_datetime(self.data.iloc[self.current_step]["ts_event"])
        trade_info = {
            "step": self.current_step,
            "timestamp": current_time,
            "action": action,
            "price": current_price,
            "shares": 0,
            "reward": 0,
            "transaction_cost": 0,
            "slippage": 0,
            "time_penalty": 0,
        }

        if (
            action == 1
        ):  # and self.total_shares_traded < self.daily_trading_limit:  # Buy
            shares_bought = (
                self.balance * np.random.uniform(0.001, 0.005)
            ) // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
            self.total_shares_traded += shares_bought
            trade_info["shares"] = shares_bought
            if shares_bought > 0:
                self.trades.append(trade_info)
        elif (
            action == 2
        ):  # and self.total_shares_traded < self.daily_trading_limit:  # Sell
            shares_sold = min(
                (self.balance * np.random.uniform(0.001, 0.005)) // current_price,
                self.shares_held,
            )
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_traded -= shares_sold
            trade_info["shares"] = shares_sold
            if shares_sold > 0:
                self.trades.append(trade_info)

    def _calculate_reward(
        self, expected_price, actual_price, transaction_time, transaction_cost
    ):
        slippage = expected_price - actual_price
        time_penalty = 100 * transaction_time / 1e9
        reward = -(slippage + time_penalty + transaction_cost)
        return reward

    def _calculate_transaction_cost(self, volume, volatility, daily_volume):
        return volatility * np.sqrt(volume / daily_volume)

    def run(self):
        self.reset()
        for _ in range(len(self.data)):
            self.step()
        return self.cumulative_reward, self.trades

    def render(self, mode="human", close=False):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares held: {self.shares_held}")
        print(f"Total shares traded: {self.total_shares_traded}")
        print(
            f'Total portfolio value: {self.balance + self.shares_held * self.data.iloc[self.current_step]["Close"]}'
        )
        print(f"Cumulative reward: {self.cumulative_reward}")
        # self.print_trades()

    def print_trades(self):
        # download all trades in a pandas dataframe using .csv
        trades_df = pd.DataFrame(self.trades)
        # Save a csv
        trades_df.to_csv("trades_ppo.csv", index=False)
        for trade in self.trades:
            print(
                f"Step: {trade['step']}, Timestamp: {trade['timestamp']}, Action: {trade['action']}, Price: {trade['price']}, Shares: {trade['shares']}, Reward: {trade['reward']}, Transaction Cost: {trade['transaction_cost']}, Slippage: {trade['slippage']}, Time Penalty: {trade['time_penalty']}"
            )


class TradingEnvironmentwithBlotter:
    def __init__(self, data, daily_trading_limit, window_size):
        self.data = preprocess_data(data)
        self.daily_trading_limit = daily_trading_limit
        self.window_size = window_size
        self.state_columns = [
            "price",
            "liquidity",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "Stoch_k",
            "Stoch_d",
            "OBV",
            "Upper_BB",
            "Middle_BB",
            "Lower_BB",
            "ATR_1",
            "ADX",
            "+DI",
            "-DI",
            "CCI",
        ]
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = 10_000_000.0  # 10 Million
        self.shares_held = 0
        self.total_shares_traded = 0
        self.cumulative_reward = 0
        self.trades = []
        self.portfolio = {
            "cash": self.balance,
            "holdings": {ticker: 0 for ticker in self.data["symbol"].unique()},
        }
        self.data["RSI"] = calculate_rsi(self.data["price"])
        self.data["pct_change"] = self.data["price"].pct_change()
        (
            self.data["rolling_mean_vol"],
            self.data["rolling_std_vol"],
            self.data["rolling_mean_liq"],
            self.data["rolling_std_liq"],
        ) = calculate_vol_and_liquidity(
            self.data["price"], self.data["liquidity"], self.window_size
        )

    def step(self):
        row = self.data.iloc[self.current_step]
        current_price = row["price"]
        current_time = pd.to_datetime(row["ts_event"])
        current_rsi = row["RSI"]
        current_vol = row["pct_change"]
        current_liq = row["liquidity"]
        mean_vol = row["rolling_mean_vol"]
        std_vol = row["rolling_std_vol"]
        mean_liq = row["rolling_mean_liq"]
        std_liq = row["rolling_std_liq"]

        if current_rsi < 30:  # Entry signal based on RSI
            trade_direction = "BUY"
            trade_price = get_trade_price(
                current_price,
                current_vol,
                current_liq,
                mean_vol,
                std_vol,
                mean_liq,
                std_liq,
                trade_direction,
            )
            trade_size = (
                self.portfolio["cash"] * np.random.uniform(0.001, 0.005)
            ) / trade_price
            if self.portfolio["cash"] >= trade_size * trade_price:
                self.portfolio["cash"] -= trade_size * trade_price
                self.portfolio["holdings"][row["symbol"]] += trade_size
                trade_status = "filled"
            else:
                trade_status = "cancelled"
        elif current_rsi > 70:  # Exit signal based on RSI
            trade_direction = "SELL"
            if self.portfolio["holdings"][row["symbol"]] > 0:
                trade_size = min(
                    self.portfolio["holdings"][row["symbol"]],
                    self.portfolio["cash"]
                    * np.random.uniform(0.001, 0.005)
                    / current_price,
                )
                trade_price = get_trade_price(
                    current_price,
                    current_vol,
                    current_liq,
                    mean_vol,
                    std_vol,
                    mean_liq,
                    std_liq,
                    trade_direction,
                )
                self.portfolio["cash"] += trade_size * trade_price
                self.portfolio["holdings"][row["symbol"]] -= trade_size
                trade_status = "filled"
            else:
                trade_size = 0
                trade_status = "cancelled"
        else:
            trade_direction = "HOLD"
            trade_size = 0
            trade_price = current_price
            trade_status = "skipped"

        if trade_size > 0:
            expected_price = row["ask_px_00"]
            actual_price = row["price"]
            transaction_time = row["ts_in_delta"]
            transaction_cost = self._calculate_transaction_cost(
                row["Volume"], 0.3, self.data["Volume"].mean()
            )
            slippage = expected_price - actual_price
            time_penalty = 1000 * transaction_time / 1e9
            reward = -(slippage + time_penalty + transaction_cost)

            self.cumulative_reward += reward
            self.trades.append(
                {
                    "step": self.current_step,
                    "timestamp": current_time,
                    "action": trade_direction,
                    "price": trade_price,
                    "shares": trade_size,
                    "symbol": row["symbol"],
                    "reward": reward,
                    "transaction_cost": transaction_cost,
                    "slippage": slippage,
                    "time_penalty": time_penalty,
                }
            )

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            self.current_step = 0

    def _calculate_transaction_cost(self, volume, volatility, daily_volume):
        return volatility * np.sqrt(volume / daily_volume)

    def run(self):
        self.reset()
        for _ in range(len(self.data)):
            self.step()
        return self.cumulative_reward, self.trades

    def render(self):
        print(f"Cumulative reward: {self.cumulative_reward}")
        row = self.data.iloc[self.current_step]
        print(
            f'Total portfolio value: {self.portfolio["cash"] + self.portfolio["holdings"][row["symbol"]]*row["Close"]}'
        )
        # get trades in a pandas dataframe
        trades_df = pd.DataFrame(self.trades)
        # Save a csv
        trades_df.to_csv("trades_blotter.csv", index=False)
        for trade in self.trades:
            print(
                f"Step: {trade['step']}, Timestamp: {trade['timestamp']}, Action: {trade['action']}, Price: {trade['price']}, Shares: {trade['shares']}, Symbol: {trade['symbol']}, Reward: {trade['reward']}, Transaction Cost: {trade['transaction_cost']}, Slippage: {trade['slippage']}, Time Penalty: {trade['time_penalty']}"
            )
