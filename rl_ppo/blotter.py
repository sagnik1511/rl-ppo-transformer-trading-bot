import argparse
import pandas as pd
from envs import TradingEnvironmentwithBlotter

MKT_INDICATORS_FILE = "data/market_indicators.csv"


def blot(
    ticker,
    daily_trading_limit,
    window_size
):

    # Loading the indicators file
    df = pd.read_csv(MKT_INDICATORS_FILE)

    # Loading required ticker
    ticker_data = df[df.symbol == ticker]

    # defining env and policy model
    env = TradingEnvironmentwithBlotter(ticker_data, daily_trading_limit, window_size)
    
    # Run the environment
    cumulative_reward, trades = env.run()

    # Render the results
    env.render()

    # Saving env runs
    pd.DataFrame(env.trades).to_csv(f"runs/trades_with_blotter.csv",index=False)


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--ticker", required=True, type=str, help="Name of the ticker"
    )
    parser.add_argument(
        "-dtl",
        "--daily-trading-limit",
        default=1000,
        type=int,
        help="daily trading limit",
    )
    parser.add_argument('-ws', '--window_size', type=int, default=60, help='Window Size')

    return parser


def main():
    parser = define_parser()
    args = parser.parse_args()
    args = vars(args)
    print(args)
    blot(**args)


if __name__ == "__main__":
    main()
