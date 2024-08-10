import argparse
import pandas as pd
from envs import TradingEnvironment
from models import TransformerPolicy
from stable_baselines3 import PPO

MKT_INDICATORS_FILE = "data/market_indicators.csv"


def train(
    ticker,
    use_transformer,
    daily_trading_limit,
    learning_rate,
    n_steps,
    batch_size,
    gamma,
    clip_range,
    epochs,
    total_timestamps,
):

    # Loading the indicators file
    df = pd.read_csv(MKT_INDICATORS_FILE)

    # Loading required ticker
    ticker_data = df[df.symbol == ticker]

    # defining env and policy model
    env = TradingEnvironment(ticker_data, daily_trading_limit)
    model_name = "transformer_policy" if use_transformer else "mlp_policy"
    policy = TransformerPolicy if use_transformer else "MlpPolicy"
    ppo_model = PPO(
        policy,
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        n_epochs=epochs,
        batch_size=batch_size,
        gamma=gamma,
        clip_range=clip_range,
    )

    # training model
    ppo_model.learn(total_timesteps=total_timestamps)

    # saving model
    ppo_model.save(
        f"checkpoints/ppo_model_trans_{model_name}_{learning_rate}_{gamma}_{clip_range}.zip"
        if use_transformer
        else f"checkpoints/ppo_model_{model_name}_{learning_rate}_{gamma}_{clip_range}.zip"
    )

    # Evaluate the model
    obs = env.reset()
    for _ in range(len(ticker_data)):
        action, _states = ppo_model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break

    # Render the final state
    env.render()

    # Saving env runs
    pd.DataFrame(env.trades).to_csv(
        (
            f"runs/trades_ppo_model_trans_{model_name}_{learning_rate}_{gamma}_{clip_range}.csv"
            if use_transformer
            else f"runs/trades_ppo_model_{model_name}_{learning_rate}_{gamma}_{clip_range}.csv"
        ),
        index=False,
    )


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--ticker", required=True, type=str, help="Name of the ticker"
    )
    parser.add_argument(
        "-ut",
        "--use-transformer",
        action="store_true",
        default=False,
        help="transformer usage flag",
    )
    parser.add_argument(
        "-dtl",
        "--daily-trading-limit",
        default=1000,
        type=int,
        help="daily trading limit",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, type=float, help="learning rate"
    )
    parser.add_argument(
        "-ns", "--n-steps", default=512, type=int, help="number of steps"
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("-g", "--gamma", default=0.95, type=float, help="gamma")
    parser.add_argument(
        "-cr", "--clip-range", default=0.2, type=float, help="clip range"
    )
    parser.add_argument("-e", "--epochs", default=6, type=int, help="number of epochs")
    parser.add_argument(
        "-ts", "--total-timestamps", required=True, type=int, help="Totat timestamps"
    )

    return parser


def main():
    parser = define_parser()
    args = parser.parse_args()
    args = vars(args)
    print(args)
    train(**args)


if __name__ == "__main__":
    main()
