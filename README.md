# RL-PPO Bot Analysis with Transformer Feature Extractor


# Environment Specs : 

## OS : 

Distributor ID: Ubuntu

Description:    Ubuntu 20.04.6 LTS

Release:        20.04

Codename:       focal

## Python 

Version : 3.10.13

## Packages

The packages and versions are stored inside `requirements.txt` file.

## Execution : 

You can run scripts to generate the required files

 - Generate market_indicators data : 
    ```bash
    python3 rl_ppo/generate_market_indicators.py
    ```

 - Run the Policy model on MlpPolicy Baseline : 
    ```bash
    python3 rl_ppo/train.py --ticker AAPL -ts 100 <add rest arguments to change configuration>
    ```

- Run the Transformer Baseline Extractor on PPO Model : 
    ```bash
    python3 rl_ppo/train.py --ticker AAPL -ts 100 -ut <add rest arguments to change configuration>
    ```