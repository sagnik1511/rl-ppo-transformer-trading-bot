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


I have used Colab instance to train the Policies and store results ->

<a target="_blank" href="https://colab.research.google.com/drive/1Cfn71uLOaxI0TYl84AgL-sAta7LoYy09?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

P.S. I have used `colab_requirements.txt` to install dependencies at colab as it doesn't alow us to update necessary ipynb dependant components.

## Model Coparison

On field| Blotter Policy | MLP Policy | Transformer Policy |
|:-:|:-:|:-:|:-:|
|reward|-12231.228875703917|-8748.478045315253|`-8560.98202774446`|
|slippage|`287.38999999997304`|354.15999999991124|346.6849999999179|
|time_penalty|5624.946076999934|690.5739430000103|`674.7037646000075`|

Rest information are shared in the codes and the colab notebook.