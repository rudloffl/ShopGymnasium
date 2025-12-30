
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from typing import Dict, List, Tuple
from collections import namedtuple
import gymnasium as gym

import sys
from pathlib import Path

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn


# Ajouter le dossier principal au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add parent directory to path to import ShopEnv
sys.path.append(str(Path(__file__).parent))
from B00_Agents.tinyshop3 import ShopEnv


def train_ppo(
    total_timesteps: int = 50_000,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 10,
    save_path: str = "./ppo_shopenv",
    log_path: str = "./logs"
):
    """
    Train a PPO agent on ShopEnv
    
    Args:
        total_timesteps: Total training steps
        eval_freq: Frequency of evaluation
        n_eval_episodes: Number of episodes for evaluation
        save_path: Path to save the model
        log_path: Path for tensorboard logs
    """
    
    # Create environment
    print("Creating environment...")
    env = ShopEnv(duration_max=7)
    
    # Check if environment follows Gym API
    print("Checking environment...")
    check_env(env, warn=True)
    
    # Wrap environment with Monitor for logging
    env = Monitor(env, log_path)
    
    # Create evaluation environment
    eval_env = Monitor(ShopEnv(), log_path)
    
    # Define custom policy architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Actor network
            vf=[256, 256]   # Critic network
        ),
        activation_fn=nn.ReLU
    )
    
    # Create PPO model
    # Use MultiInputPolicy for Dict observation spaces
    print("Creating PPO model...")
    model = PPO(
        "MultiInputPolicy",  # Changed from MlpPolicy for Dict observation space
        env,
        # policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64, #Was 64
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        verbose=0,
        tensorboard_log=log_path
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path,
        name_prefix="ppo_shopenv"
    )
    
    # Train the agent
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{save_path}/ppo_shopenv_final"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    return model, env


def evaluate_ppo(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False
):
    """
    Evaluate a trained PPO agent
    
    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment
    env = ShopEnv(duration_max=7)
    
    # Evaluation metricsTrue
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from model (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
        if render:
            logs = env.render()
            fig, ax = plt.subplots(6, 1, figsize=(20, 16))

            # Machine Gantt
            maqgantt = pd.DataFrame(logs['maq_utilization'], columns=['start', 'end', 'product', 'machine'])
            machines = sorted(maqgantt['machine'].unique())
            products = maqgantt['product'].unique()
            colors = list(mcolors.TABLEAU_COLORS.values())
            if len(products) > len(colors):
                colors = plt.cm.tab20(np.linspace(0, 1, len(products)))
            product_colors = dict(zip(products, colors))
            for idx, row in maqgantt.iterrows():
                machine_idx = machines.index(row['machine'])
                duration = row['end'] - row['start']
                
                ax[0].barh(machine_idx, duration, left=row['start'], 
                        height=0.6, 
                        color=product_colors[row['product']],
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.8)
            legend_elements = [Rectangle((0, 0), 1, 1, fc=product_colors[prod], 
                             edgecolor='black', label=prod) 
                  for prod in products]
            ax[0].legend(handles=legend_elements,)#  title='Product', loc='upper right', bbox_to_anchor=(1.12, 1)
            ax[0].set_yticks(range(len(machines)))
            ax[0].set_yticklabels(machines)
            ax[0].set_title('Machine Utilization Gantt Chart')

            # Prod records
            prod_records = pd.DataFrame(logs['prod_trace'], columns=['ts', 'end', 'product', 'machine']).set_index('ts')
            for prod in prod_records['product'].unique():
                df = prod_records[prod_records['product']==prod]
                #df = df.rename({'product': prod}, axis='columns')
                df[prod] = int(prod)
                df[prod].plot(ax=ax[5])
            ax[5].set_title('production Records')
            ax[5].legend()

            # Sales
            sales = pd.DataFrame(logs['sales_log'], columns=['ts', 'product' , 'qty']).set_index('ts')
            for prod in df['product'].unique():
                df = sales[sales['product'] == prod]
                df = df.rename({'qty': prod}, axis='columns')
                df[prod].plot(kind='bar', ax=ax[1])
            ax[1].set_title('Sales')
            ax[1].legend()

            # Orders
            ax[3].set_title('Raw Product Orders')
            try:
                
                sales = pd.DataFrame(logs['order_log'], columns=['ts', 'raw_orders']).set_index('ts')
                sales['raw_orders'].plot(kind='bar', ax=ax[3])
                
            except:
                pass

            # Stocks
            stocks = pd.DataFrame(logs['stock_utilization'], columns=['ts', 'stockarea', 'qty']).set_index('ts')
            for stockname in stocks['stockarea'].unique():
                df = stocks[stocks['stockarea'] == stockname]
                df.rename({'qty': stockname}, axis='columns')[stockname].plot(ax=ax[2])
            ax[2].legend()
            ax[2].set_title('Stock Levels')

            # Products Stolen
            try:
                stolen = pd.DataFrame(logs['stealing_log'], columns=['ts', 'product', 'lost']).set_index('ts')
                for prod in stolen['product'].unique():
                    df = stolen[stolen['product']==prod]
                    df = df.rename({'lost': prod}, axis='columns')
                    df[prod].plot(kind='bar', ax=ax[4])
            except:
                pass
            ax[4].legend()
            ax[4].set_title('Product Stolen')

            plt.tight_layout()

            plt.savefig(f'episode_{episode}.png', transparent=True)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Calculate statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards)
    }
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("="*50)
    
    env.close()
    
    return results, episode_rewards, episode_lengths


def plot_evaluation_results(episode_rewards: list, episode_lengths: list, save_path: str = None):
    """Plot evaluation results"""
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
    
    # Plot rewards
    ax1.plot(episode_rewards, marker='o')
    ax1.axhline(y=np.mean(episode_rewards), color='r', linestyle='--', label='Mean')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Configuration
    TRAIN = True  # Set to False to only evaluate
    TOTAL_TIMESTEPS = 100_000
    MODEL_PATH = "./ppo_shopenv/best_model"
    
    if TRAIN:
        # Train the model
        model, env = train_ppo(
            total_timesteps=TOTAL_TIMESTEPS,
            eval_freq=10_000, #Was 5  
            n_eval_episodes=10,
            save_path="./ppo_shopenv",
            log_path="./logs"
        )
        
        # Evaluate the trained model
        results, rewards, lengths = evaluate_ppo(
            model_path=MODEL_PATH,
            n_episodes=5,
            render=True
        )
    else:
        # Only evaluate existing model
        results, rewards, lengths = evaluate_ppo(
            model_path=MODEL_PATH,
            n_episodes=2,
            render=True
        )
    
    # Plot results
    plot_evaluation_results(rewards, lengths, save_path="./evaluation_results.png")
    
    print("\nTraining complete! To view tensorboard logs, run:")
    print("tensorboard --logdir=./logs")