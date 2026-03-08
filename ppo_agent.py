"""
PPO Agent for DroneSwarmEnvHybrid2 (10-dim action, rastgele start/target).
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from env import DroneSwarmEnvHybrid2


def make_env(
    n_obstacles=5,
    n_obstacles_range=None,
    wall_sliding=True,
    offset_scale=0.6,
    formation_coef=0.3,
    proximity_threshold=2.0,
    proximity_penalty_coef=0.1,
    min_drone_separation=1.5,
    min_drone_separation_penalty=15.0,
    seed=0,
):
    def _init():
        e = DroneSwarmEnvHybrid2(
            grid_size=50.0,
            n_obstacles=n_obstacles,
            n_obstacles_range=n_obstacles_range,
            safety_radius=2.0,
            max_speed=2.0,
            offset_scale=offset_scale,
            max_steps=500,
            wall_sliding=wall_sliding,
            formation_coef=formation_coef,
            proximity_threshold=proximity_threshold,
            proximity_penalty_coef=proximity_penalty_coef,
            min_drone_separation=min_drone_separation,
            min_drone_separation_penalty=min_drone_separation_penalty,
            seed=seed,
        )
        return Monitor(e)
    return _init


def build_ppo_agent(env, tensorboard_log="./logs_hybrid_2/tensorboard/", policy_kwargs=None):
    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[256, 256])
    return PPO(
        policy="MlpPolicy", env=env,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5, policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log, verbose=1,
    )


def train(
    total_timesteps=1_000_000,
    n_envs=4,
    n_obstacles=5,
    n_obstacles_range=None,
    wall_sliding=True,
    offset_scale=0.6,
    formation_coef=0.3,
    proximity_threshold=2.0,
    proximity_penalty_coef=0.1,
    min_drone_separation=1.5,
    min_drone_separation_penalty=15.0,
    save_dir="./models_hybrid_2/",
    log_dir="./logs_hybrid_2/",
    eval_freq=25_000,
    save_freq=50_000,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = DummyVecEnv([
        make_env(
            n_obstacles, n_obstacles_range, wall_sliding, offset_scale, formation_coef,
            proximity_threshold, proximity_penalty_coef,
            min_drone_separation, min_drone_separation_penalty,
            i,
        )
        for i in range(n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([
        make_env(
            n_obstacles, n_obstacles_range, wall_sliding, offset_scale, formation_coef,
            proximity_threshold, proximity_penalty_coef,
            min_drone_separation, min_drone_separation_penalty,
            42,
        )
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = build_ppo_agent(train_env, os.path.join(log_dir, "tensorboard/"))

    callbacks = CallbackList([
        EvalCallback(eval_env, best_model_save_path=os.path.join(save_dir, "best/"),
                     log_path=os.path.join(log_dir, "eval/"),
                     eval_freq=max(1, eval_freq // n_envs), n_eval_episodes=10,
                     deterministic=True, render=False),
        CheckpointCallback(save_freq=max(1, save_freq // n_envs),
                          save_path=os.path.join(save_dir, "checkpoints/"),
                          name_prefix="ppo_hybrid_2"),
    ])

    print("Training PPO (Hybrid2 - rastgele start/target)...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(save_dir, "ppo_hybrid_2_final"))
    train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    return model, train_env
