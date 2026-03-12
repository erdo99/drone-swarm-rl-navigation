"""
ppo_agent_shared.py — Shared Policy PPO Agent

MİMARİ:
  - Stable-Baselines3 PPO + VecNormalize
  - Tek model, 4 drone'u kapsıyor
  - Obs: (48,) = 4 drone × 12 lokal obs  ← v2: OBS_DIM 10→12 (+2 ray)
  - Act: (8,)  = 4 drone × 2 hız

4 ayrı model DEĞİL — 1 model, ağırlıklar paylaşılıyor (parameter sharing).
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor


class SyncVecNormalizeCallback(BaseCallback):
    """Eval ortamı, train ortamının obs normalizasyonunu kullansın (aynı referans)."""
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env

    def _init_callback(self) -> None:
        train_env = self.model.get_env()
        if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = train_env.obs_rms  # aynı obje → otomatik güncel

    def _on_step(self) -> bool:
        return True


def make_env(env_kwargs: dict):
    def _init():
        # Shared V3 ortamı: route-corridor + min_start_target_dist + proximity
        from env_shared_v3 import DroneSwarmSharedEnv
        return Monitor(DroneSwarmSharedEnv(**env_kwargs))
    return _init


def build_agent(env, save_dir: str = "models_shared", tensorboard_log: str = None, ent_coef: float = 0.2):
    os.makedirs(save_dir, exist_ok=True)
    if tensorboard_log is None:
        tensorboard_log = os.path.join("logs_shared", "tensorboard")
    # SB3 obs boyutunu ortamdan otomatik alır — OBS_DIM değişikliği burada
    # manuel güncelleme gerektirmez. net_arch bağımsız.
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=tensorboard_log,
        verbose=1,
    )


def train(
    total_timesteps: int = 2_000_000,
    save_dir: str = "models_shared",
    log_dir: str = "logs_shared",
    env_kwargs: dict = None,
    n_envs: int = 4,
    eval_freq: int = 25_000,
    save_freq: int = 50_000,
    ent_coef: float = 0.02,
):
    if env_kwargs is None:
        env_kwargs = {}

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = DummyVecEnv([make_env(env_kwargs) for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = build_agent(train_env, save_dir, os.path.join(log_dir, "tensorboard"), ent_coef=ent_coef)

    callbacks = CallbackList([
        SyncVecNormalizeCallback(eval_env),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_dir, "best"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(save_dir, "checkpoints"),
            name_prefix="ppo_shared",
        ),
    ])

    # OBS_DIM=12 (v2), 4*12=48 obs toplam
    print("Training PPO (Parameter Sharing + Local Obs, 48 obs → 8 act)...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(save_dir, "ppo_shared_final")
    model.save(final_path)
    train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))

    print(f"\nBitti! Model: {save_dir}")
    print(f"  Final: {final_path}.zip")
    print(f"  Best:  {save_dir}/best/best_model.zip")
    return model, train_env