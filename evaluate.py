"""
Hybrid2 model - render olmadan headless değerlendirme.

Kullanım:
    python hybrid_2/evaluate.py --model models_hybrid_2/best/best_model
    python hybrid_2/evaluate.py --model models_hybrid_2/best/best_model --n_episodes 20
"""

import argparse
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env import DroneSwarmEnvHybrid2


def evaluate(model_path, vec_norm_path=None, n_episodes=20, n_obstacles=5, seed=42):
    if not model_path or not os.path.exists(model_path + ".zip"):
        print("Model bulunamadı:", model_path)
        return

    def make_env():
        return Monitor(DroneSwarmEnvHybrid2(
            grid_size=50.0, n_obstacles=n_obstacles, seed=seed,
            wall_sliding=True, offset_scale=0.6, formation_coef=0.3,
        ))

    env = DummyVecEnv([make_env])
    if vec_norm_path and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)
    rewards, lengths, goals = [], [], []

    for ep in range(n_episodes):
        res = env.reset()
        obs = res[0] if isinstance(res, (tuple, list)) else res
        ep_rew, ep_len = 0.0, 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rew += float(reward[0])
            ep_len += 1
            if done[0]:
                break
        dist = info[0].get("dist_to_goal", 999)
        rewards.append(ep_rew)
        lengths.append(ep_len)
        goals.append(dist < 3.0)
        print(f"  Episode {ep+1:2d}: reward={ep_rew:8.1f}  steps={ep_len:3d}  goal={dist < 3.0}")

    print("\n" + "=" * 50)
    print("  Hybrid2 Değerlendirme Sonuçları")
    print("=" * 50)
    print(f"  Mean reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean steps:     {np.mean(lengths):.1f}")
    print(f"  Goal rate:      {np.mean(goals):.1%} ({int(np.sum(goals))}/{n_episodes})")
    print("=" * 50)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models_hybrid_2/best/best_model")
    p.add_argument("--vec_norm", default="models_hybrid_2/vec_normalize.pkl")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--n_obstacles", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    evaluate(
        model_path=args.model,
        vec_norm_path=args.vec_norm,
        n_episodes=args.n_episodes,
        n_obstacles=args.n_obstacles,
        seed=args.seed,
    )
