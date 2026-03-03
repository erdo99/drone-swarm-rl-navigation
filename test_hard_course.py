"""
Hybrid2 - Zorlu sabit parkur testi.

Her episode aynı layout: sol alt başlangıç, sağ üst hedef, ortada artı şeklinde engel bloğu.
Eğitilmiş model bu zorlu parkurda ne kadar başarılı?
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env_hard_course import DroneSwarmEnvHybrid2HardCourse
from hard_course_config import HARD_OBSTACLES, HARD_START, HARD_TARGET


def run(model_path, vec_norm_path=None, n_episodes=20):
    if not model_path or not os.path.exists(model_path + ".zip"):
        print("Model bulunamadı:", model_path)
        return

    def make_env():
        return Monitor(DroneSwarmEnvHybrid2HardCourse(
            grid_size=50.0, n_obstacles=len(HARD_OBSTACLES),
            wall_sliding=True, offset_scale=0.6, formation_coef=0.3, seed=42,
            swap_start_target=True,
        ))

    env = DummyVecEnv([make_env])
    if vec_norm_path and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)
    rewards, lengths, goals, collisions_list = [], [], [], []

    print("=" * 55)
    print("  ZORLU PARKUR TESTİ (Sabit Layout)")
    print(f"  Başlangıç: ({HARD_TARGET[0]:.0f}, {HARD_TARGET[1]:.0f})  |  Hedef: ({HARD_START[0]:.0f}, {HARD_START[1]:.0f}) [swap]")
    print(f"  {len(HARD_OBSTACLES)} engel (hard_course_config.py)")
    print("=" * 55)

    for ep in range(n_episodes):
        res = env.reset()
        obs = res[0] if isinstance(res, (tuple, list)) else res
        ep_rew, ep_len, ep_coll = 0.0, 0, 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rew += float(reward[0])
            ep_len += 1
            ep_coll += info[0].get("collisions", 0)
            if done[0]:
                break
        dist = info[0].get("dist_to_goal", 999)
        goal_ok = dist < 3.0
        rewards.append(ep_rew)
        lengths.append(ep_len)
        goals.append(goal_ok)
        collisions_list.append(ep_coll)
        status = "HEDEF" if goal_ok else "ÇARPIŞMA/STEP" if ep_coll > 0 else "STEP"
        print(f"  Episode {ep+1:2d}: reward={ep_rew:8.1f}  steps={ep_len:3d}  coll={ep_coll}  -> {status}")

    print("\n" + "=" * 55)
    print("  ZORLU PARKUR SONUÇLARI")
    print("=" * 55)
    print(f"  Mean reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean steps:     {np.mean(lengths):.1f}")
    print(f"  Hedefe ulaşma:  {np.mean(goals):.1%} ({int(np.sum(goals))}/{n_episodes})")
    print(f"  Ort. çarpışma:  {np.mean(collisions_list):.2f} / episode")
    print("=" * 55)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zorlu sabit parkur testi")
    p.add_argument("--model", default="models_hybrid_2/best/best_model")
    p.add_argument("--vec_norm", default="models_hybrid_2/vec_normalize.pkl")
    p.add_argument("--n_episodes", type=int, default=20)
    args = p.parse_args()
    run(model_path=args.model, vec_norm_path=args.vec_norm, n_episodes=args.n_episodes)
