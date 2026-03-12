"""
Shared Policy (V2) - Zorlu sabit parkur testi (Pygame ile görsel).

env_shared_v3 ile eğitilmiş model + hard_course_config.py layout.

Kullanım (proje kökünden):
  python test_hard_course_pygame_v2.py --model models_shared/best/best_model
  python test_hard_course_pygame_v2.py --model models_shared/ppo_shared_final --n_episodes 5 --fps 8
"""

import argparse
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, _here)

import numpy as np

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from shared.env_shared_hard_course import DroneSwarmSharedHardCourseEnv
from hard_course_config import HARD_OBSTACLES


def run(model_path: str, vec_norm: str = None, n_episodes: int = 5, fps: int = 8, swap: bool = False):
    gs = 50.0
    n_obst = len(HARD_OBSTACLES)

    def make_env():
        return Monitor(DroneSwarmSharedHardCourseEnv(
            grid_size=gs,
            n_obstacles=n_obst,
            random_obstacles=False,
            swap_start_target=swap,
            random_swap=False,
            max_steps=500,
            formation_coef=0.3,
            render_mode="human",
        ))

    # Vec env: model buraya bağlı olacak
    vec = DummyVecEnv([make_env])
    if vec_norm and os.path.exists(vec_norm):
        vec = VecNormalize.load(vec_norm, vec)
        vec.training = False
        vec.norm_reward = False

    model = PPO.load(model_path, env=vec)
    print(f"Model yüklendi: {model_path}")

    env = vec.envs[0].env  # DroneSwarmSharedHardCourseEnv

    # Pygame görüntü sistemini başlat
    pygame.init()
    clock = pygame.time.Clock()

    for ep in range(n_episodes):
        obs = vec.reset()
        done = False
        ep_rew = 0.0
        step = 0

        print("=" * 50)
        print(f"[Episode {ep+1}/{n_episodes}] Başlıyor (swap={swap}, {n_obst} engel)")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec.step(action)

            r = float(rewards[0])
            d = bool(dones[0])
            info0 = infos[0]

            ep_rew += r
            step += 1
            done = d

            env.render()
            clock.tick(fps)

        term = info0.get("termination", "timeout")
        dist = info0.get("dist_to_goal", None)
        coll_info = info0.get("collision_debug")
        if dist is not None:
            print(f"[Episode {ep+1}] Sonuç: {term} | Step: {step} | Reward: {ep_rew:.1f} | Dist: {dist:.1f}")
        else:
            print(f"[Episode {ep+1}] Sonuç: {term} | Step: {step} | Reward: {ep_rew:.1f}")
        if coll_info:
            for c in coll_info:
                kind = []
                if c.get("hit_wall"):
                    kind.append("duvar")
                if c.get("hit_obstacle"):
                    kind.append("engel")
                kind_str = "+".join(kind) if kind else "bilinmiyor"
                px, py = float(c["pos"][0]), float(c["pos"][1])
                print(f"    - Drone {c['drone']}: {kind_str} çarpışması, pos=({px:.1f}, {py:.1f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Shared Policy - Zorlu sabit parkur Pygame testi")
    p.add_argument("--model", default="models_shared/best/best_model")
    p.add_argument("--vec_norm", default=None, help="VecNormalize pkl (boş ise model klasöründen bulunur)")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--swap", action="store_true", help="Start/hedef yer değiştirsin (HARD_START<->HARD_TARGET)")
    args = p.parse_args()

    vec = args.vec_norm
    if vec is None:
        base = os.path.dirname(args.model.rstrip("/"))
        if os.path.basename(base) == "best":
            base = os.path.dirname(base)
        vec = os.path.join(base, "vec_normalize.pkl")

    run(
        model_path=args.model,
        vec_norm=vec,
        n_episodes=args.n_episodes,
        fps=args.fps,
        swap=args.swap,
    )

