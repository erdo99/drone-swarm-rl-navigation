"""
visualize_pygame_v2.py — Shared Policy (V2) modelini görselleştir.

env_shared_v3 ile eğitilmiş modeli, pygame üzerinden 2B gridde izlemek için.

Kullanım (proje kökünden):
  python visualize_pygame_v2.py --model models_shared/best/best_model
  python visualize_pygame_v2.py --model models_shared/ppo_shared_final --n_episodes 5 --fps 8
  python visualize_pygame_v2.py --model models_shared.3/best/best_model --old_env   # eski davranış: engeller rotada değil, gride rastgele
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)


def visualize(model_path: str, vecnorm_path: str = None,
              n_episodes: int = 5, fps: int = 10, env_kwargs: dict = None):
    # V2: env_shared_v3 tabanlı shared policy ile uyumlu
    from env_shared_v3 import DroneSwarmSharedEnv

    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs["render_mode"] = "human"

    # Pygame video sistemini başlat
    pygame.init()

    def make():
        return DroneSwarmSharedEnv(**env_kwargs)

    vec_env = DummyVecEnv([make])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)
    print(f"Model: {model_path}")

    clock = pygame.time.Clock()
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        print(f"\nEpisode {ep+1}/{n_episodes}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    vec_env.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    vec_env.close()
                    pygame.quit()
                    return
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            done = dones[0]
            ep_reward += float(rewards[0])
            step += 1
            vec_env.envs[0].render()
            clock.tick(fps)

        term = infos[0].get("termination", "timeout")
        print(f"  Sonuç: {term}, ödül={ep_reward:.1f}, adım={step}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models_shared/best/best_model")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--old_env", action="store_true", help="Engeller rotada degil, grid'e rastgele (eski davranis)")
    args = parser.parse_args()

    env_kwargs = {}
    if args.old_env:
        env_kwargs["obstacles_on_route"] = False

    vecnorm = args.vecnorm
    if vecnorm is None:
        base = os.path.dirname(args.model.rstrip("/"))
        if os.path.basename(base) == "best":
            base = os.path.dirname(base)
        vecnorm = os.path.join(base, "vec_normalize.pkl")

    visualize(
        model_path=args.model,
        vecnorm_path=vecnorm,
        n_episodes=args.n_episodes,
        fps=args.fps,
        env_kwargs=env_kwargs if env_kwargs else None,
    )
