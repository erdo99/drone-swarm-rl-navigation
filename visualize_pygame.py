"""
Hybrid2 model için Pygame görselleştirme (rastgele start/target).
"""

import argparse
import os
import sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env import DroneSwarmEnvHybrid2

CELL = 12
BG = (13, 17, 23)
GRID = (40, 50, 65)
OBST = (180, 60, 60)
TGT = (60, 220, 60)
TGT_O = (34, 221, 34)
DRONE_COLORS = [(100, 180, 255), (100, 255, 180), (255, 200, 100), (200, 100, 255)]
CENTER = (255, 255, 100)
TRAIL = (255, 255, 68)
TXT = (255, 255, 255)


def to_screen(pos, gs):
    return int(pos[0] * CELL), int((gs - pos[1]) * CELL)


def run(model_path=None, vec_norm="models_hybrid_2/vec_normalize.pkl",
        n_obstacles=5, offset_scale=0.6, formation_coef=0.3, seed=42, n_episodes=1,
        wall_sliding=True, fps=30):
    gs, size = 50.0, int(50 * CELL)
    raw_env = DroneSwarmEnvHybrid2(
        grid_size=gs, n_obstacles=n_obstacles, offset_scale=offset_scale,
        formation_coef=formation_coef, wall_sliding=wall_sliding, seed=seed,
    )

    model, wrapped, obs_vec, display_env = None, None, None, raw_env
    if model_path and os.path.exists(model_path + ".zip"):
        def _make_env():
            return Monitor(DroneSwarmEnvHybrid2(
                grid_size=gs, n_obstacles=n_obstacles, offset_scale=offset_scale,
                formation_coef=formation_coef, wall_sliding=wall_sliding, seed=seed,
            ))
        wrapped = DummyVecEnv([_make_env])
        if vec_norm and os.path.exists(vec_norm):
            wrapped = VecNormalize.load(vec_norm, wrapped)
            wrapped.training = False
            wrapped.norm_reward = False
        model = PPO.load(model_path, env=wrapped)
        res = wrapped.reset()
        obs_vec = res[0] if isinstance(res, (tuple, list)) else res
        display_env = wrapped.venv.envs[0].env

    raw_env.reset(seed=seed)
    pygame.init()
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Drone Swarm Hybrid2 (Rastgele Start/Target)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    total_reward, step_count, episode_count = 0.0, 0, 0
    trail, max_trail = [], 80
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
                break
        if not running:
            break

        if model is not None:
            a, _ = model.predict(obs_vec, deterministic=True)
            action = a[0]
            obs_vec, rewards, dones, infos = wrapped.step(a)
            reward, info, done_env = float(rewards[0]), infos[0], dones[0]
        else:
            action = raw_env.action_space.sample()
            _, reward, term, trunc, info = raw_env.step(action)
            done_env = term or trunc

        total_reward += reward
        step_count += 1
        trail.append(display_env.center_pos.copy())
        if len(trail) > max_trail:
            trail.pop(0)

        screen.fill(BG)
        for x in range(0, int(gs) + 1, 5):
            sx = int(x * CELL)
            pygame.draw.line(screen, GRID, (sx, 0), (sx, size))
            pygame.draw.line(screen, GRID, (0, sx), (size, sx))
        for p in trail:
            px, py = to_screen(p, gs)
            pygame.draw.circle(screen, TRAIL, (px, py), 2)
        for o in display_env.obstacles:
            px, py = to_screen(o, gs)
            pygame.draw.circle(screen, OBST, (px, py), int(display_env.obstacle_radius * CELL))
        tx, ty = to_screen(display_env.target_pos, gs)
        pygame.draw.circle(screen, TGT_O, (tx, ty), int(2.5 * CELL), 2)
        pygame.draw.circle(screen, TGT, (tx, ty), int(2.0 * CELL))
        for i, dp in enumerate(display_env._get_drone_positions()):
            px, py = to_screen(dp, gs)
            pygame.draw.circle(screen, DRONE_COLORS[i], (px, py), int(1.2 * CELL))
        cx, cy = to_screen(display_env.center_pos, gs)
        pygame.draw.circle(screen, CENTER, (cx, cy), 4)
        screen.blit(font.render(f"Ep {episode_count+1}/{n_episodes} | Step {step_count}", True, TXT), (10, 10))
        screen.blit(font.render(f"Reward: {total_reward:.1f}", True, (170, 255, 170)), (10, 32))
        screen.blit(font.render(f"Dist: {info.get('dist_to_goal', 0):.1f}", True, (170, 170, 255)), (10, 54))

        if done_env:
            episode_count += 1
            if episode_count >= n_episodes:
                screen.blit(font.render("Tamamlandı", True, (170, 255, 170)), (size//2 - 50, size//2 - 12))
                pygame.display.flip()
                pygame.time.wait(1500)
                break
            if model is not None:
                res = wrapped.reset()
                obs_vec = res[0] if isinstance(res, (tuple, list)) else res
            else:
                raw_env.reset(seed=seed + episode_count)
            total_reward, step_count, trail = 0.0, 0, []

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--vec_norm", default="models_hybrid_2/vec_normalize.pkl")
    p.add_argument("--n_obstacles", type=int, default=5)
    p.add_argument("--offset_scale", type=float, default=0.6)
    p.add_argument("--formation_coef", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=30, help="Görselleştirme hızı (düşük = yavaş, örn. 5-10)")
    p.add_argument("--no_wall_sliding", action="store_true", help="Wall sliding'i kapat")
    args = p.parse_args()
    run(model_path=args.model, vec_norm=args.vec_norm, n_obstacles=args.n_obstacles,
        offset_scale=args.offset_scale, formation_coef=args.formation_coef,
        seed=args.seed, n_episodes=args.n_episodes,
        wall_sliding=not args.no_wall_sliding, fps=args.fps)
