"""
Hybrid2 - Zorlu sabit parkur testi (Pygame ile görsel).

Her episode aynı layout, render ile izlenir.
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env_hard_course import DroneSwarmEnvHybrid2HardCourse
from hard_course_config import HARD_OBSTACLES

CELL, FPS = 12, 8
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


def _parse_pos(s):
    """'x,y' -> (float, float) veya None"""
    if not s or not s.strip():
        return None
    parts = s.strip().split(",")
    if len(parts) != 2:
        return None
    try:
        return (float(parts[0].strip()), float(parts[1].strip()))
    except ValueError:
        return None


def run(model_path=None, vec_norm="models_hybrid_2/vec_normalize.pkl",
        n_episodes=5, fps=8, swap_start_target=True, start_pos=None, target_pos=None):
    gs, size = 50.0, int(50 * CELL)
    custom_start = np.array(start_pos, dtype=np.float32) if start_pos else None
    custom_target = np.array(target_pos, dtype=np.float32) if target_pos else None
    raw_env = DroneSwarmEnvHybrid2HardCourse(
        grid_size=gs, n_obstacles=len(HARD_OBSTACLES),
        wall_sliding=True, offset_scale=0.6, formation_coef=0.3, seed=42,
        swap_start_target=swap_start_target,
        custom_start=custom_start, custom_target=custom_target,
    )

    model, wrapped, obs_vec, display_env = None, None, None, raw_env
    if model_path and os.path.exists(model_path + ".zip"):
        def _make_env():
            return Monitor(DroneSwarmEnvHybrid2HardCourse(
                grid_size=gs, n_obstacles=len(HARD_OBSTACLES),
                formation_coef=0.3, wall_sliding=True, offset_scale=0.6, seed=42,
                swap_start_target=swap_start_target,
                custom_start=custom_start, custom_target=custom_target,
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
    else:
        display_env = raw_env

    raw_env.reset(seed=42)
    pygame.init()
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Hybrid2 - Zorlu Parkur (Sabit Layout)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    total_reward, step_count, episode_count = 0.0, 0, 0
    trail, max_trail = [], 80
    running = True

    if start_pos is not None and target_pos is not None:
        swap_txt = f"Özel: start={start_pos}, hedef={target_pos}"
    else:
        swap_txt = "SWAP (start=sağ üst, hedef=sol alt)" if swap_start_target else "NORMAL (start=sol alt, hedef=sağ üst)"
    print("=" * 50)
    print("  Zorlu Parkur - Pygame Görsel Test")
    print(f"  {swap_txt} | {len(HARD_OBSTACLES)} engel: hard_course_config.py")
    print("=" * 50)

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

        goal_ok = info.get("dist_to_goal", 999) < 3.0
        status = "HEDEF!" if goal_ok else "Gidiyor..."
        screen.blit(font.render(f"Ep {episode_count+1}/{n_episodes} | Step {step_count} | {status}", True, TXT), (10, 10))
        screen.blit(font.render(f"Reward: {total_reward:.1f}", True, (170, 255, 170)), (10, 32))
        screen.blit(font.render(f"Dist: {info.get('dist_to_goal', 0):.1f}", True, (170, 170, 255)), (10, 54))

        if done_env:
            print(f"[Episode {episode_count+1}] Bitti: {'HEDEF' if goal_ok else 'ÇARPIŞMA/STEP'} | Step: {step_count} | Reward: {total_reward:.1f}")
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
                raw_env.reset(seed=42)
            total_reward, step_count, trail = 0.0, 0, []

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zorlu parkur - Pygame ile görsel test")
    p.add_argument("--model", default="models_hybrid_2/best/best_model")
    p.add_argument("--vec_norm", default="models_hybrid_2/vec_normalize.pkl")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--no_swap", action="store_true", help="Start/hedef yer değiştirmesin (normal: sol alt -> sağ üst)")
    p.add_argument("--start", type=str, default="", help="Başlangıç noktası 'x,y' (örn. 5,5). Verilirse --target da gerekir.")
    p.add_argument("--target", type=str, default="", help="Hedef noktası 'x,y' (örn. 45,45)")
    args = p.parse_args()
    start_pos = _parse_pos(args.start) if args.start else None
    target_pos = _parse_pos(args.target) if args.target else None
    if (start_pos is None) != (target_pos is None):
        print("Uyarı: --start ve --target birlikte verilmeli. Config/swap kullanılıyor.")
        start_pos = target_pos = None
    run(model_path=args.model, vec_norm=args.vec_norm,
        n_episodes=args.n_episodes, fps=args.fps,
        swap_start_target=not args.no_swap, start_pos=start_pos, target_pos=target_pos)
