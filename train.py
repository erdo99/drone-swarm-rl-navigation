"""
Hybrid2 Drone Swarm eğitimi: rastgele başlangıç/hedef konumları.

Kullanım:
    python hybrid_2/train.py
    python hybrid_2/train.py --timesteps 500000 --n_obstacles 3
"""

import argparse
import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from ppo_agent import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--n_obstacles", type=int, default=5, help="Sabit engel sayısı (n_obstacles_range verilmezse)")
    p.add_argument("--n_obstacles_range", type=str, default="5,9", help="Engel sayısı aralığı (min,max) her episode rastgele; boş bırakılırsa sabit n_obstacles")
    p.add_argument("--no_wall_sliding", action="store_true")
    p.add_argument("--offset_scale", type=float, default=0.6)
    p.add_argument("--formation_coef", type=float, default=0.3)
    p.add_argument("--save_dir", default="./models_hybrid_2/")
    p.add_argument("--log_dir", default="./logs_hybrid_2/")
    p.add_argument("--eval_freq", type=int, default=25_000)
    p.add_argument("--save_freq", type=int, default=50_000)
    args = p.parse_args()

    n_obs_range = None
    if args.n_obstacles_range:
        parts = args.n_obstacles_range.strip().split(",")
        if len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
            n_obs_range = (lo, hi) if lo <= hi else None
    if n_obs_range:
        print("Engel sayısı: her episode", n_obs_range[0], "-", n_obs_range[1], "arası rastgele")
    print("=" * 60)
    print("  Drone Swarm - Hybrid2 (Rastgele Start/Target)")
    print("=" * 60)
    model, env = train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_obstacles=args.n_obstacles,
        n_obstacles_range=n_obs_range,
        wall_sliding=not args.no_wall_sliding,
        offset_scale=args.offset_scale,
        formation_coef=args.formation_coef,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )
    print("\nBitti! Model:", args.save_dir)
    print("TensorBoard: tensorboard --logdir", args.log_dir + "/tensorboard/")


if __name__ == "__main__":
    main()
