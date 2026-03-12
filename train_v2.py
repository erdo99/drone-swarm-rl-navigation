"""
train_v2.py — Shared Policy (Parameter Sharing) Eğitim Giriş Noktası

Kullanım (proje kökünden):
  python train_v2.py
  python train_v2.py --timesteps 3000000
  python train_v2.py --timesteps 1000000 --n_obstacles_range "7,9"

v4 env değişiklikleri (`env_shared_v3.py` içinde uygulanır):
  - Ray 4→8 (dünyaya sabitlenmiş 45° aralıklı) → OBS_DIM 12→16, toplam 64 obs
  - Başarı: merkez < 3.0 VE her drone < 5.0
  - Geride drone cezası aktif

4 ayrı model DEĞİL: 1 PPO modeli, her drone aynı ağırlıkları kullanır (parameter sharing).
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from ppo_agent_v2 import train


def _next_available_dirs(save_dir: str, log_dir: str):
    """Kayıtlı model varsa yeni klasör: models_shared.1, .2, ..."""
    save_dir = os.path.normpath(save_dir).replace("\\", "/").rstrip("/")
    log_dir = os.path.normpath(log_dir).replace("\\", "/").rstrip("/")
    save_parent = os.path.dirname(save_dir) or "."
    log_parent = os.path.dirname(log_dir) or "."
    save_name = os.path.basename(save_dir)
    log_name = os.path.basename(log_dir)
    for i in range(100):
        if i == 0:
            sd = save_dir + "/"
            ld = log_dir + "/"
        else:
            sd = os.path.join(save_parent, save_name + "." + str(i)).replace("\\", "/") + "/"
            ld = os.path.join(log_parent, log_name + "." + str(i)).replace("\\", "/") + "/"
        if not os.path.exists(sd):
            return sd, ld
    return save_dir + "/", log_dir + "/"


def main():
    parser = argparse.ArgumentParser(description="Drone Swarm — Shared Policy Eğitimi")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--save_dir", type=str, default="models_shared")
    parser.add_argument("--log_dir", type=str, default="logs_shared")
    parser.add_argument("--n_obstacles", type=int, default=5)
    parser.add_argument("--n_obstacles_range", type=str, default="7,9")
    parser.add_argument("--no_random", action="store_true", help="Sabit engel sayısı")
    parser.add_argument("--grid_size", type=float, default=50.0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--formation_coef", type=float, default=0.3)
    parser.add_argument("--ent_coef", type=float, default=0.02, help="PPO entropy coefficient (varsayılan 0.02)")
    parser.add_argument("--no_obstacles_on_route", action="store_true", help="Engeller rotaya degil grid'e rastgele yerlestirilir")
    parser.add_argument(
        "--route_obstacle_ratio",
        type=float,
        default=0.6,
        help="Engellerin ne kadarı start→target rotası koridorunda olsun (0-1)",
    )
    parser.add_argument("--proximity_threshold", type=float, default=2.0, help="Drone-drone yakınlık eşiği")
    parser.add_argument("--proximity_penalty_coef", type=float, default=0.1, help="Yakınlık ceza katsayısı")
    parser.add_argument("--min_drone_separation", type=float, default=1.5, help="İç içe girme eşiği")
    parser.add_argument("--min_drone_separation_penalty", type=float, default=15.0, help="İç içe girme cezası")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=25_000)
    parser.add_argument("--save_freq", type=int, default=50_000)
    parser.add_argument("--no_auto_version", action="store_true", help="Aynı klasöre yaz (üzerine yaz)")
    args = parser.parse_args()

    save_dir = args.save_dir
    log_dir = args.log_dir
    if not args.no_auto_version:
        save_dir, log_dir = _next_available_dirs(save_dir, log_dir)
        if save_dir != args.save_dir or log_dir != args.log_dir:
            print("Mevcut klasör dolu, yeni sürüm kullanılıyor:")
            print("  save_dir:", save_dir)
            print("  log_dir:", log_dir)

    if args.no_random:
        n_obstacles_range = (args.n_obstacles, args.n_obstacles)
        random_obstacles = False
    else:
        lo, hi = map(int, args.n_obstacles_range.split(","))
        n_obstacles_range = (lo, hi)
        random_obstacles = True

    env_kwargs = dict(
        grid_size=args.grid_size,
        n_obstacles=args.n_obstacles,
        n_obstacles_range=n_obstacles_range,
        random_obstacles=random_obstacles,
        max_steps=args.max_steps,
        formation_coef=args.formation_coef,
        obstacles_on_route=not args.no_obstacles_on_route,
        route_obstacle_ratio=args.route_obstacle_ratio,
        proximity_threshold=args.proximity_threshold,
        proximity_penalty_coef=args.proximity_penalty_coef,
        min_drone_separation=args.min_drone_separation,
        min_drone_separation_penalty=args.min_drone_separation_penalty,
    )

    print("=" * 55)
    print("Drone Swarm — Shared Policy (Parameter Sharing)")
    print("=" * 55)
    print(f"Timesteps   : {args.timesteps:,}")
    print(f"Engel       : {'rastgele ' + str(n_obstacles_range)}" if random_obstacles else f"Engel: sabit {args.n_obstacles}")
    print(f"ent_coef    : {args.ent_coef}")
    # OBS_DIM=16 (v4: 8-ray), 4*16=64 obs, 4*2=8 act
    print("MİMARİ: 1 model, 64 obs → 8 act (4×16 → 4×2), net [256,256], n_envs={}".format(args.n_envs))
    print("=" * 55)

    train(
        total_timesteps=args.timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        env_kwargs=env_kwargs,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        ent_coef=args.ent_coef,
    )


if __name__ == "__main__":
    main()