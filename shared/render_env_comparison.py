"""
render_env_comparison.py — Env layout görselleştirme

1) Tek resim: Eski shared env (obstacles_on_route=False) nasıl ortam oluşturuyor
   - Başlangıç, hedef, engeller, drone konumları

2) 30'lu kolaj: Eski shared env için 30 farklı ortam (5x6 grid)

3) Tek resim: Yeni shared env (obstacles_on_route=True, rota koridoru) layout'u

4) 30'lu kolaj: Yeni shared env için 30 farklı ortam

5) Tek resim: Centralized Hybrid2 env (old/env.py) layout'u

6) 30'lu kolaj: Centralized Hybrid2 env için 30 farklı ortam

7) Tek resim: Shared V2 env (env_shared_v2.py, Hybrid2 ile aynı engel yerleşimi) layout'u

8) 30'lu kolaj: Shared V2 env için 30 farklı ortam

Kullanım (proje kökünden):
  python shared/render_env_comparison.py
  python shared/render_env_comparison.py --seed 42
  python shared/render_env_comparison.py --out_dir ./output
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

_old_dir = os.path.join(_root, "old")
if os.path.isdir(_old_dir) and _old_dir not in sys.path:
    sys.path.insert(0, _old_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_layout(env):
    """Env reset edilmiş olmalı. positions/target/obstacles döner (shared veya centralized)."""
    # Shared-policy env'de: positions, target
    # Hybrid2 env'de: drone_positions, target_pos
    if hasattr(env, "positions"):
        positions = env.positions.copy()
    else:
        positions = env.drone_positions.copy()

    if hasattr(env, "target"):
        target = env.target.copy()
    else:
        target = env.target_pos.copy()

    obstacles = getattr(env, "obstacles", None)
    if obstacles is None:
        obstacles_arr = np.zeros((0, 2))
    else:
        obstacles_arr = np.array(obstacles, dtype=np.float32).copy()

    grid_size = getattr(env, "grid_size", 50.0)
    obstacle_radius = getattr(env, "obstacle_radius", 3.0)

    return {
        "positions": positions,
        "target": target,
        "obstacles": obstacles_arr,
        "grid_size": grid_size,
        "obstacle_radius": obstacle_radius,
    }


def draw_layout(ax, layout, title="", small=False):
    """Bir layout'u ax üzerine çizer."""
    gs = layout["grid_size"]
    pos = layout["positions"]
    tgt = layout["target"]
    obs = layout["obstacles"]
    r_obs = layout["obstacle_radius"]

    ax.set_xlim(0, gs)
    ax.set_ylim(0, gs)
    ax.set_aspect("equal")
    ax.set_facecolor("#141420")

    # Engeller
    for o in obs:
        ax.add_patch(mpatches.Circle(o, r_obs, color="#B43C3C", alpha=0.8))

    # Hedef
    ax.add_patch(mpatches.Circle(tgt, 2.0, color="#3CDD3C", alpha=0.9))

    # Merkez (başlangıç)
    center = pos.mean(axis=0)
    ax.plot(center[0], center[1], "y*", markersize=14 if not small else 6)

    # Drone'lar
    colors = ["#64B4FF", "#FFC864", "#64FFBA", "#C864FF"]
    for i, p in enumerate(pos):
        ax.add_patch(mpatches.Circle(p, 1.0, color=colors[i], alpha=0.95))

    # Start → Target çizgisi (isteğe bağlı)
    ax.plot([center[0], tgt[0]], [center[1], tgt[1]], "w--", alpha=0.4, linewidth=1)

    if title:
        ax.set_title(title, fontsize=8 if small else 10)
    if small:
        ax.set_xticks([])
        ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description="Env layout görselleştirme")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default="./shared/env_comparison_output")
    parser.add_argument("--n_collection", type=int, default=30, help="Kolajdaki ortam sayısı")
    args = parser.parse_args()

    from env_shared import DroneSwarmSharedEnv
    # Shared V2 env (env_shared_v2.py) – Hybrid2 ile aynı engel yerleşimi
    try:
        import importlib
        env_v2_module = importlib.import_module("env_shared_v2")
        DroneSwarmSharedEnvV2 = getattr(env_v2_module, "DroneSwarmSharedEnv")
    except Exception:
        DroneSwarmSharedEnvV2 = None

    # Shared V3 env (env_shared_v3.py) – rota koridorlu yeni tasarım
    try:
        import importlib as _importlib_v3
        env_v3_module = _importlib_v3.import_module("env_shared_v3")
        DroneSwarmSharedEnvV3 = getattr(env_v3_module, "DroneSwarmSharedEnv")
    except Exception:
        DroneSwarmSharedEnvV3 = None

    # Centralized Hybrid2 env (V1) — old/env.py
    try:
        from old.env import DroneSwarmEnvHybrid2
    except ImportError:
        DroneSwarmEnvHybrid2 = None

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1) TEK RESİM: Eski env (obstacles_on_route=False) ---
    env_old = DroneSwarmSharedEnv(
        grid_size=50.0,
        n_obstacles_range=(7, 9),
        random_obstacles=True,
        obstacles_on_route=False,
    )
    env_old.reset(seed=args.seed)
    layout_single = get_layout(env_old)

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    draw_layout(ax1, layout_single, title="Eski env: Engeller rastgele grid (obstacles_on_route=False)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    path1 = os.path.join(args.out_dir, "env_old_single.png")
    fig1.savefig(path1, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(f"Kaydedildi: {path1}")

    # --- 2) 30'LU KOLAJ: Eski shared env (obstacles_on_route=False), her biri farklı reset ---
    n = args.n_collection
    ncol = 6
    nrow = (n + ncol - 1) // ncol

    fig2, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
    axes = np.atleast_2d(axes)

    for idx in range(n):
        env_old.reset(seed=args.seed + idx)
        layout = get_layout(env_old)
        r, c = idx // ncol, idx % ncol
        draw_layout(axes[r, c], layout, title=f"#{idx+1}", small=True)

    # Fazla subplot'ları gizle
    for idx in range(n, nrow * ncol):
        r, c = idx // ncol, idx % ncol
        axes[r, c].set_visible(False)

    fig2.suptitle(f"Eski env: {n} farklı ortam kolajı (obstacles_on_route=False)", fontsize=12)
    path2 = os.path.join(args.out_dir, "env_old_grid_30.png")
    fig2.savefig(path2, dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"Kaydedildi: {path2}")

    # --- 3) Yeni shared env: tek örnek ---
    env_new = DroneSwarmSharedEnv(
        grid_size=50.0,
        n_obstacles_range=(7, 9),
        random_obstacles=True,
        obstacles_on_route=True,
        route_corridor_width=16.0,
    )
    env_new.reset(seed=args.seed)
    layout_new = get_layout(env_new)

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
    draw_layout(ax3, layout_new, title="Yeni env: Engeller rotada (obstacles_on_route=True)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    path3 = os.path.join(args.out_dir, "env_new_single.png")
    fig3.savefig(path3, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(f"Kaydedildi: {path3}")

    # --- 4) Yeni shared env: 30'lu kolaj ---
    fig4, axes4 = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
    axes4 = np.atleast_2d(axes4)
    for idx in range(n):
        env_new.reset(seed=args.seed + idx)
        layout = get_layout(env_new)
        r, c = idx // ncol, idx % ncol
        draw_layout(axes4[r, c], layout, title=f"#{idx+1}", small=True)
    for idx in range(n, nrow * ncol):
        r, c = idx // ncol, idx % ncol
        axes4[r, c].set_visible(False)
    fig4.suptitle(f"Yeni env: {n} farklı ortam kolajı (obstacles_on_route=True)", fontsize=12)
    path4 = os.path.join(args.out_dir, "env_new_grid_30.png")
    fig4.savefig(path4, dpi=100, bbox_inches="tight")
    plt.close(fig4)
    print(f"Kaydedildi: {path4}")

    # --- 5) Centralized Hybrid2 env (old/env.py) tek örnek ---
    if DroneSwarmEnvHybrid2 is not None:
        env_central = DroneSwarmEnvHybrid2(
            grid_size=50.0,
            n_obstacles=8,
            n_obstacles_range=(7, 9),
            safety_radius=2.0,
            max_speed=2.0,
            offset_scale=0.6,
            max_steps=500,
            wall_sliding=True,
            formation_coef=0.3,
            proximity_threshold=2.0,
            proximity_penalty_coef=0.1,
            min_drone_separation=1.5,
            min_drone_separation_penalty=15.0,
            seed=args.seed,
        )
        env_central.reset(seed=args.seed)
        layout_central = get_layout(env_central)

        fig5, ax5 = plt.subplots(1, 1, figsize=(8, 8))
        draw_layout(ax5, layout_central, title="Centralized Hybrid2 env (old/env.py)")
        ax5.set_xlabel("x")
        ax5.set_ylabel("y")
        path5 = os.path.join(args.out_dir, "env_central_single.png")
        fig5.savefig(path5, dpi=120, bbox_inches="tight")
        plt.close(fig5)
        print(f"Kaydedildi: {path5}")

        # --- 6) Centralized Hybrid2 env: 30'lu kolaj ---
        fig6, axes6 = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
        axes6 = np.atleast_2d(axes6)
        for idx in range(n):
            env_central.reset(seed=args.seed + idx)
            layout = get_layout(env_central)
            r, c = idx // ncol, idx % ncol
            draw_layout(axes6[r, c], layout, title=f"#{idx+1}", small=True)
        for idx in range(n, nrow * ncol):
            r, c = idx // ncol, idx % ncol
            axes6[r, c].set_visible(False)
        fig6.suptitle(f"Centralized Hybrid2: {n} farklı ortam kolajı", fontsize=12)
        path6 = os.path.join(args.out_dir, "env_central_grid_30.png")
        fig6.savefig(path6, dpi=100, bbox_inches="tight")
        plt.close(fig6)
        print(f"Kaydedildi: {path6}")

    # --- 7) Shared V2 env: tek örnek ve 30'lu kolaj ---
    if DroneSwarmSharedEnvV2 is not None:
        env_v2 = DroneSwarmSharedEnvV2(
            grid_size=50.0,
            n_obstacles=5,
            n_obstacles_range=(5, 9),
            random_obstacles=True,
            safety_radius=2.0,
            obstacle_radius=3.0,
            max_speed=2.0,
            formation_size=4.0,
            formation_coef=0.3,
            momentum_alpha=0.7,
            max_steps=500,
            proximity_threshold=2.0,
            proximity_penalty_coef=0.1,
            min_drone_separation=1.5,
            min_drone_separation_penalty=15.0,
        )
        env_v2.reset(seed=args.seed)
        layout_v2 = get_layout(env_v2)

        fig7, ax7 = plt.subplots(1, 1, figsize=(8, 8))
        draw_layout(ax7, layout_v2, title="Shared V2 env (env_shared_v2.py, Hybrid2 obstacle logic)")
        ax7.set_xlabel("x")
        ax7.set_ylabel("y")
        path7 = os.path.join(args.out_dir, "env_shared_v2_single.png")
        fig7.savefig(path7, dpi=120, bbox_inches="tight")
        plt.close(fig7)
        print(f"Kaydedildi: {path7}")

        fig8, axes8 = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
        axes8 = np.atleast_2d(axes8)
        for idx in range(n):
            env_v2.reset(seed=args.seed + idx)
            layout = get_layout(env_v2)
            r, c = idx // ncol, idx % ncol
            draw_layout(axes8[r, c], layout, title=f"#{idx+1}", small=True)
        for idx in range(n, nrow * ncol):
            r, c = idx // ncol, idx % ncol
            axes8[r, c].set_visible(False)
        fig8.suptitle(f"Shared V2 env: {n} farklı ortam kolajı (Hybrid2 obstacle logic)", fontsize=12)
        path8 = os.path.join(args.out_dir, "env_shared_v2_grid_30.png")
        fig8.savefig(path8, dpi=100, bbox_inches="tight")
        plt.close(fig8)
        print(f"Kaydedildi: {path8}")

    # --- 8) Shared V3 env: tek örnek ve 30'lu kolaj ---
    if DroneSwarmSharedEnvV3 is not None:
        env_v3 = DroneSwarmSharedEnvV3(
            grid_size=50.0,
            n_obstacles=5,
            n_obstacles_range=(5, 9),
            random_obstacles=True,
            obstacles_on_route=True,
            route_obstacle_ratio=0.6,
            safety_radius=2.0,
            obstacle_radius=3.0,
            max_speed=2.0,
            formation_size=4.0,
            formation_coef=0.3,
            momentum_alpha=0.7,
            max_steps=500,
            proximity_threshold=2.0,
            proximity_penalty_coef=0.1,
            min_drone_separation=1.5,
            min_drone_separation_penalty=15.0,
        )
        env_v3.reset(seed=args.seed)
        layout_v3 = get_layout(env_v3)

        fig9, ax9 = plt.subplots(1, 1, figsize=(8, 8))
        draw_layout(ax9, layout_v3, title="Shared V3 env (env_shared_v3.py, route corridor + min_start_target_dist)")
        ax9.set_xlabel("x")
        ax9.set_ylabel("y")
        path9 = os.path.join(args.out_dir, "env_shared_v3_single.png")
        fig9.savefig(path9, dpi=120, bbox_inches="tight")
        plt.close(fig9)
        print(f"Kaydedildi: {path9}")

        fig10, axes10 = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
        axes10 = np.atleast_2d(axes10)
        for idx in range(n):
            env_v3.reset(seed=args.seed + idx)
            layout = get_layout(env_v3)
            r, c = idx // ncol, idx % ncol
            draw_layout(axes10[r, c], layout, title=f"#{idx+1}", small=True)
        for idx in range(n, nrow * ncol):
            r, c = idx // ncol, idx % ncol
            axes10[r, c].set_visible(False)
        fig10.suptitle(f"Shared V3 env: {n} farklı ortam kolajı (route corridor + 5–9 obstacles)", fontsize=12)
        path10 = os.path.join(args.out_dir, "env_shared_v3_grid_30.png")
        fig10.savefig(path10, dpi=100, bbox_inches="tight")
        plt.close(fig10)
        print(f"Kaydedildi: {path10}")

    print(f"\nTüm görseller: {args.out_dir}")


if __name__ == "__main__":
    main()
