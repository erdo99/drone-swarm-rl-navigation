"""
render_env_comparison.py — Env layout görselleştirme

1) Tek resim: Eski env (obstacles_on_route=False) nasıl ortam oluşturuyor
   - Başlangıç, hedef, engeller, drone konumları

2) 30'lu kolaj: 30 farklı ortam ızgara halinde (5x6 grid)

Kullanım (proje kökünden):
  python shared/render_env_comparison.py
  python shared/render_env_comparison.py --seed 42
  python shared/render_env_comparison.py --out_dir ./output
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_layout(env):
    """Env reset edilmiş olmalı. positions, target, obstacles döner."""
    return {
        "positions": env.positions.copy(),
        "target": env.target.copy(),
        "obstacles": env.obstacles.copy() if env.obstacles is not None else np.zeros((0, 2)),
        "grid_size": env.grid_size,
        "obstacle_radius": env.obstacle_radius,
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

    # --- 2) 30'LU KOLAJ: Her biri farklı reset ---
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

    # --- 3) Yeni env: tek örnek ---
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

    # --- 4) Yeni env: 30'lu kolaj ---
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

    print(f"\nTüm görseller: {args.out_dir}")


if __name__ == "__main__":
    main()
