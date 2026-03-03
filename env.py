"""
DroneSwarmEnvHybrid2 - Hybrid + rastgele başlangıç/hedef konumları.

Hybrid ile aynı action/obs yapısı, fakat reset'te:
- Başlangıç: haritanın herhangi bir yerinde (güvenli marjla)
- Hedef: başlangıçtan yeterince uzak, yine herhangi bir yerde
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import math


class DroneSwarmEnvHybrid2(gym.Env):
    """
    Hybrid_2: Ortak hız + per-drone offset, rastgele başlangıç/hedef.
    Her episode'da start ve target haritada farklı bölgelerde olabilir.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    FORMATION_OFFSETS = np.array([
        [-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5],
    ], dtype=np.float32)

    def __init__(
        self,
        grid_size: float = 50.0,
        n_obstacles: int = 5,
        n_obstacles_range: Optional[Tuple[int, int]] = None,
        safety_radius: float = 2.0,
        max_speed: float = 2.0,
        offset_scale: float = 0.6,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        wall_sliding: bool = True,
        formation_coef: float = 0.3,
        min_start_target_dist: float = 15.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.n_obstacles_range = n_obstacles_range
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.offset_scale = offset_scale
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.wall_sliding = wall_sliding
        self.formation_coef = formation_coef
        self.min_start_target_dist = min_start_target_dist

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        obs_dim = 2 + 2 + 2 + 8 + 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.np_random = np.random.default_rng(seed)
        self.renderer = None
        self.drone_positions = np.zeros((4, 2), dtype=np.float32)
        self.drone_velocities = np.zeros((4, 2), dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.obstacles: List[np.ndarray] = []
        self.obstacle_radius = 3.0
        self.step_count = 0
        self.done = False

    @property
    def center_pos(self) -> np.ndarray:
        return self.drone_positions.mean(axis=0).astype(np.float32)

    @property
    def center_vel(self) -> np.ndarray:
        return self.drone_velocities.mean(axis=0).astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.done = False
        self.drone_velocities = np.zeros((4, 2), dtype=np.float32)

        margin = 6.0
        lo, hi = margin, self.grid_size - margin

        # Rastgele başlangıç merkezi (haritanın herhangi bir yerinde)
        center = self.np_random.uniform(lo, hi, size=2).astype(np.float32)
        self.drone_positions = center + self.FORMATION_OFFSETS

        # Rastgele hedef, başlangıçtan en az min_start_target_dist uzakta
        for _ in range(500):
            target = self.np_random.uniform(lo, hi, size=2).astype(np.float32)
            if np.linalg.norm(target - center) >= self.min_start_target_dist:
                self.target_pos = target
                break
        else:
            self.target_pos = center + np.array([15.0, 0.0], dtype=np.float32)
            self.target_pos = np.clip(self.target_pos, lo, hi).astype(np.float32)

        n_obs = self.n_obstacles
        if self.n_obstacles_range is not None:
            lo, hi = self.n_obstacles_range
            n_obs = int(self.np_random.integers(lo, hi + 1))
        self._n_obstacles_this_episode = n_obs
        self._generate_obstacles()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert not self.done
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        alpha = 0.7
        shared_vel = action[0:2] * self.max_speed
        collisions: List[int] = []

        for i in range(4):
            offset = action[2 + 2*i:4 + 2*i] * self.max_speed * self.offset_scale
            desired = shared_vel + offset
            self.drone_velocities[i] = alpha * desired + (1 - alpha) * self.drone_velocities[i]
            new_pos = self.drone_positions[i] + self.drone_velocities[i]

            if self.wall_sliding:
                new_pos, self.drone_velocities[i] = self._apply_wall_sliding(
                    new_pos, self.drone_velocities[i]
                )
            else:
                new_pos = np.clip(new_pos, 3.0, self.grid_size - 3.0)

            all_new = self.drone_positions.copy()
            all_new[i] = new_pos
            hit = self._check_collisions(all_new)
            if i in hit:
                self.drone_velocities[i] *= 0.1
                collisions.append(i)
            else:
                self.drone_positions[i] = new_pos

        reward, terminated = self._compute_reward(collisions)
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            self.done = True

        obs = self._get_obs()
        info = self._get_info()
        info["collisions"] = len(collisions)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        if self.renderer is not None:
            import pygame
            pygame.quit()
            self.renderer = None

    def _generate_obstacles(self):
        self.obstacles = []
        center = self.center_pos
        n_obs = getattr(self, "_n_obstacles_this_episode", self.n_obstacles)
        for _ in range(n_obs):
            for _ in range(200):
                pos = self.np_random.uniform(5.0, self.grid_size - 5.0, size=2).astype(np.float32)
                if (np.linalg.norm(pos - center) > 7.0 and
                    np.linalg.norm(pos - self.target_pos) > 7.0 and
                    not any(np.linalg.norm(pos - o) < self.obstacle_radius * 2.5 for o in self.obstacles)):
                    self.obstacles.append(pos)
                    break

    def _get_drone_positions(self) -> np.ndarray:
        return self.drone_positions.astype(np.float32)

    def _check_collisions(self, drone_positions: np.ndarray) -> List[int]:
        collisions = []
        for i, dp in enumerate(drone_positions):
            for obs in self.obstacles:
                if np.linalg.norm(dp - obs) < self.safety_radius + self.obstacle_radius:
                    collisions.append(i)
                    break
            if np.any(dp < 1.0) or np.any(dp > self.grid_size - 1.0):
                if i not in collisions:
                    collisions.append(i)
        return collisions

    def _apply_wall_sliding(
        self, pos: np.ndarray, vel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        min_p, max_p = 3.0, self.grid_size - 3.0
        new_pos, new_vel = pos.copy(), vel.copy()
        for d in range(2):
            if new_pos[d] < min_p:
                new_pos[d], new_vel[d] = min_p, 0.0
            elif new_pos[d] > max_p:
                new_pos[d], new_vel[d] = max_p, 0.0
        return new_pos, new_vel

    def _obstacle_ray_distances(self, drone_pos: np.ndarray, n_rays: int = 4) -> np.ndarray:
        angles = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
        dists = np.ones(n_rays, dtype=np.float32) * self.grid_size
        for i, a in enumerate(angles):
            d = np.array([math.cos(a), math.sin(a)], dtype=np.float32)
            for obs in self.obstacles:
                rel = obs - drone_pos
                proj = np.dot(rel, d)
                if proj > 0 and np.linalg.norm(rel - proj * d) < self.obstacle_radius + self.safety_radius:
                    dists[i] = min(dists[i], proj)
            for dim in range(2):
                if d[dim] > 1e-6:
                    t = (self.grid_size - 3.0 - drone_pos[dim]) / d[dim]
                elif d[dim] < -1e-6:
                    t = (3.0 - drone_pos[dim]) / d[dim]
                else:
                    continue
                dists[i] = min(dists[i], t)
        return np.clip(dists / self.grid_size, 0.0, 1.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        norm = self.grid_size
        center = self.center_pos
        ideal = center + self.FORMATION_OFFSETS
        formation_errors = (self.drone_positions - ideal).flatten() / norm
        rel_goal = (self.target_pos - center) / norm
        rays = np.concatenate([
            self._obstacle_ray_distances(dp) for dp in self.drone_positions
        ])
        return np.concatenate([
            center / norm,
            self.center_vel / self.max_speed,
            rel_goal,
            formation_errors,
            rays,
        ]).astype(np.float32)

    def _compute_reward(self, collisions: List[int]) -> Tuple[float, bool]:
        center = self.center_pos
        dist = float(np.linalg.norm(center - self.target_pos))
        reward = -dist * 0.01 - len(collisions) * 10.0 - 0.01
        ideal = center + self.FORMATION_OFFSETS
        formation_err = np.linalg.norm(self.drone_positions - ideal)
        reward -= float(formation_err) * self.formation_coef
        terminated = False
        if dist < 3.0:
            reward += 1000.0
            terminated = True
        if len(collisions) >= 2:
            reward -= 20.0
            terminated = True
        return reward, terminated

    def _get_info(self) -> Dict[str, Any]:
        d = float(np.linalg.norm(self.center_pos - self.target_pos))
        return {
            "dist_to_goal": d,
            "step": self.step_count,
            "center_pos": self.center_pos.copy(),
            "target_pos": self.target_pos.copy(),
        }

    def _render_human(self):
        try:
            import pygame
        except ImportError:
            return
        cell = 12
        size = int(self.grid_size * cell)
        if self.renderer is None:
            pygame.init()
            self.renderer = pygame.display.set_mode((size, size))
            pygame.display.set_caption("Drone Swarm (Hybrid2 - Rastgele Start/Target)")
            self._clock = pygame.time.Clock()
        self.renderer.fill((20, 20, 30))
        for o in self.obstacles:
            pygame.draw.circle(
                self.renderer, (180, 60, 60),
                (int(o[0] * cell), int((self.grid_size - o[1]) * cell)),
                int(self.obstacle_radius * cell),
            )
        tx, ty = int(self.target_pos[0] * cell), int((self.grid_size - self.target_pos[1]) * cell)
        pygame.draw.circle(self.renderer, (60, 220, 60), (tx, ty), int(3.0 * cell))
        colors = [(100, 180, 255), (100, 255, 180), (255, 200, 100), (200, 100, 255)]
        for i, dp in enumerate(self.drone_positions):
            px, py = int(dp[0] * cell), int((self.grid_size - dp[1]) * cell)
            pygame.draw.circle(self.renderer, colors[i], (px, py), int(1.2 * cell))
        cx, cy = int(self.center_pos[0] * cell), int((self.grid_size - self.center_pos[1]) * cell)
        pygame.draw.circle(self.renderer, (255, 255, 100), (cx, cy), 4)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()

    def _render_rgb(self) -> np.ndarray:
        import io
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_facecolor("#141420")
        for o in self.obstacles:
            ax.add_patch(plt.Circle(o, self.obstacle_radius, color="#B43C3C", alpha=0.8))
        ax.add_patch(plt.Circle(self.target_pos, 2.0, color="#3CDD3C", alpha=0.8))
        for i, dp in enumerate(self.drone_positions):
            ax.add_patch(plt.Circle(dp, 1.0, color=["#64B4FF", "#64FFBA", "#FFC864", "#C864FF"][i], alpha=0.9))
        ax.plot(*self.center_pos, "y*", markersize=10)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        plt.close(fig)
        buf.seek(0)
        import PIL.Image
        return np.array(PIL.Image.open(buf))[:, :, :3]
