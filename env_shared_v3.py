"""
env_shared_v3.py — v3 Sürümü
v3 varsayılan: engeller tamamen rastgele (old gibi); obstacles_on_route=True ile rotaya yerleştirilebilir.
  - Ray 8, OBS_DIM 16 (4×16=64 obs)
  - Başarı: sadece merkez hedefe < 3.0
  

Mimari:
  - Parameter Sharing / CTDE
  - 4 drone × 16 lokal obs = 64 toplam obs
  - 4 drone × 2 hız = 8 toplam act
  - Her drone SADECE kendi lokal obs'unu kullanır → decentralized execution
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List


class DroneSwarmSharedEnv(gym.Env):
    """
    Gymnasium ortamı — Parameter Sharing / CTDE mimarisi için.

    Her drone:
      [pos_x, pos_y, vel_x, vel_y, to_target_x, to_target_y,
       formation_err_x, formation_err_y,
       ray_0, ray_45, ray_90, ray_135, ray_180, ray_225, ray_270, ray_315]
      → 16 boyutlu lokal gözlem (v4: ray 4→8, dünyaya sabitlenmiş)

    Sürü: 4 drone, kare formasyon
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    N_DRONES = 4
    OBS_DIM = 16  # v4: 8→16 (ray 4→8)
    ACT_DIM = 2

    def __init__(
        self,
        grid_size: float = 50.0,
        n_obstacles: int = 5,
        n_obstacles_range: Optional[Tuple[int, int]] = (5, 9),
        random_obstacles: bool = True,
        obstacles_on_route: bool = False,         # False: tümü rastgele (old gibi); True: route_obstacle_ratio kadarı rotada
        route_obstacle_ratio: float = 0.6,        # obstacles_on_route=True iken: engellerin kaçı rotada
        safety_radius: float = 2.0,
        obstacle_radius: float = 3.0,
        max_speed: float = 2.0,
        formation_size: float = 4.0,
        formation_coef: float = 0.3,
        momentum_alpha: float = 0.7,
        max_steps: int = 500,
        proximity_threshold: float = 2.0,
        proximity_penalty_coef: float = 0.1,
        min_drone_separation: float = 1.5,
        min_drone_separation_penalty: float = 15.0,
        min_start_target_dist: float = 15.0,      # YENİ: start-target minimum mesafe
        collision_penalty: float = 50.0,          # Her çarpışan drone cezası (önceki 10, hızlı çarpışmayı caydırmak için 50)
        heavy_collision_penalty: float = 100.0,   # 2+ drone çarpışınca ek ceza (önceki 20)
        success_reward: float = 2000.0,           # Hedefe ulaşma ödülü (önceki 1000)
        dist_reward_coef: float = 0.02,           # Mesafe azalma bonusu (her step dist_prev - dist_now)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.n_obstacles_range = n_obstacles_range
        self.random_obstacles = random_obstacles
        self.obstacles_on_route = obstacles_on_route
        self.route_obstacle_ratio = route_obstacle_ratio
        self.safety_radius = safety_radius
        self.obstacle_radius = obstacle_radius
        self.max_speed = max_speed
        self.formation_size = formation_size
        self.formation_coef = formation_coef
        self.momentum_alpha = momentum_alpha
        self.max_steps = max_steps
        self.proximity_threshold = proximity_threshold
        self.proximity_penalty_coef = proximity_penalty_coef
        self.min_drone_separation = min_drone_separation
        self.min_drone_separation_penalty = min_drone_separation_penalty
        self.min_start_target_dist = min_start_target_dist
        self.collision_penalty = collision_penalty
        self.heavy_collision_penalty = heavy_collision_penalty
        self.success_reward = success_reward
        self.dist_reward_coef = dist_reward_coef
        self.render_mode = render_mode

        s = formation_size / 2
        self.formation_offsets = np.array([
            [-s, -s], [s, -s], [-s, s], [s, s],
        ], dtype=np.float32)

        # Obs: 4 drone × 16 = 64
        total_obs = self.N_DRONES * self.OBS_DIM
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_obs,), dtype=np.float32
        )
        # Act: 4 drone × 2 = 8
        total_act = self.N_DRONES * self.ACT_DIM
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_act,), dtype=np.float32
        )

        self.positions = None
        self.velocities = None
        self.obstacles = None
        self.target = None
        self.step_count = 0
        self.np_random = np.random.default_rng(None)

        self._pygame = None
        self._screen = None
        self._clock = None

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Engel sayısı: her episode yeniden örnekle
        if self.random_obstacles and self.n_obstacles_range is not None:
            n_obs = int(self.np_random.integers(
                self.n_obstacles_range[0],
                self.n_obstacles_range[1] + 1
            ))
        else:
            n_obs = self.n_obstacles

        margin = 8.0
        lo, hi = margin, self.grid_size - margin

        # --- YENİ: Rastgele başlangıç merkezi (tüm harita) ---
        start_center = self.np_random.uniform(lo, hi, size=2).astype(np.float32)

        # --- YENİ: min_start_target_dist garantili rastgele hedef ---
        target = None
        for _ in range(500):
            candidate = self.np_random.uniform(lo, hi, size=2).astype(np.float32)
            if np.linalg.norm(candidate - start_center) >= self.min_start_target_dist:
                target = candidate
                break
        if target is None:
            # Fallback: start'tan sabit yönde
            direction = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            target = np.clip(
                start_center + direction * self.min_start_target_dist,
                lo, hi
            ).astype(np.float32)

        self.target = target
        self.positions = (start_center + self.formation_offsets).astype(np.float32)
        self.velocities = np.zeros((self.N_DRONES, 2), dtype=np.float32)
        self._prev_center_dist = float(np.linalg.norm(start_center - self.target))
        self.obstacles = self._generate_obstacles(n_obs, start_center, target)
        self.step_count = 0

        obs = self._get_obs()
        return obs, self._get_info()

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).reshape(self.N_DRONES, self.ACT_DIM)
        desired_velocities = action * self.max_speed
        alpha = self.momentum_alpha

        collisions = []
        collision_positions = {}
        for i in range(self.N_DRONES):
            blended_vel = alpha * desired_velocities[i] + (1.0 - alpha) * self.velocities[i]
            new_pos = self.positions[i] + blended_vel
            new_pos, new_vel = self._apply_wall_sliding(new_pos.copy(), blended_vel.copy())

            all_new = self.positions.copy()
            all_new[i] = new_pos
            hit = self._check_collisions(all_new)

            if i in hit:
                self.velocities[i] = new_vel * 0.1
                collisions.append(i)
                collision_positions[i] = new_pos
            else:
                self.positions[i] = new_pos.astype(np.float32)
                self.velocities[i] = new_vel.astype(np.float32)

        reward, done, info = self._compute_reward(collisions, collision_positions)
        self._prev_center_dist = info.get("dist_to_goal", float(np.linalg.norm(self.positions.mean(axis=0) - self.target)))
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------
    # OBS
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs_list = []
        center = self.positions.mean(axis=0)
        for i in range(self.N_DRONES):
            pos_norm = self.positions[i] / self.grid_size
            vel_norm = self.velocities[i] / self.max_speed
            to_target = self.target - self.positions[i]
            target_norm = to_target / (self.grid_size * np.sqrt(2) + 1e-6)
            ideal_pos = center + self.formation_offsets[i]
            formation_err = (self.positions[i] - ideal_pos) / self.grid_size
            rays = self._ray_distances_8dir(i)  # v4: 4→8 ray
            drone_obs = np.concatenate([
                pos_norm,       # 2
                vel_norm,       # 2
                target_norm,    # 2
                formation_err,  # 2
                rays,           # 8  → toplam 16
            ]).astype(np.float32)
            obs_list.append(drone_obs)
        return np.concatenate(obs_list)

    def get_per_drone_obs(self) -> List[np.ndarray]:
        """Her drone'un lokal obs'unu ayrı ayrı döner (inference için)."""
        full_obs = self._get_obs()
        return [full_obs[i * self.OBS_DIM:(i + 1) * self.OBS_DIM] for i in range(self.N_DRONES)]

    # ------------------------------------------------------------------
    # RAY CASTING
    # ------------------------------------------------------------------

    def _ray_distances_8dir(self, drone_idx: int) -> np.ndarray:
        """
        8 yönde ray cast — dünyaya sabitlenmiş, 45° aralıklı.
        Yönler: 0°(+x), 45°, 90°(+y), 135°, 180°(-x), 225°, 270°(-y), 315°

        Hedefe göre dönen 4-ray'den farklı olarak bu yönler sabit kalır;
        model köşegen engelleri de görür ve hangi tarafta geçit olduğunu
        daha net çıkarabilir.
        """
        import math
        pos = self.positions[drone_idx]
        max_ray = self.grid_size
        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        results = []

        for deg in angles_deg:
            rad = math.radians(deg)
            direction = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
            min_dist = float(max_ray)

            # Duvar mesafesi
            for dim in range(2):
                if abs(direction[dim]) > 1e-6:
                    d = ((self.grid_size if direction[dim] > 0 else 0.0) - pos[dim]) / direction[dim]
                    min_dist = min(min_dist, max(0.0, float(d)))

            # Engel mesafesi
            if self.obstacles is not None and len(self.obstacles) > 0:
                for obs_pos in self.obstacles:
                    to_obs = obs_pos - pos
                    proj = float(np.dot(to_obs, direction))
                    if proj > 0:
                        perp = float(np.linalg.norm(to_obs - proj * direction))
                        if perp < self.obstacle_radius:
                            hit_dist = proj - math.sqrt(max(0.0, self.obstacle_radius**2 - perp**2))
                            min_dist = min(min_dist, max(0.0, hit_dist))

            results.append(min_dist / max_ray)

        return np.array(results, dtype=np.float32)

    # ------------------------------------------------------------------
    # COLLISION CHECK
    # ------------------------------------------------------------------

    def _check_collisions(self, drone_positions: np.ndarray) -> List[int]:
        collisions = []
        for i, pos in enumerate(drone_positions):
            hit = False
            # Sınır kontrolü
            if (pos[0] <= self.safety_radius or pos[0] >= self.grid_size - self.safety_radius or
                    pos[1] <= self.safety_radius or pos[1] >= self.grid_size - self.safety_radius):
                hit = True
            # Engel kontrolü
            if not hit and self.obstacles is not None and len(self.obstacles) > 0:
                for obs_pos in self.obstacles:
                    if np.linalg.norm(pos - obs_pos) < self.safety_radius + self.obstacle_radius:
                        hit = True
                        break
            if hit:
                collisions.append(i)
        return collisions

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        collisions: Optional[List[int]] = None,
        collision_positions: Optional[Dict] = None,
    ) -> Tuple[float, bool, Dict]:
        if collisions is None:
            collisions = self._check_collisions(self.positions)
        if collision_positions is None:
            collision_positions = {}

        n_collisions = len(collisions)
        center = self.positions.mean(axis=0)
        dist_to_target = float(np.linalg.norm(center - self.target))

        # Her drone'un hedefe uzaklığı
        drone_dists = [float(np.linalg.norm(self.positions[i] - self.target))
                       for i in range(self.N_DRONES)]
        max_drone_dist = max(drone_dists)

        reward = -0.01 * dist_to_target - 0.01
        # Mesafe azalma bonusu (hedefe yaklaşma teşviki)
        reward += self.dist_reward_coef * (self._prev_center_dist - dist_to_target)
        done = False
        info = {}

        # Çarpışma cezası (collision_penalty, heavy_collision_penalty)
        reward -= self.collision_penalty * n_collisions
        if n_collisions >= 2:
            reward -= self.heavy_collision_penalty
            done = True
            info["termination"] = "heavy_collision"

        # Formasyon cezası
        ideal_positions = center + self.formation_offsets
        formation_error = float(np.linalg.norm(self.positions - ideal_positions))
        reward -= self.formation_coef * formation_error

        # Drone-drone mesafe cezaları
        reward -= self._proximity_penalty()
        reward -= self._min_separation_penalty()

        # Başarı koşulu: sadece sürü merkezi hedefe < 3.0
        if dist_to_target < 3.0:
            reward += self.success_reward
            done = True
            info["termination"] = "success"

        info["dist_to_goal"] = dist_to_target
        info["max_drone_dist"] = max_drone_dist
        info["formation_error"] = formation_error
        info["n_collisions"] = n_collisions
        info["center_pos"] = center.tolist()
        info["target_pos"] = self.target.tolist()

        return reward, done, info

    def _proximity_penalty(self) -> float:
        penalty = 0.0
        thresh = self.proximity_threshold
        coef = self.proximity_penalty_coef
        for i in range(self.N_DRONES):
            for j in range(i + 1, self.N_DRONES):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d < thresh:
                    penalty += coef * (thresh - d)
        return penalty

    def _min_separation_penalty(self) -> float:
        penalty = 0.0
        for i in range(self.N_DRONES):
            for j in range(i + 1, self.N_DRONES):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d < self.min_drone_separation:
                    penalty += self.min_drone_separation_penalty
        return penalty

    # ------------------------------------------------------------------
    # OBSTACLE GENERATION
    # ------------------------------------------------------------------

    def _generate_obstacles(
        self,
        n_obs: int,
        start_center: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        obstacles_on_route=True: engellerin route_obstacle_ratio kadari start->target rotasinda.
        obstacles_on_route=False: tumu rastgele (old ile uyumlu: margin 5, start/target 7, engel-engel 2.5*radius).
        """
        obstacles: List[np.ndarray] = []
        if self.obstacles_on_route:
            min_clear = self.obstacle_radius + self.safety_radius + self.formation_size + 2.0
            margin = self.obstacle_radius + 1.0
        else:
            # old env ile aynı değerler
            min_clear = 7.0
            margin = 5.0

        if self.obstacles_on_route:
            n_on_route = max(1, int(round(n_obs * self.route_obstacle_ratio)))
            n_random = n_obs - n_on_route
        else:
            n_on_route = 0
            n_random = n_obs

        # 1) Rotaya yerleştirilen engeller
        route_vec = target - start_center
        route_len = np.linalg.norm(route_vec)

        for _ in range(n_on_route):
            for _ in range(300):
                # start→target arasında rastgele t (kenarlardan uzak)
                t = self.np_random.uniform(0.2, 0.8)
                point_on_route = start_center + t * route_vec
                # Rotaya dik küçük sapma
                perp = np.array([-route_vec[1], route_vec[0]], dtype=np.float32)
                perp_len = np.linalg.norm(perp)
                if perp_len > 1e-6:
                    perp /= perp_len
                lateral = self.np_random.uniform(
                    -(self.obstacle_radius + 1.0),
                    (self.obstacle_radius + 1.0)
                )
                pos = (point_on_route + lateral * perp).astype(np.float32)
                pos = np.clip(pos, margin, self.grid_size - margin)

                if (np.linalg.norm(pos - start_center) > min_clear and
                        np.linalg.norm(pos - target) > min_clear):
                    ok = all(
                        np.linalg.norm(pos - ex) >= 2 * self.obstacle_radius + 1.0
                        for ex in obstacles
                    )
                    if ok:
                        obstacles.append(pos)
                        break

        # 2) Grid'e rastgele engeller (obstacles_on_route=False iken old ile aynı: 2.5*radius aralık)
        obs_spacing = 2.5 * self.obstacle_radius if not self.obstacles_on_route else 2 * self.obstacle_radius + 1.0
        for _ in range(n_random):
            for _ in range(200):
                pos = self.np_random.uniform(margin, self.grid_size - margin, size=2).astype(np.float32)
                if (np.linalg.norm(pos - start_center) > min_clear and
                        np.linalg.norm(pos - target) > min_clear):
                    ok = all(np.linalg.norm(pos - ex) >= obs_spacing for ex in obstacles)
                    if ok:
                        obstacles.append(pos)
                        break

        return np.array(obstacles, dtype=np.float32) if obstacles else np.zeros((0, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # WALL SLIDING
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------

    def _get_info(self) -> Dict[str, Any]:
        center = self.positions.mean(axis=0) if self.positions is not None else np.zeros(2)
        dist = float(np.linalg.norm(center - self.target)) if self.target is not None else 0.0
        return {
            "dist_to_goal": dist,
            "step": self.step_count,
            "center_pos": center.tolist(),
            "target_pos": self.target.tolist() if self.target is not None else [0.0, 0.0],
            "n_obstacles": len(self.obstacles) if self.obstacles is not None else 0,
        }

    # ------------------------------------------------------------------
    # RENDER — PYGAME
    # ------------------------------------------------------------------

    def _render_pygame(self):
        try:
            import pygame
        except ImportError:
            return

        CELL = 12
        W = H = int(self.grid_size * CELL)

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Drone Swarm — Shared Policy (CTDE)")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self._screen.fill((20, 20, 30))

        # Rota çizgisi
        if self.positions is not None and self.target is not None:
            center = self.positions.mean(axis=0)
            pygame.draw.line(
                self._screen, (60, 60, 80),
                (int(center[0] * CELL), int((self.grid_size - center[1]) * CELL)),
                (int(self.target[0] * CELL), int((self.grid_size - self.target[1]) * CELL)),
                1
            )

        # Engeller
        if self.obstacles is not None:
            for obs_pos in self.obstacles:
                pygame.draw.circle(
                    self._screen, (180, 60, 60),
                    (int(obs_pos[0] * CELL), int((self.grid_size - obs_pos[1]) * CELL)),
                    int(self.obstacle_radius * CELL)
                )

        # Hedef
        pygame.draw.circle(
            self._screen, (60, 220, 60),
            (int(self.target[0] * CELL), int((self.grid_size - self.target[1]) * CELL)),
            int(3.0 * CELL), 2
        )

        # Drone'lar
        colors = [(100, 180, 255), (255, 200, 80), (180, 255, 120), (255, 120, 200)]
        for i, pos in enumerate(self.positions):
            cx = int(pos[0] * CELL)
            cy = int((self.grid_size - pos[1]) * CELL)
            pygame.draw.circle(self._screen, colors[i], (cx, cy), int(1.2 * CELL))

        # Kütle merkezi
        center = self.positions.mean(axis=0)
        pygame.draw.circle(
            self._screen, (255, 255, 255),
            (int(center[0] * CELL), int((self.grid_size - center[1]) * CELL)), 3
        )

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    # ------------------------------------------------------------------
    # RENDER — RGB ARRAY (YENİ: Hybrid2'den)
    # ------------------------------------------------------------------

    def _render_rgb(self) -> np.ndarray:
        """Matplotlib ile render; video kaydetmek veya Jupyter ortaminda gostermek icin."""
        try:
            import io
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            return np.zeros((500, 500, 3), dtype=np.uint8)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_facecolor("#141420")
        ax.set_aspect("equal")

        # Rota
        if self.positions is not None and self.target is not None:
            center = self.positions.mean(axis=0)
            ax.plot(
                [center[0], self.target[0]],
                [center[1], self.target[1]],
                color="#404060", linewidth=1, linestyle="--"
            )

        # Engeller
        if self.obstacles is not None:
            for obs_pos in self.obstacles:
                ax.add_patch(plt.Circle(obs_pos, self.obstacle_radius, color="#B43C3C", alpha=0.8))

        # Hedef
        ax.add_patch(plt.Circle(self.target, 3.0, color="#3CDD3C", alpha=0.6, fill=False, linewidth=2))
        ax.plot(*self.target, "g*", markersize=12)

        # Drone'lar
        drone_colors = ["#64B4FF", "#FFC864", "#B4FF78", "#FF78C8"]
        for i, pos in enumerate(self.positions):
            ax.add_patch(plt.Circle(pos, 1.2, color=drone_colors[i], alpha=0.9))
            ax.annotate(str(i), pos, color="white", fontsize=7,
                        ha="center", va="center", fontweight="bold")

        # Kütle merkezi
        center = self.positions.mean(axis=0)
        ax.plot(*center, "w+", markersize=10, markeredgewidth=2)

        ax.set_title(f"Step: {self.step_count} | Dist: {np.linalg.norm(center - self.target):.1f}",
                     color="white", fontsize=9)
        fig.patch.set_facecolor("#141420")
        ax.tick_params(colors="gray")

        import io
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=80, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        try:
            from PIL import Image
            return np.array(Image.open(buf))[:, :, :3]
        except ImportError:
            import struct, zlib
            # PIL yoksa boş frame dön
            return np.zeros((500, 500, 3), dtype=np.uint8)