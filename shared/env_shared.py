"""
env_shared.py — Parameter Sharing + Local Observation Ortamı

MİMARİ:
  Centralized Training, Decentralized Execution (CTDE)
  - 1 PPO model eğitilir, 4 drone için aynı ağırlıklar kullanılır
  - Her drone sadece kendi lokal gözlemini kullanarak karar verir
  - Eğitim sırasında Gymnasium wrapper tüm drone'ları birleştirir

DÜZELTMELER (v2):
  1. Momentum eklendi (alpha=0.7, Hybrid2 ile aynı)
  2. Ray sayısı 2 → 4 (ileri, sağ, geri, sol — 4 yön)
  3. OBS_DIM 10 → 12 (2 ekstra ray için güncellendi)
  4. Formasyon hatası normalize ölçeği tutarlı hale getirildi

OBSERVATION UZAYI (12 boyut, per-drone):
  [0-1]   Kendi pozisyonu (x, y) — normalize [0,1]
  [2-3]   Kendi hızı (vx, vy) — normalize [-1,1]
  [4-5]   Hedefe göre relative vektör (dx, dy) — normalize
  [6-7]   İdeal formasyondan sapma (ex, ey) — normalize (grid_size ile)
  [8-11]  4-yönlü ray mesafesi (ileri, sağ, geri, sol) — [0,1]

ACTION UZAYI (2 boyut, per-drone):
  [0]  vx — [-1,1] × max_speed
  [1]  vy — [-1,1] × max_speed
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DroneSwarmSharedEnv(gym.Env):
    """
    Gymnasium ortamı — Parameter Sharing / CTDE mimarisi için.

    Bu ortam 4 drone'u tek bir agent gibi sarmalıyor.
    Ama içeride her drone için ayrı obs üretip ayrı action alıyor.

    Stable-Baselines3 ile kullanmak için:
      - obs shape: (N_DRONES * OBS_DIM,) = (48,)   <- 4*12
      - act shape: (N_DRONES * ACT_DIM,) = (8,)
    Model içeride her 12 boyutu ayrı drone obs'u olarak işler.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    N_DRONES = 4
    OBS_DIM = 12   # 2(pos) + 2(vel) + 2(goal) + 2(formation_err) + 4(rays)
    ACT_DIM = 2

    def __init__(
        self,
        grid_size: float = 50.0,
        n_obstacles: int = 5,
        n_obstacles_range: tuple = (5, 9),
        random_obstacles: bool = True,
        safety_radius: float = 2.0,
        obstacle_radius: float = 3.0,
        max_speed: float = 2.0,
        formation_size: float = 4.0,
        formation_coef: float = 0.3,
        momentum_alpha: float = 0.7,
        max_steps: int = 500,
        obstacles_on_route: bool = True,
        route_corridor_width: float = 16.0,
        proximity_threshold: float = 2.0,
        proximity_penalty_coef: float = 0.1,
        min_drone_separation: float = 1.5,
        min_drone_separation_penalty: float = 15.0,
        render_mode=None,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.n_obstacles_range = n_obstacles_range
        self.random_obstacles = random_obstacles
        self.safety_radius = safety_radius
        self.obstacle_radius = obstacle_radius
        self.max_speed = max_speed
        self.formation_size = formation_size
        self.formation_coef = formation_coef
        self.momentum_alpha = momentum_alpha
        self.max_steps = max_steps
        self.obstacles_on_route = obstacles_on_route
        self.route_corridor_width = route_corridor_width
        self.proximity_threshold = proximity_threshold
        self.proximity_penalty_coef = proximity_penalty_coef
        self.min_drone_separation = min_drone_separation
        self.min_drone_separation_penalty = min_drone_separation_penalty
        self.render_mode = render_mode

        s = formation_size / 2
        self.formation_offsets = np.array([
            [-s, -s], [s, -s], [-s, s], [s, s],
        ], dtype=np.float32)

        total_obs = self.N_DRONES * self.OBS_DIM
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_obs,), dtype=np.float32
        )
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if self.random_obstacles:
            n_obs = int(self.np_random.integers(
                self.n_obstacles_range[0],
                self.n_obstacles_range[1] + 1
            ))
        else:
            n_obs = self.n_obstacles

        margin = 8.0
        start_center = self.np_random.uniform(margin, self.grid_size - margin, size=2).astype(np.float32)
        target = self.np_random.uniform(margin, self.grid_size - margin, size=2).astype(np.float32)
        while np.linalg.norm(target - start_center) < 15.0:
            target = self.np_random.uniform(margin, self.grid_size - margin, size=2).astype(np.float32)

        self.target = target
        self.positions = (start_center + self.formation_offsets).astype(np.float32)
        self.velocities = np.zeros((self.N_DRONES, 2), dtype=np.float32)
        self.obstacles = self._generate_obstacles(n_obs, start_center, target)
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def _check_collisions(self, drone_positions: np.ndarray) -> list:
        """Hangi drone'lar carpısma bolgesinde."""
        collisions = []
        for i, pos in enumerate(drone_positions):
            hit = False
            if (pos[0] <= self.safety_radius or pos[0] >= self.grid_size - self.safety_radius or
                    pos[1] <= self.safety_radius or pos[1] >= self.grid_size - self.safety_radius):
                hit = True
            if not hit and self.obstacles is not None and len(self.obstacles) > 0:
                for obs_pos in self.obstacles:
                    if np.linalg.norm(pos - obs_pos) < self.safety_radius + self.obstacle_radius:
                        hit = True
                        break
            if hit:
                collisions.append(i)
        return collisions

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).reshape(self.N_DRONES, self.ACT_DIM)
        desired_velocities = action * self.max_speed

        alpha = self.momentum_alpha

        collisions = []
        collision_positions = {}
        for i in range(self.N_DRONES):
            # Momentum ile yumusak hiz guncelleme (Hybrid2 ile ayni mantik)
            blended_vel = alpha * desired_velocities[i] + (1.0 - alpha) * self.velocities[i]
            new_pos = self.positions[i] + blended_vel
            new_pos, new_vel = self._apply_wall_sliding(new_pos.copy(), blended_vel.copy())

            all_new = self.positions.copy()
            all_new[i] = new_pos
            hit = self._check_collisions(all_new)

            if i in hit:
                self.velocities[i] = new_vel * 0.1  # carpısma: hiz sonumlenir
                collisions.append(i)
                collision_positions[i] = new_pos
            else:
                self.positions[i] = new_pos.astype(np.float32)
                self.velocities[i] = new_vel.astype(np.float32)

        reward, done, info = self._compute_reward(collisions, collision_positions)
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    def _get_obs(self):
        obs_list = []
        center = self.positions.mean(axis=0)
        for i in range(self.N_DRONES):
            pos_norm = self.positions[i] / self.grid_size
            vel_norm = self.velocities[i] / self.max_speed
            to_target = self.target - self.positions[i]
            target_norm = to_target / (self.grid_size * np.sqrt(2) + 1e-6)

            # DUZELTME: formation_err normalize olcegi tutarli
            # obs: ham_hata / grid_size  →  reward: ham_hata * formation_coef
            # Her ikisi de ayni fiziksel hatadan turetiliyor, olcekler tutarli
            ideal_pos = center + self.formation_offsets[i]
            formation_err = (self.positions[i] - ideal_pos) / self.grid_size

            rays = self._ray_distances_4dir(i)  # 4 yonlu ray

            drone_obs = np.concatenate([
                pos_norm,        # [0-1]
                vel_norm,        # [2-3]
                target_norm,     # [4-5]
                formation_err,   # [6-7]
                rays,            # [8-11]
            ]).astype(np.float32)
            obs_list.append(drone_obs)
        return np.concatenate(obs_list)

    def _ray_distances_4dir(self, drone_idx: int) -> np.ndarray:
        """
        4 yonlu ray: ileri, sag, geri, sol.
        'Ileri' hedefe dogru yon, diger uc buna gore donduruluyor.
        Hybrid2'deki 4-ray ile ayni kapsam (tum yonler).
        """
        pos = self.positions[drone_idx]
        to_target = self.target - pos
        dist = np.linalg.norm(to_target)
        if dist < 1e-6:
            forward = np.array([1.0, 0.0], dtype=np.float32)
        else:
            forward = (to_target / dist).astype(np.float32)

        right = np.array([ forward[1], -forward[0]], dtype=np.float32)
        back  = -forward
        left  = -right

        directions = [forward, right, back, left]
        max_ray = self.grid_size
        results = []

        for direction in directions:
            min_dist = max_ray

            # Duvar mesafesi
            for dim in range(2):
                if abs(direction[dim]) > 1e-6:
                    if direction[dim] > 0:
                        d = (self.grid_size - pos[dim]) / direction[dim]
                    else:
                        d = (0.0 - pos[dim]) / direction[dim]
                    min_dist = min(min_dist, max(0.0, float(d)))

            # Engel mesafesi
            if self.obstacles is not None and len(self.obstacles) > 0:
                for obs_pos in self.obstacles:
                    to_obs = obs_pos - pos
                    proj = np.dot(to_obs, direction)
                    if proj > 0:
                        perp = np.linalg.norm(to_obs - proj * direction)
                        if perp < self.obstacle_radius:
                            hit_dist = proj - np.sqrt(max(0.0, self.obstacle_radius**2 - perp**2))
                            min_dist = min(min_dist, max(0.0, float(hit_dist)))

            results.append(min_dist / max_ray)

        return np.array(results, dtype=np.float32)

    def _proximity_penalty(self) -> float:
        """Drone-drone yakınlık cezası: d < proximity_threshold ise coef * (thresh - d)."""
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
        """İki drone min_drone_separation altına inerse ek ceza (iç içe girmeyi azaltır)."""
        penalty = 0.0
        min_sep = self.min_drone_separation
        p_per_pair = self.min_drone_separation_penalty
        for i in range(self.N_DRONES):
            for j in range(i + 1, self.N_DRONES):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d < min_sep:
                    penalty += p_per_pair
        return penalty

    def _compute_reward(self, collisions: list = None, collision_positions: dict = None):
        if collisions is None:
            collisions = self._check_collisions(self.positions)
            collision_positions = {}
        if collision_positions is None:
            collision_positions = {}
        n_collisions = len(collisions)

        center = self.positions.mean(axis=0)
        dist_to_target = float(np.linalg.norm(center - self.target))
        reward = -0.01 * dist_to_target - 0.01
        done = False
        info = {}

        # Carpısma cezasi
        reward -= 10.0 * n_collisions
        if n_collisions >= 2:
            reward -= 20.0
            done = True
            info["termination"] = "heavy_collision"

        # Formasyon hatasi — ham deger, formation_coef ile olceklenir
        ideal_positions = center + self.formation_offsets
        formation_error = float(np.linalg.norm(self.positions - ideal_positions))
        reward -= self.formation_coef * formation_error

        # Drone-drone yakınlık cezaları (Hybrid ile aynı)
        reward -= self._proximity_penalty()
        reward -= self._min_separation_penalty()

        if dist_to_target < 3.0:
            reward += 1000.0
            done = True
            info["termination"] = "success"

        info["dist_to_goal"] = dist_to_target
        info["formation_error"] = formation_error
        info["n_collisions"] = n_collisions
        return reward, done, info

    def _apply_wall_sliding(
        self, pos: np.ndarray, vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Duvar sinirini asarsa konum sabitleniyor, o eksendeki hiz sifirlanıyor."""
        min_p, max_p = 3.0, self.grid_size - 3.0
        new_pos, new_vel = pos.copy(), vel.copy()
        for d in range(2):
            if new_pos[d] < min_p:
                new_pos[d], new_vel[d] = min_p, 0.0
            elif new_pos[d] > max_p:
                new_pos[d], new_vel[d] = max_p, 0.0
        return new_pos, new_vel

    def _generate_obstacles(self, n_obs, start_center, target):
        obstacles = []
        min_dist_from_spawn = self.obstacle_radius + self.safety_radius + self.formation_size + 2.0
        margin = self.obstacle_radius + 1.0

        if self.obstacles_on_route:
            # Engeller hedefe giden rotada: start-target arasi koridor
            seg = target - start_center
            dir_vec = seg / (float(np.linalg.norm(seg)) + 1e-6)
            perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)
            half_width = self.route_corridor_width / 2.0
            # Rota boyunca start ve target'tan uzak bolge (t: 0.15 - 0.85)
            t_lo, t_hi = 0.15, 0.85
        else:
            dir_vec = perp = None

        for _ in range(n_obs):
            for _ in range(200):
                if self.obstacles_on_route and dir_vec is not None:
                    t = float(self.np_random.uniform(t_lo, t_hi))
                    along = start_center + t * seg
                    off = float(self.np_random.uniform(-half_width, half_width))
                    pos = (along + off * perp).astype(np.float32)
                    pos = np.clip(pos, margin, self.grid_size - margin)
                else:
                    pos = self.np_random.uniform(margin, self.grid_size - margin, size=2).astype(np.float32)

                if (np.linalg.norm(pos - start_center) > min_dist_from_spawn and
                        np.linalg.norm(pos - target) > min_dist_from_spawn):
                    ok = True
                    for existing in obstacles:
                        if np.linalg.norm(pos - existing) < 2 * self.obstacle_radius + 1.0:
                            ok = False
                            break
                    if ok:
                        obstacles.append(pos)
                        break
        return np.array(obstacles, dtype=np.float32) if obstacles else np.zeros((0, 2), dtype=np.float32)

    def _render_pygame(self):
        try:
            import pygame
        except ImportError:
            return
        CELL = 12
        W = H = self.grid_size * CELL
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((int(W), int(H)))
            pygame.display.set_caption("Drone Swarm — Shared Policy (CTDE)")
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        self._screen.fill((20, 20, 30))
        if self.obstacles is not None:
            for obs_pos in self.obstacles:
                pygame.draw.circle(
                    self._screen, (180, 60, 60),
                    (int(obs_pos[0] * CELL), int((self.grid_size - obs_pos[1]) * CELL)),
                    int(self.obstacle_radius * CELL)
                )
        pygame.draw.circle(
            self._screen, (60, 220, 60),
            (int(self.target[0] * CELL), int((self.grid_size - self.target[1]) * CELL)),
            int(3.0 * CELL), 2
        )
        colors = [(100, 180, 255), (255, 200, 80), (180, 255, 120), (255, 120, 200)]
        for i, pos in enumerate(self.positions):
            cx, cy = int(pos[0] * CELL), int((self.grid_size - pos[1]) * CELL)
            pygame.draw.circle(self._screen, colors[i], (cx, cy), int(1.2 * CELL))
        center = self.positions.mean(axis=0)
        pygame.draw.circle(
            self._screen, (255, 255, 255),
            (int(center[0] * CELL), int((self.grid_size - center[1]) * CELL)), 3
        )
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def get_per_drone_obs(self):
        """Her drone'un lokal gozlemini ayri ayri dondurur (inference icin)."""
        full_obs = self._get_obs()
        return [full_obs[i * self.OBS_DIM:(i + 1) * self.OBS_DIM] for i in range(self.N_DRONES)]
