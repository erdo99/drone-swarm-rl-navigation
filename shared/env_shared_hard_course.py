"""
Shared hard course env — sabit zorlu parkur (hard_course_config.py) + parameter sharing.

env_shared_v3 ile aynı observation/action yapısı (48 obs, 8 act) — train_v2 ile eğitilen
model bu ortamda test edilebilir.

Fark: Start/target ve engeller rastgele değil, hard_course_config.py'deki
HARD_START, HARD_TARGET ve HARD_OBSTACLES kullanılır.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from typing import Optional, Dict, Tuple

from hard_course_config import HARD_START, HARD_TARGET, HARD_OBSTACLES
from env_shared_v3 import DroneSwarmSharedEnv


class DroneSwarmSharedHardCourseEnv(DroneSwarmSharedEnv):
    """
    Sabit hard course (config'ten) + shared policy ortamı.

    reset() her seferinde aynı layout'u kurar:
      - swap_start_target=False: start=HARD_START, target=HARD_TARGET
      - swap_start_target=True : start=HARD_TARGET, target=HARD_START
    random_swap=True ise her episode'ta swap rastgele seçilir.
    """

    def __init__(
        self,
        swap_start_target: bool = False,
        random_swap: bool = False,
        **kwargs,
    ):
        # random_obstacles kapalı; engeller config'ten
        # Çağıran random_obstacles göndermiş olsa bile yok say.
        kwargs.pop("random_obstacles", None)
        super().__init__(random_obstacles=False, **kwargs)
        self.swap_start_target = swap_start_target
        self.random_swap = random_swap

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        # RNG
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0

        # Swap kararı
        swap = (self.np_random.random() < 0.5) if self.random_swap else self.swap_start_target

        start_center = HARD_TARGET.copy() if swap else HARD_START.copy()
        target = HARD_START.copy() if swap else HARD_TARGET.copy()

        self.target = target.astype(np.float32)
        self.positions = (start_center + self.formation_offsets).astype(np.float32)
        self.velocities = np.zeros((self.N_DRONES, 2), dtype=np.float32)
        self.obstacles = HARD_OBSTACLES.astype(np.float32)

        obs = self._get_obs()
        info = {"dist_to_goal": float(np.linalg.norm(self.positions.mean(axis=0) - self.target))}
        return obs, info

