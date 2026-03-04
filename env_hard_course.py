"""
DroneSwarmEnvHybrid2HardCourse - Sabit zorlu parkur.

Engel konumları hybrid_2/hard_course_config.py dosyasından okunur.
O dosyada HARD_START, HARD_TARGET, HARD_OBSTACLES'ı düzenleyebilirsin.
"""

import numpy as np
from typing import Tuple, Dict, Optional

from env import DroneSwarmEnvHybrid2
from hard_course_config import HARD_START, HARD_TARGET, HARD_OBSTACLES


class DroneSwarmEnvHybrid2HardCourse(DroneSwarmEnvHybrid2):
    """Her reset'te aynı zorlu parkuru kullanır."""

    def __init__(self, swap_start_target: bool = False,
                 custom_start: Optional[np.ndarray] = None,
                 custom_target: Optional[np.ndarray] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.swap_start_target = swap_start_target
        self.custom_start = np.array(custom_start, dtype=np.float32) if custom_start is not None else None
        self.custom_target = np.array(custom_target, dtype=np.float32) if custom_target is not None else None

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

        # Başlangıç ve hedef: custom verilmişse onları kullan, yoksa config + swap
        if self.custom_start is not None and self.custom_target is not None:
            start = self.custom_start.copy()
            target = self.custom_target.copy()
        else:
            start = HARD_TARGET.copy() if self.swap_start_target else HARD_START.copy()
            target = HARD_START.copy() if self.swap_start_target else HARD_TARGET.copy()
        center = start
        self.drone_positions = center + self.FORMATION_OFFSETS
        self.target_pos = target
        self.obstacles = [o.copy() for o in HARD_OBSTACLES]

        return self._get_obs(), self._get_info()
