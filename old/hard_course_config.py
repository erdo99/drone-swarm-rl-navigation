"""
Zorlu parkur engel konumları - hard_course_editor.py ile düzenlendi.
Grid 50x50. Start/hedef min 7 birim uzak olmalı.
"""

import numpy as np

# Başlangıç: sol alt köşe (engellerden uzak)
HARD_START = np.array([5.0, 5.0], dtype=np.float32)
# Hedef: sağ üst köşe (engellerden uzak)
HARD_TARGET = np.array([45.0, 45.0], dtype=np.float32)

# Engel konumları [x, y]
HARD_OBSTACLES = np.array([
    [10.08, 20.92],
    [14.67, 35.08],
    [45.17, 30.17],
    [43.92, 11.50],
    [43.92, 21.08],
    [34.58, 9.17],
    [24.25, 9.08],
    [27.67, 24.83],
    [28.00, 40.00],
], dtype=np.float32)
