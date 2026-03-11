"""
Zorlu parkur engel editörü - tıklayarak engel ekle, config'e kaydet.

Simülasyonla aynı boyut (600x600). Sol tık: engel ekle, Sağ tık: son engeli sil.
S: hard_course_config.py dosyasına kaydet  |  ESC: çık
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)

from hard_course_config import HARD_START, HARD_TARGET

CELL = 12
GRID_SIZE = 50.0
OBSTACLE_RADIUS = 3.0  # Simülasyondaki sabit yarıçap
BG = (13, 17, 23)
GRID = (40, 50, 65)
TGT = (60, 220, 60)
TGT_O = (34, 221, 34)
START_COL = (255, 255, 100)
OBST_COL = (180, 60, 60)
TXT = (255, 255, 255)


def to_screen(pos, gs=GRID_SIZE):
    return int(pos[0] * CELL), int((gs - pos[1]) * CELL)


def to_grid(screen_x, screen_y, gs=GRID_SIZE):
    gx = screen_x / CELL
    gy = gs - screen_y / CELL
    return round(gx, 2), round(gy, 2)


def save_config(obstacles):
    path = os.path.join(os.path.dirname(__file__), "hard_course_config.py")
    lines = []
    lines.append('"""')
    lines.append("Zorlu parkur engel konumları - hard_course_editor.py ile düzenlendi.")
    lines.append("Grid 50x50. Start (10,10), Target (40,40) — min 7 birim uzak olmalı.")
    lines.append('"""')
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("HARD_START = np.array([10.0, 10.0], dtype=np.float32)")
    lines.append("HARD_TARGET = np.array([40.0, 40.0], dtype=np.float32)")
    lines.append("")
    lines.append("# Engel konumları [x, y]")
    lines.append("HARD_OBSTACLES = np.array([")
    for ox, oy in obstacles:
        lines.append(f"    [{ox:.2f}, {oy:.2f}],")
    lines.append("], dtype=np.float32)")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Kaydedildi: {path} ({len(obstacles)} engel)")


def main():
    obstacles = []  # Boş başla, tıklayarak ekle
    size = int(GRID_SIZE * CELL)  # 600

    pygame.init()
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Zorlu Parkur Editörü - Sol: ekle | Sağ: sil | S: kaydet | ESC: çık")
    font = pygame.font.Font(None, 22)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_s:
                    save_config(obstacles)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                sx, sy = pygame.mouse.get_pos()
                gx, gy = to_grid(sx, sy)
                if 0 <= gx <= GRID_SIZE and 0 <= gy <= GRID_SIZE:
                    if e.button == 1:  # Sol tık: ekle
                        obstacles.append([gx, gy])
                    elif e.button == 3 and obstacles:  # Sağ tık: sil
                        obstacles.pop()

        screen.fill(BG)
        for x in range(0, int(GRID_SIZE) + 1, 5):
            px = int(x * CELL)
            pygame.draw.line(screen, GRID, (px, 0), (px, size))
            pygame.draw.line(screen, GRID, (0, px), (size, px))

        # Hedef
        tx, ty = to_screen(HARD_TARGET)
        pygame.draw.circle(screen, TGT_O, (tx, ty), int(2.5 * CELL), 2)
        pygame.draw.circle(screen, TGT, (tx, ty), int(2.0 * CELL))

        # Başlangıç
        sx, sy = to_screen(HARD_START)
        pygame.draw.circle(screen, START_COL, (sx, sy), 6)

        # Engeller (sabit yarıçap = OBSTACLE_RADIUS)
        r = int(OBSTACLE_RADIUS * CELL)
        for ox, oy in obstacles:
            px, py = to_screen([ox, oy])
            pygame.draw.circle(screen, OBST_COL, (px, py), r)

        # Yardım metni
        txt = font.render("Sol tık: engel ekle | Sağ tık: sil | S: kaydet | ESC: çık", True, TXT)
        screen.blit(txt, (10, 10))
        cnt = font.render(f"Engel sayısı: {len(obstacles)}", True, TXT)
        screen.blit(cnt, (10, 32))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
