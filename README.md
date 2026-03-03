# Drone Swarm RL Navigation

PPO tabanlı drone sürü navigasyonu — rastgele başlangıç/hedef, engel kaçınma, formation kontrolü.

## Özellikler

- **Hybrid2:** Ortak hız + per-drone offset (10-dim action)
- **Rastgele başlangıç/hedef:** Her episode farklı konumlarda
- **Engel sayısı:** 5–9 arası rastgele (veya sabit)
- **Zorlu parkur:** `hard_course_config.py` ile sabit test parkuru

## Kurulum

```bash
pip install stable-baselines3 gymnasium numpy pygame
```

## Eğitim

```bash
python train.py --timesteps 2000000
python train.py --timesteps 2000000 --n_obstacles_range "5,9"   # Engel sayısı 5-9 arası
python train.py --timesteps 500000 --n_obstacles 5 --n_obstacles_range ""  # Sabit 5 engel
```

TensorBoard:
```bash
tensorboard --logdir ./logs_hybrid_2/tensorboard/
```

## Test & Görselleştirme

```bash
# Rastgele parkur
python visualize_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5 --fps 10

# Zorlu sabit parkur
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --no_swap  # Normal yön
```

## Model Kayıtları

- `models_hybrid_2/best/best_model.zip` — En iyi model
- `models_hybrid_2/ppo_hybrid_2_final.zip` — Son model
- `models_hybrid_2/vec_normalize.pkl` — VecNormalize istatistikleri
