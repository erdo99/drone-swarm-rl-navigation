# Drone Swarm RL Navigation

## Proje Tanıtımı

Bu proje, **PPO (Proximal Policy Optimization)** algoritması ile eğitilmiş dört droneden oluşan bir sürünün, engellerden kaçınarak hedef noktaya navigasyonunu sağlar. Gymnasium ortamında tanımlanan 50×50 grid haritada drone sürüsü hem oluşum (formation) halinde kalır hem de ortak hız + bireysel offset ile hareket eder.

**Özellikler:**
- **Hybrid2 aksiyon yapısı:** 10 boyutlu aksiyon (ortak hız + per-drone offset)
- **Rastgele başlangıç/hedef:** Her episode’da farklı konumlar; model tüm yönlerde genelleme öğrenir
- **Engel sayısı:** 5–9 arası rastgele veya sabit
- **Zorlu parkur:** Sabit engel yerleşimli test parkuru (`hard_course_config.py`)

---

## Sistemi Çalıştırmak İçin Gereken Adımlar

### 1. Gereksinimler

- Python 3.9+
- pip

### 2. Depoyu Klonlama

```bash
git clone https://github.com/erdo99/drone-swarm-rl-navigation.git
cd drone-swarm-rl-navigation
```

### 3. Bağımlılıkları Yükleme

```bash
pip install stable-baselines3 gymnasium numpy pygame
```

### 4. Model Eğitimi

Eğitilmiş bir model yoksa önce eğitim yapmanız gerekir:

```bash
python train.py --timesteps 2000000
```

İsteğe bağlı parametreler:
- `--n_obstacles_range "5,9"` — Engel sayısını her episode 5–9 arası rastgele yapar (varsayılan)
- `--n_obstacles_range ""` — Engel sayısını sabit tutar (`--n_obstacles` ile belirlenir)
- `--save_dir`, `--log_dir` — Model ve log klasörleri

Eğitim sırasında ilerlemeyi izlemek için ayrı bir terminalde:

```bash
tensorboard --logdir ./logs_hybrid_2/tensorboard/
```

Tarayıcıda `http://localhost:6006` adresini açın.

### 5. Görselleştirme ile Test

**Rastgele parkur (eğitim ortamına benzer):**

```bash
python visualize_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5 --fps 10
```

**Zorlu sabit parkur (hard_course_config.py):**

```bash
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --no_swap   # Normal yön (sol alt → sağ üst)
```

### 6. Değerlendirme (Headless)

```bash
python evaluate.py --model models_hybrid_2/best/best_model --n_episodes 20
```

---

## Model ve Log Konumları

| Konum | Açıklama |
|-------|----------|
| `models_hybrid_2/best/best_model.zip` | En iyi değerlendirme skoruna sahip model |
| `models_hybrid_2/ppo_hybrid_2_final.zip` | Eğitimin sonundaki model |
| `models_hybrid_2/vec_normalize.pkl` | Observation/reward normalizasyon istatistikleri |
| `models_hybrid_2/checkpoints/` | Ara checkpoint’ler |
| `logs_hybrid_2/tensorboard/` | TensorBoard logları |
