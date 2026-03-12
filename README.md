# Drone Swarm RL Navigation

## Proje Tanıtımı

Bu proje, **PPO (Proximal Policy Optimization)** ile eğitilmiş dört droneden oluşan bir sürünün engellerden kaçınarak hedef noktaya navigasyonunu sağlar. Gymnasium ortamında 50×50 grid, **Shared Policy (CTDE)** mimarisi ve **route corridor** engel yerleşimi kullanılır. Algoritma seçimi ve parametre gerekçeleri için [DEVLOG.md](DEVLOG.md) dosyasına bakınız.

**V1 (Centralized Hybrid2):** V1 sürümü ve dokümantasyonu `old/` klasöründe bulunmaktadır; detaylar için [old/README.md](old/README.md) dosyasına bakabilirsiniz.

**Özellikler:**
- **Shared Policy (CTDE):** 4 drone × 12 lokal obs (48), 4 × 2 hız (8 act), parameter sharing
- **Route corridor:** Engellerin bir kısmı start→target rotası boyunca yerleştirilir (`route_obstacle_ratio`)
- **Rastgele başlangıç/hedef:** Her episode farklı konumlar; tüm yönlerde genelleme
- **Engel sayısı:** 5–9 arası rastgele

### Demo Videoları

| Video | Açıklama |
|-------|----------|
| [test_hard_course_5ep_8fps.mp4](videos/test_hard_course_5ep_8fps.mp4) | Zorlu parkur, 5 episode |
| [visualize_random_5ep_10fps.mp4](videos/visualize_random_5ep_10fps.mp4) | Rastgele parkur, 5 episode |

---

## Sinir Ağı ve Parametreler

### Ağ Mimarisi

- **Policy:** MLP, Stable-Baselines3 `MlpPolicy`
- **Gizli katmanlar:** `[256, 256]`
- **Giriş:** 48 boyutlu observation (4 drone × 12 lokal, VecNormalize)
- **Çıkış:** 8 boyutlu aksiyon (4 drone × 2 hız) + value fonksiyonu

### Gözlem Uzayı (48 boyut, per drone 12)

Her drone için: pos(2) + vel(2) + to_target(2) + formation_err(2) + 4 ray = 12. Toplam 4×12 = 48.

### Aksiyon Uzayı (8 boyut)

Her drone için 2 boyut (vx, vy). Toplam 4×2 = 8.

### Ortam Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `grid_size` | 50.0 | Harita boyutu |
| `n_obstacles_range` | (5, 9) | Engel sayısı aralığı |
| `obstacles_on_route` | True | Engeller rotada yerleşir |
| `route_obstacle_ratio` | 0.6 | Engelin kaçı rotada olsun (0–1) |
| `formation_coef` | 0.3 | Formation hatası ceza katsayısı |
| `max_steps` | 500 | Episode maksimum adım |

### PPO Hiperparametreleri

| Parametre | Değer |
|-----------|-------|
| Learning rate | 3e-4 |
| n_steps | 2048 |
| Batch size | 256 |
| n_epochs | 10 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.02 |
| Value function coef | 0.5 |

### Ödül Fonksiyonu

- Hedefe mesafe cezası, çarpışma cezası, formation cezası, proximity/min_separation cezaları
- Başarı ödülü +1000, ağır çarpışma episode sonu

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
pip install -r requirements.txt
```

veya:

```bash
pip install stable-baselines3 gymnasium numpy pygame matplotlib
```

### 4. Model Eğitimi

```bash
python train_v2.py --timesteps 2000000 --n_obstacles_range 5,9
```

İsteğe bağlı: `--route_obstacle_ratio 0.6`, `--ent_coef 0.02`, `--no_obstacles_on_route`

TensorBoard ile izleme:

```bash
tensorboard --logdir ./logs_shared --port 6006
```

Tarayıcıda `http://localhost:6006` adresini açın.

### 5. Görselleştirme ve Test

`train_v2` ile eğitilen modeller `models_shared/best/` altındadır. Test için `shared/` altındaki ilgili script'ler kullanılabilir (örn. `shared/visualize_pygame.py`, `shared/test_shared_hard_course_pygame.py`).

### 6. Engel Yerleşimi Karşılaştırması

Farklı ortam versiyonlarının engel dağılımlarını karşılaştırmak için:

```bash
python shared/render_env_comparison.py --seed 42 --n_collection 30
```

Çıktılar `shared/env_comparison_output/` klasörüne kaydedilir.

---

## Hazır Modelleri Başkalarının Kullanması

- `models_shared/best/best_model.zip` — env_shared_v3 ile eğitilmiş en iyi model
- `models_shared/vec_normalize.pkl` — VecNormalize istatistikleri

1. Depoyu klonlayın ve bağımlılıkları kurun.
2. `shared/` altındaki test/görselleştirme script'leri ile modeli yükleyip çalıştırın.

(Auto-version: `models_shared.1`, `logs_shared.1` gibi alt klasörler oluşabilir.)

---

## Model ve Log Konumları

| Konum | Açıklama |
|-------|----------|
| `models_shared/best/best_model.zip` | En iyi model |
| `models_shared/vec_normalize.pkl` | VecNormalize istatistikleri |
| `models_shared/checkpoints/` | Ara checkpoint'ler |
| `logs_shared/tensorboard/` | TensorBoard logları |

---

## Geliştirme Notları

- **Ortam:** `env_shared_v3.py` — `route_obstacle_ratio`, `obstacles_on_route` ile engel yerleşimi
- **Engel karşılaştırması:** `shared/render_env_comparison.py`
- **Hyperparameter:** `shared/ppo_agent_shared.py`
