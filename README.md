# Drone Swarm RL Navigation

> **⚠️ Eski sürüm hakkında not:** Bu repodaki ilk `env.py` / `ppo_agent.py` / `train.py` / `visualize_pygame.py` dosyaları artık kullanılmamaktadır. Hibrit-2 mimarisine dayanan o sürüm çeşitli yeniden başlatma hatalarına ve gözlem boyutu uyumsuzluklarına yol açtığından production ortamında güvenilir biçimde çalışmamaktaydı. Aşağıdaki tüm belgeler yeni **Shared Policy (Parameter Sharing)** mimarisine aittir.

---

## Proje Tanıtımı

Bu proje, **PPO (Proximal Policy Optimization)** algoritması ile eğitilmiş dört droneden oluşan bir sürünün engellerden kaçınarak hedef noktaya navigasyonunu sağlar. Gymnasium ortamında tanımlanan 50×50 grid haritada drone sürüsü hem oluşum (formation) halinde kalır hem de bireysel hız komutlarıyla hareket eder. Algoritma seçimi (PPO vs MAPPO vb.), centralized vs multi-agent tercihi ve parametre gerekçeleri için [DEVLOG.md](DEVLOG.md) dosyasına bakınız.

**Özellikler:**
- **Shared Policy / Parameter Sharing:** 1 model, 4 drone; her drone kendi lokal gözlemini kullanır, ağırlıklar paylaşılır
- **Decentralized execution:** Her drone yalnızca kendi 12 boyutlu lokal gözlemine erişir
- **Rastgele başlangıç/hedef:** Her episode'da farklı konumlar; model tüm yönlerde genelleme öğrenir
- **Rota tabanlı engel yerleştirme:** Engellerin ayarlanabilir bir oranı start→target koridoruna yerleştirilir
- **Engel sayısı:** 5–9 arası rastgele veya sabit
- **Zorlu parkur:** Sabit engel yerleşimli test parkuru (`hard_course_config.py`)

### Demo Videoları

**Yeni sistem (Shared Policy — güncel):**

| Video | Açıklama |
|-------|----------|
| [test_hard_course_v2_5ep_8fps.mp4](videos/shared/test_hard_course_v2_5ep_8fps.mp4) | Zorlu parkur, 5 episode, 8 FPS — Shared Policy modeli — `test_hard_course_pygame_v2.py` |
| [visualize_random_v2_35ep_10fps.mp4](videos/shared/visualize_random_v2_35ep_10fps.mp4) | Rastgele parkur, 35 episode, 10 FPS — `visualize_pygame_v2.py` |

**Eski sistem (Hybrid2 — yalnızca referans amaçlı):**

| Video | Açıklama |
|-------|----------|
| [test_hard_course_5ep_8fps.mp4](videos/legacy/test_hard_course_5ep_8fps.mp4) | Zorlu parkur, 5 episode, 8 FPS — Hybrid2 modeli — `test_hard_course_pygame.py` |
| [visualize_random_5ep_10fps.mp4](videos/legacy/visualize_random_5ep_10fps.mp4) | Rastgele parkur, 5 episode, 10 FPS — `visualize_pygame.py` |

---

## Sinir Ağı ve Parametreler

### Ağ Mimarisi

- **Policy:** MLP (çok katmanlı perceptron), Stable-Baselines3 `MlpPolicy`
- **Gizli katmanlar:** `[256, 256]` — 2 katman, her birinde 256 nöron
- **Giriş:** 64 boyutlu observation vektörü (4 drone × 16 lokal obs; VecNormalize ile normalize)
- **Çıkış:** 8 boyutlu aksiyon vektörü (4 drone × 2 hız) + value fonksiyonu
- **Mimari:** 1 model, ağırlıklar 4 drone arasında paylaşılır (parameter sharing)

### Gözlem Uzayı (Observation — 64 boyut = 4 × 16)

Her drone için 16 boyutlu lokal gözlem:

| Boyut | Açıklama |
|-------|----------|
| 0–1 | Drone pozisyonu (x, y) — grid_size ile normalize [0, 1] |
| 2–3 | Drone hızı (vx, vy) — max_speed ile normalize [-1, 1] |
| 4–5 | Hedefe göre relative vektör (dx, dy) — drone → hedef, normalize |
| 6–7 | Formation hatası (fx, fy) — ideal formasyondan sapma (normalize) |
| 8–15 | Engel ray mesafeleri — 8 sabit yön (0°,45°,90°,...,315°); en yakın engel/duvar mesafesi [0, 1] |

### Aksiyon Uzayı (Action — 8 boyut = 4 × 2)

Her drone için 2 boyutlu hız komutu:

| Boyut | Açıklama |
|-------|----------|
| 2i, 2i+1 (i=0..3) | Drone i hızı (vx, vy) — [-1, 1] × max_speed |

### Ortam Parametreleri (`env_shared_v3.py`)

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `grid_size` | 50.0 | Harita boyutu (50×50) |
| `n_obstacles` | 5 | Engel sayısı (sabit mod) |
| `n_obstacles_range` | (5, 9) | Engel sayısı aralığı (rastgele mod) |
| `random_obstacles` | True | Her episode rastgele engel sayısı |
| `obstacles_on_route` | True | Engellerin bir kısmı start→target koridoruna yerleştirilsin |
| `route_obstacle_ratio` | 0.6 | Engellerin ne kadarı rotada olsun (0–1) |
| `safety_radius` | 2.0 | Çarpışma algılama yarıçapı |
| `obstacle_radius` | 3.0 | Engel yarıçapı |
| `max_speed` | 2.0 | Maksimum hız |
| `formation_size` | 4.0 | Kare formasyonun kenar uzunluğu |
| `formation_coef` | 0.3 | Formation hatası ceza katsayısı |
| `momentum_alpha` | 0.7 | Hız yumuşatma katsayısı |
| `max_steps` | 500 | Episode maksimum adım sayısı |
| `min_start_target_dist` | 15.0 | Start-target arası minimum mesafe |

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

### Ödül Fonksiyonu (Reward)

Her adımda ödül aşağıdaki bileşenlerden oluşur:

- **Hedefe mesafe cezası:** `-0.01 * dist(center, target)`  
  Sürü merkezi hedefe yaklaştıkça ceza azalır, uzaklaştıkça artar.
- **Çarpışma cezası:** `-10.0 * (#collisions)`  
  Her drone–engel veya drone–duvar çarpışması için ek ceza.
- **Zaman cezası:** `-0.01`  
  Her adımda küçük negatif ödül; daha kısa sürede hedefe gitmeye teşvik eder.
- **Formation cezası:** `-formation_coef * ||positions - ideal_positions||`  
  Sürü ideal kare formasyonundan ne kadar saparsa ceza artar (varsayılan `formation_coef=0.3`).
- **Drone-drone yakınlık cezası:** `-proximity_penalty_coef * (threshold - dist)` (eşik altındaki çiftler için)
- **İç içe girme cezası:** `-min_drone_separation_penalty` (min_drone_separation altına girilirse)
- **Başarı ödülü:** `+1000.0`  
  Başarı koşulu v4: hem sürü merkezi hedefe **3.0**'dan yakın (`dist(center,target) < 3.0`)  
  **ve** her drone hedefe **5.0**'dan yakın (`max_i dist(drone_i, target) < 5.0`) olmalıdır.
- **Ağır çarpışma cezası:** `-20.0` (aynı anda ≥2 drone çarpışırsa, episode sonlanır)

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
python train_v2.py --timesteps 2000000
```

İsteğe bağlı parametreler:
- `--n_obstacles_range "7,9"` — Engel sayısını her episode 7–9 arası rastgele yapar (varsayılan `5,9`)
- `--no_random` — Engel sayısını sabit tutar (`--n_obstacles` ile belirlenir)
- `--no_obstacles_on_route` — Engelleri rotaya değil, grid'e rastgele yerleştirir
- `--route_obstacle_ratio 0.6` — Engellerin ne kadarının rotada olacağını ayarlar (varsayılan `0.6`)
- `--ent_coef 0.02` — Entropi katsayısı (varsayılan `0.02`)
- `--save_dir`, `--log_dir` — Model ve log klasörleri
- `--n_envs 4` — Paralel ortam sayısı

Eğitim sırasında ilerlemeyi izlemek için ayrı bir terminalde:

```bash
tensorboard --logdir ./logs_shared/tensorboard/
```

Tarayıcıda `http://localhost:6006` adresini açın.

### 5. Görselleştirme ile Test

**Rastgele parkur (eğitim ortamına benzer):**

```bash
python visualize_pygame_v2.py --model models_shared/best/best_model
```

`--n_episodes 5 --fps 10` ile farklı değerler verilebilir. Eski davranışı (engeller rotaya değil grid'e rastgele) kullanmak için `--old_env` parametresi eklenebilir.

**Zorlu sabit parkur (`hard_course_config.py`):**

```bash
python test_hard_course_pygame_v2.py --model models_shared/best/best_model --n_episodes 5
python test_hard_course_pygame_v2.py --model models_shared/best/best_model --swap   # Start/hedef yer değiştir
```

Kendi sabit parkurunuzu (engel konumları + start/target) tanımlamak için `hard_course_editor.py` script'ini kullanabilirsiniz:

```bash
python hard_course_editor.py
```

- Sol tık: yeni engel ekler  
- Sağ tık: son eklenen engeli siler  
- `S` tuşu: güncel engel konumlarını ve start/target bilgisini `hard_course_config.py` içine yazar  
- `ESC`: çıkış

`hard_course_config.py` güncellendikten sonra, aynı **hazır modeli** yeni parkurunuzda test etmek için tekrar:

```bash
python test_hard_course_pygame_v2.py --model models_shared/best/best_model --n_episodes 5 --fps 8
```

komutunu çalıştırmanız yeterlidir.

### 6. Değerlendirme (Headless)

```bash
python evaluate.py --model models_shared/best/best_model --n_episodes 20
```

---

## Hazır Modelleri Başkalarının Kullanması

Repoda, eğitimden elde edilmiş hazır modeller paylaşılmaktadır:

- `models_shared/best/best_model.zip` — En iyi değerlendirme skoruna sahip model
- `models_shared/vec_normalize.pkl` — Observation/reward normalizasyon istatistikleri

Bu modeli kendi bilgisayarınızda test etmek için:

1. Depoyu klonlayın ve bağımlılıkları kurun:

```bash
git clone https://github.com/erdo99/drone-swarm-rl-navigation.git
cd drone-swarm-rl-navigation
pip install -r requirements.txt
```

2. Rastgele parkurda hazır modeli görselleştirin:

```bash
python visualize_pygame_v2.py --model models_shared/best/best_model --n_episodes 10 --fps 8
```

3. Zorlu sabit parkurda test edin:

```bash
python test_hard_course_pygame_v2.py --model models_shared/best/best_model --n_episodes 10 --fps 8
```

---

## Model ve Log Konumları

| Konum | Açıklama |
|-------|----------|
| `models_shared/best/best_model.zip` | Değerlendirme metriklerine göre en iyi skor üreten checkpoint |
| `models_shared/ppo_shared_final.zip` | Eğitimin sonundaki model |
| `models_shared/vec_normalize.pkl` | Observation/reward normalizasyon istatistikleri |
| `models_shared/checkpoints/` | Ara checkpoint'ler |
| `logs_shared/tensorboard/` | TensorBoard logları |
| `logs_shared/eval/` | EvalCallback değerlendirme logları |

---

## Geliştirme Notları

Bu projeyi genişletmek isteyenler için bazı ipuçları:

- **Yeni ödül fonksiyonu denemek:** `env_shared_v3.py` içindeki `_compute_reward` fonksiyonunda mesafe, çarpışma ve formation bileşenleri açıkça ayrılmıştır. Yeni ödül terimleri eklemek veya katsayıları değiştirmek için bu fonksiyonu düzenleyebilirsiniz.
- **Observation'a yeni sensörler eklemek:** `env_shared_v3.py` içindeki `_get_obs`, `OBS_DIM` sabiti ve `observation_space` tanımı birlikte güncellenmelidir. `OBS_DIM` değiştiğinde `ppo_agent_v2.py`'de ayrıca bir şey yapmak gerekmez; SB3 boyutu ortamdan otomatik alır.
- **Ortam zorluğunu değiştirmek:** Engel sayısı ve dağılımı `n_obstacles`, `n_obstacles_range`, `obstacles_on_route` ve `route_obstacle_ratio` parametreleriyle kontrol edilir. Hard course için sabit layout `hard_course_config.py` dosyasındadır.
- **Hyperparameter denemeleri:** PPO ayarları (learning rate, batch size, n_steps, ent_coef vb.) `ppo_agent_v2.py` içindeki `build_agent` fonksiyonunda ve `train_v2.py`'nin CLI argümanlarında tek bir yerde toplanmıştır.

Bağımlılıkları tek komutla kurmak için:

```bash
pip install -r requirements.txt
```# Drone Swarm RL Navigation

>