# Drone Swarm RL Navigation

## Proje Tanıtımı

Bu proje, **PPO (Proximal Policy Optimization)** algoritması ile eğitilmiş dört droneden oluşan bir sürünün, engellerden kaçınarak hedef noktaya navigasyonunu sağlar. Gymnasium ortamında tanımlanan 50×50 grid haritada drone sürüsü hem oluşum (formation) halinde kalır hem de ortak hız + bireysel offset ile hareket eder.

**Özellikler:**
- **Hybrid2 aksiyon yapısı:** 10 boyutlu aksiyon (ortak hız + per-drone offset)
- **Rastgele başlangıç/hedef:** Her episode'da farklı konumlar; model tüm yönlerde genelleme öğrenir
- **Engel sayısı:** 5–9 arası rastgele veya sabit
- **Zorlu parkur:** Sabit engel yerleşimli test parkuru (`hard_course_config.py`)

### Demo Videoları

| Video | Açıklama |
|-------|----------|
| [test_hard_course_5ep_8fps.mp4](videos/test_hard_course_5ep_8fps.mp4) | Zorlu parkur, 5 episode, 8 FPS — `test_hard_course_pygame.py` |
| [visualize_random_5ep_10fps.mp4](videos/visualize_random_5ep_10fps.mp4) | Rastgele parkur, 5 episode, 10 FPS — `visualize_pygame.py` |

---

## Sinir Ağı ve Parametreler

### Ağ Mimarisi

- **Policy:** MLP (çok katmanlı perceptron), Stable-Baselines3 `MlpPolicy`
- **Gizli katmanlar:** `[256, 256]` — 2 katman, her birinde 256 nöron
- **Giriş:** 30 boyutlu observation vektörü (VecNormalize ile normalize)
- **Çıkış:** 10 boyutlu aksiyon vektörü (policy) + value fonksiyonu

### Gözlem Uzayı (Observation — 30 boyut)

| Boyut | Açıklama |
|-------|----------|
| 0–1 | Sürü merkezi pozisyonu (x, y) — grid_size ile normalize [0, 1] |
| 2–3 | Sürü ortalama hızı (vx, vy) — max_speed ile normalize [-1, 1] |
| 4–5 | Hedefe göre relative vektör (dx, dy) — merkez → hedef, normalize |
| 6–13 | Formation hataları — 4 drone × 2 boyut, ideal pozisyondan sapma (normalize) |
| 14–29 | Engel ray mesafeleri — 4 drone × 4 ray; her drone için 0°, 90°, 180°, 270° yönünde en yakın engel/duvar mesafesi [0, 1] |

### Aksiyon Uzayı (Action — 10 boyut)

| Boyut | Açıklama |
|-------|----------|
| 0–1 | Ortak hız (vx, vy) — tüm sürü için geçerli, [-1, 1] × max_speed |
| 2–3 | Drone 0 offset (vx, vy) — ortak hıza eklenen bireysel düzeltme × offset_scale |
| 4–5 | Drone 1 offset |
| 6–7 | Drone 2 offset |
| 8–9 | Drone 3 offset |

### Ortam Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `grid_size` | 50.0 | Harita boyutu (50×50) |
| `n_obstacles` | 5 | Engel sayısı (sabit mod) |
| `n_obstacles_range` | (5, 9) | Engel sayısı aralığı (rastgele mod) |
| `safety_radius` | 2.0 | Çarpışma algılama yarıçapı |
| `obstacle_radius` | 3.0 | Engel yarıçapı |
| `max_speed` | 2.0 | Maksimum hız |
| `offset_scale` | 0.6 | Per-drone offset katsayısı |
| `formation_coef` | 0.3 | Formation hatası ceza katsayısı |
| `max_steps` | 500 | Episode maksimum adım sayısı |

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
| Entropy coef | 0.01 |
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
- **Başarı ödülü:** `+1000.0` (hedefe varınca, `dist < 3.0`)  
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
python visualize_pygame.py --model models_hybrid_2/best/best_model
```
Varsayılan: 100 episode, 5 FPS. `--n_episodes 5 --fps 10` ile farklı değerler verilebilir.

**Zorlu sabit parkur (hard_course_config.py):**

```bash
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --no_swap   # Normal yön (sol alt → sağ üst)
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
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5 --fps 8
```

komutunu çalıştırmanız yeterlidir.

### 6. Değerlendirme (Headless)

```bash
python evaluate.py --model models_hybrid_2/best/best_model --n_episodes 20
```

---

## Hazır Modelleri Başkalarının Kullanması

Repoda, eğitimden elde edilmiş hazır modeller paylaşılmaktadır:

- `models_hybrid_2/best/best_model.zip` — En iyi değerlendirme skoruna sahip model
- `models_hybrid_2/vec_normalize.pkl` — Observation/reward normalizasyon istatistikleri

Bu modeli kendi bilgisayarınızda test etmek için:

1. Depoyu klonlayın ve bağımlılıkları kurun:

```bash
git clone https://github.com/erdo99/drone-swarm-rl-navigation.git
cd drone-swarm-rl-navigation
pip install -r requirements.txt
```

2. Rastgele parkurda hazır modeli görselleştirin:

```bash
python visualize_pygame.py --model models_hybrid_2/best/best_model --n_episodes 10 --fps 8
```

3. Zorlu sabit parkurda test edin:

```bash
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --n_episodes 10 --fps 8
```

İsteğe bağlı olarak, zorlu parkurda başlangıç/hedef noktalarını terminalden parametre vererek özelleştirebilirsiniz:

```bash
python test_hard_course_pygame.py ^
  --model models_hybrid_2/best/best_model ^
  --start 10,10 --target 40,40 --n_episodes 5 --fps 8
```

Bu komut, `hard_course_config.py`’deki sabit layout’u korurken sadece start/target koordinatlarını değiştirir; böylece farklı senaryolarda aynı eğitilmiş modeli deneyebilirsiniz.

---

## Model ve Log Konumları

| Konum | Açıklama |
|-------|----------|
| `models_hybrid_2/best/best_model.zip` | Değerlendirme metriklerine göre en iyi skor üreten checkpoint |
| `models_hybrid_2/ppo_hybrid_2_final.zip` | Eğitimin sonundaki model (pratikte daha dengeli davranış; testlerde bunu kullanmanız önerilir) |
| `models_hybrid_2/vec_normalize.pkl` | Observation/reward normalizasyon istatistikleri |
| `models_hybrid_2/checkpoints/` | Ara checkpoint’ler |
| `logs_hybrid_2/tensorboard/` | TensorBoard logları |

---

## Geliştirme Notları

Bu projeyi genişletmek isteyenler için bazı ipuçları:

- **Yeni ödül fonksiyonu denemek:** `env.py` içindeki `_compute_reward` fonksiyonunda mesafe, çarpışma ve formation bileşenleri açıkça ayrılmıştır. Yeni ödül terimleri eklemek veya katsayıları değiştirmek için bu fonksiyonu düzenleyebilirsiniz.
- **Observation’a yeni sensörler eklemek:** `env.py` içindeki `_get_obs` ve `observation_space` tanımı (obs_dim) birlikte güncellenmelidir. Örneğin daha fazla ray eklemek isterseniz `_obstacle_ray_distances` ve rays concat kısmını büyütmeniz gerekir.
- **Ortam zorluğunu değiştirmek:** Engel sayısı ve dağılımı `n_obstacles`, `n_obstacles_range` ve `_generate_obstacles` üzerinden kontrol edilir. Hard course için sabit layout `hard_course_config.py` dosyasındadır.
- **Hyperparameter denemeleri:** PPO ayarları (learning rate, batch size, n_steps, vs.) `ppo_agent.py` içindeki `build_ppo_agent` fonksiyonunda tek bir yerde toplanmıştır; farklı konfigürasyonlar için burayı değiştirmek veya CLI parametreleri eklemek yeterlidir.

Bağımlılıkları tek komutla kurmak için:

```bash
pip install -r requirements.txt
```
