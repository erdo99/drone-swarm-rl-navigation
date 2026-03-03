# TESTING.md — Test ve Doğrulama

Sistemin test senaryoları, sınırları ve sonuçları.

---

## Test Ortamları

### 1. Rastgele parkur (eğitim ortamı)

`visualize_pygame.py` ve `evaluate.py` — her episode rastgele başlangıç/hedef, rastgele engel sayısı (5–9).

**Kullanım:**
```bash
python visualize_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5 --fps 10
python evaluate.py --model models_hybrid_2/best/best_model --n_episodes 20
```

### 2. Zorlu sabit parkur (hard course)

`test_hard_course_pygame.py` ve `test_hard_course.py` — `hard_course_config.py` içindeki sabit engel yerleşimi.

**Kullanım:**
```bash
# Swap mod: start=sağ üst, hedef=sol alt
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --n_episodes 5

# Normal mod: start=sol alt, hedef=sağ üst
python test_hard_course_pygame.py --model models_hybrid_2/best/best_model --no_swap
```

---

## Engel Konfigürasyonları

### Sabit engel sayısı

`--n_obstacles N` ile test:

```bash
python evaluate.py --model models_hybrid_2/best/best_model --n_obstacles 3 --n_episodes 20
python evaluate.py --model models_hybrid_2/best/best_model --n_obstacles 7 --n_episodes 20
```

| Engel sayısı | Beklenen | Not |
|--------------|----------|-----|
| 2–3         | Yüksek başarı | Eğitimden daha az engel |
| 5–9         | Orta–yüksek   | Eğitim aralığında |
| 10+         | Düşük         | Eğitimde görülmemiş |

### Zorlu parkur (hard_course_config.py)

9 sabit engel, belirli pozisyonlarda. Dar geçitler ve köşe manevraları içerir.

---

## Wall Sliding Açık/Kapalı Karşılaştırması

### Wall sliding açık (varsayılan)

- Duvar/duvara temas edildiğinde hız duvar yönünde sıfırlanır, diğer yönde hareket devam eder.
- Köşelerde kayarak dönüş mümkün.

**Test:**
```bash
python train.py --timesteps 500000   # wall_sliding=True (varsayılan)
python evaluate.py --model ...       # ortam wall_sliding=True
```

### Wall sliding kapalı

- Pozisyon `clip(3, 47)` ile sınırlanır; duvara çarpınca duvarda kalır.

**Test:**
```bash
python train.py --timesteps 500000 --no_wall_sliding
# evaluate.py wall_sliding parametresini desteklemiyorsa, env.py'de değişiklik gerekir
```

**Gözlem:** Wall sliding kapalıyken duvara yapışma daha sık; özellikle köşelerde takılma artar. Açık mod daha akıcı navigasyon sağlar.

---

## Test Senaryoları Özeti

| Senaryo              | Script                     | Engel     | Start/hedef |
|----------------------|----------------------------|-----------|-------------|
| Rastgele, headless   | `evaluate.py`              | 5 (veya N)| Rastgele    |
| Rastgele, görsel     | `visualize_pygame.py`      | 5–9       | Rastgele    |
| Hard course, swap    | `test_hard_course_pygame.py` | 9 sabit | Sağ üst → sol alt |
| Hard course, normal  | `test_hard_course_pygame.py --no_swap` | 9 sabit | Sol alt → sağ üst |

---

## Sistem Sınırları

- **Engel yoğunluğu:** 10+ engelde performans belirgin düşer (eğitimde 5–9 kullanıldı).
- **Grid boyutu:** 50×50 sabit; farklı boyutlar için yeniden eğitim gerekir.
- **Drone sayısı:** 4 sabit; farklı sürü boyutu için ortam ve model değişikliği gerekir.
- **VecNormalize:** Model yüklenirken `vec_normalize.pkl` kullanılmalı; aksi halde observation ölçeği uyumsuz olur.
