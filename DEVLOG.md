# DEVLOG.md — Geliştirme Süreci Kaydı

Bu dosya projenin geliştirme sürecini kronolojik olarak belgeler. Kararlar, denemeler ve öğrenilenler.

---

## Başta Yapılan Kabuller

Proje kapsamında aşağıdaki kabuller benimsenmiştir; mimari ve giriş/çıkış parametreleri bu kabullere göre tasarlanmıştır.

### Merkezi politika ve sürü koordinasyonu

- **Tek merkezi sinir ağı (PPO):** Tüm sürü için tek bir politika (policy) eğitilir. Bu tercih, drone’lar arasında **eşler arası (peer-to-peer) iletişim** olduğu ve her birimin kendi sensör verileriyle sürü merkezini (barycenter) hesaplayıp karar alabileceği varsayımına dayanır.
- **Hibrit aksiyon yapısı:** Her drone için ortak bir hız bileşeni ile bireysel bir hız düzeltmesi (offset) kullanılır. Politika çıktısında yalnızca bu offset’lerin uygulanacağı; ortak hızın sürü koordinasyonu ile zaten belirlendiği kabul edilir.
- **İletişim gecikmesi:** Sürü içi bilgi paylaşımında iletişimden kaynaklı gecikme veya bant genişliği kaybı **yok** kabul edilir; gözlemler anlık ve senkron sayılır.

### Algılama ve gözlem uzayı

- **Engel konum bilgisi:** Gerçek uygulamada drone’ların **lidar** ve **radar** benzeri sensörlerle engellerin konum/mesafe bilgisini elde edebileceği varsayılır. Buna uygun olarak gözlem uzayında her drone için belirli yönlerde **ray tabanlı mesafe** (engel/duvar uzaklığı) girişleri kullanılmıştır; böylece simülasyon, gerçek sensör çıktılarına indirgenmiş bir temsil ile uyumlu tutulur.

---

## Problemi Nasıl Parçaladım?

Ana hedef: 4 drone sürüsünün engellerden kaçınarak hedefe gitmesi.

**Alt problemlere böldüm:**
1. **Ortam tasarımı** — Grid, engel, drone hareketi, çarpışma
2. **Gözlem (observation)** — Agent neyi bilmeli? Merkez pozisyon, hedef yönü, engel mesafeleri, formation durumu
3. **Aksiyon (action)** — 4 drone tek tek mi, ortak mı? Hybrid: ortak hız + per-drone offset
4. **Ödül şekillendirme** — Hedefe yaklaşma, çarpışma cezası, formation cezası
5. **Eğitim** — PPO, hyperparameter’lar
6. **Test** — Rastgele parkur + sabit zorlu parkur

---

## Hangi Yaklaşımları Denedim? Hangisi İşe Yaramadı?

### Observation tasarımı
- **Denenen:** Başta sadece merkez + hedef. Engel bilgisi yoktu.
- **Sonuç:** Engelden kaçınmayı öğrenemedi, sık çarpışıyordu.
- **Çözüm:** Her drone için 4 yönlü ray-casting (engel/duvar mesafesi) eklendi. 16 boyut daha observation’a eklendi.

### Action space
- **Alternatif 1:** 4 drone × 2 = 8 boyut, her drone bağımsız. Ağ çıktı boyutu karmaşık, coordination zor.
- **Alternatif 2:** Sadece ortak hız (2 boyut). Formation korunamıyor, dar geçitlerde sıkışma.
- **Seçilen:** Hybrid — ortak hız (2) + 4 drone offset (8) = 10 boyut. Hem birlikte hareket hem ince ayar mümkün.

### Engel sayısı
- **Sabit 5 engel:** Model belirli sayıya alışıyor, genelleme zayıf.
- **Yapılan:** `n_obstacles_range=(5, 9)` ile her episode farklı sayıda engel. Daha robust sonuç.

### Ödül şekillendirme
- **İlk deneme:** Sadece hedefe mesafe. Formation bozuluyordu.
- **İkinci deneme:** Formation cezası çok yüksek — drone’lar çok sıkı kümeleniyordu, engelden kaçınamıyordu.
- **Mevcut:** `formation_coef=0.3` — dengeli. `-dist*0.01 - collision*10 - formation_err*0.3`, hedefe varma +1000.

---

## Kritik Karar Noktaları

### 1. Rastgele başlangıç/hedef mi, sabit mi?
- **Sabit (sol alt → sağ üst):** Kolay öğrenir ama sadece o yönü bilir.
- **Rastgele:** Daha zor ama tüm yönlerde genelleme. Seçildi: rastgele, min 15 birim mesafe.

### 2. Wall sliding açık mı kapalı mı?
- **Kapalı:** Duvara çarpınca durur; bazı senaryolarda mantıklı.
- **Açık (varsayılan):** Duvar boyunca kayar, köşeden dönebilir. Daha az takılma. Varsayılan: açık.

### 3. VecNormalize kullanımı
- **Kullanmadan:** Gözlemler ölçek farklılıkları nedeniyle eğitimi zorlaştırıyordu.
- **Kullanarak:** `norm_obs=True`, `clip_obs=10` — stabil ve daha hızlı öğrenme.

---

## Nerede Takıldım? Nasıl Çözdüm?

### Import / path hataları
- **Sorun:** `from hybrid_2.env import` — proje köküne göre path, farklı çalışma dizinlerinde hata.
- **Çözüm:** `sys.path.insert` ile script’in bulunduğu dizin eklendi; `from env import` gibi local import’a geçildi.

### Hard course ortamı ile uyum
- **Sorun:** Eğitim rastgele parkurda, test sabit `hard_course_config` ile. Observation/action aynı olmalı.
- **Çözüm:** `DroneSwarmEnvHybrid2HardCourse`, `DroneSwarmEnvHybrid2`’den türetildi; sadece `reset` override, start/target sabit config’den.

### Zorlu parkurda başlangıç/hedef yönü
- **İstek:** Hem sol alt → sağ üst hem sağ üst → sol alt test edilebilmeli.
- **Çözüm:** `swap_start_target` parametresi eklendi; `--no_swap` ile CLI’dan seçilebilir.

---

## Zaman Nasıl Harcandı?

- Ortam + observation/action tasarımı: ~%30  
- Ödül şekillendirme + hyperparameter denemeleri: ~%25  
- PPO eğitimi (2M step): ~%15 (bekleme)  
- Görselleştirme (Pygame), test script’leri, README: ~%20  
- Import/path düzeltmeleri, CLI, .gitignore vb.: ~%10  

---

## Baştan Başlasam Neyi Farklı Yapardım?

1. **Curriculum learning:** Engel sayısını 2 → 5 → 9 gibi kademeli artırmak; önce basit senaryolarda öğretip sonra zorlaştırmak.
2. **Reward logging:** Ödül bileşenlerini (mesafe, çarpışma, formation) ayrı ayrı log’lamak; hangi bileşenin baskın olduğunu görmek.
3. **Daha erken test:** İlk 100K step’te bile hard course’ta periyodik test; overfitting veya yanlış yöne gidişi erkenden fark etmek.
4. **Ray sayısı:** 4 ray yerine 8 ray denemek; daha iyi engel algılama sağlayabilir.
