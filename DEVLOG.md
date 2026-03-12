# DEVLOG.md — Geliştirme Süreci Kaydı

Bu dosya projenin geliştirme sürecini kronolojik olarak belgeler. Kararlar, denemeler ve öğrenilenler.

---

## Tasarlayanın Notu

Tek bir drone'a sabit bir rota öğretmeye kıyasla, çoklu drone sürüsü için ortak bir politika öğrenmek daha zorlu bir problemdir. Sinir ağlarını girdi → çıktı arasında bir fonksiyon öğrenmek olarak düşünürsek, bu projede:

- **Öğrenilecek davranış sayısı** (engel kaçınma, hedefe yönelme, sürü formasyonunu koruma, birbirine çarpmama),
- **Durum uzayının boyutu** (4 drone, engeller, duvarlar, hedef konumu)

klasik tek-agent navigasyon görevlerine göre anlamlı biçimde daha yüksektir. Dolayısıyla, tek bir eğitim sürecinde hem engellerden kaçmayı, hem hedefe ulaşmayı, hem de sürü içi çarpışmalardan kaçınmayı aynı anda öğretmek gerekir.

Bu karmaşıklığı yönetebilmek için, aşağıda ayrıntılı olarak açıklanan bazı **basitleştirici kabuller** yapılmıştır (parameter sharing, ideal formasyon, iletişimsiz gecikme olmaması, vb.). Bu kabuller, problemi yapay olarak kolaylaştırmaktan çok, odaklanmak istediğimiz asıl özelliği — **çoklu agent koordinasyonu ve engel kaçınma** — öne çıkarmak amacıyla seçilmiştir.

Problem çözümü sırasında farklı RL mimarileri, eğitim stratejileri ve giriş/çıkış tasarımları da araştırılmıştır. Bunların detayları, aşağıdaki "Hangi Yaklaşımları Denedim?" ve "Sinir Ağı Mimarileri Üzerine Notlar" bölümlerinde özetlenmiştir.

---

## RL Algoritması Seçimi: PPO Neden? Alternatifler Neden Elendi?

Bu projede **PPO (Proximal Policy Optimization)** seçildi. Alternatifler şöyle değerlendirildi:

| Algoritma | Neden seçilmedi / elendi |
|-----------|---------------------------|
| **MAPPO** | Mevcut mimari zaten parameter sharing ile decentralized execution yapıyor; her drone kendi lokal obs'unu kullanıyor. MAPPO'nun getireceği ek karmaşıklık bu setup'ta fazladan değer üretmedi. |
| **SAC / TD3** | Rastgele başlangıç/hedef ortamında PPO daha stabil. SAC replay buffer farklı episode'lardan karışık örnek alacağı için davranış tutarlılığı PPO'da daha iyi bulundu. |
| **A3C / A2C** | PPO daha stabil (clip + GAE); bu ortam için daha iyi sonuç. |
| **DQN / DDQN** | **Sürekli aksiyon** gerekli; dar geçitlerde ince manevra için continuous kontrol şart. |
| **QMIX / COMA** | Discrete action odaklı; bu projede continuous + parameter sharing daha uygun. |

**Özet:** PPO sürekli aksiyon, parameter sharing ve SB3 ile uyumlu olduğu için seçildi. MAPPO denenseydi ilginç olurdu ama mevcut yapı zaten her drone'un lokal obs ile bağımsız karar ürettiği bir setup, PPO bu durumda yeterli.

---

## Başta Yapılan Kabuller

Proje kapsamında aşağıdaki kabuller benimsenmiştir; mimari ve giriş/çıkış parametreleri bu kabullere göre tasarlanmıştır.

### Parameter Sharing ve sürü koordinasyonu

- **Parameter Sharing (CTDE):** 4 drone için tek model; her drone kendi 12 boyutlu lokal gözlemini alır, aynı ağırlıklarla aksiyon üretir. Neden parameter sharing? (1) Her drone görece benzer bir görevi yapıyor — engelden kaç, formasyonu koru, hedefe git. Aynı ağırlıkları paylaşmak bu benzerlikten doğal olarak faydalanıyor. (2) Ayrı 4 model eğitmek yerine tek bir modeli optimize etmek hem eğitimi kolaylaştırıyor hem de genellemeyi artırıyor. (3) Decentralized execution gerçekçi: her drone sadece kendi sensörüne erişiyor, merkezi bir koordinatöre gerek kalmıyor.
- **Lokal gözlem:** Her drone yalnızca kendi pozisyonunu, hızını, hedefe olan göreceli vektörünü, formasyon hatasını ve 4 yönlü ray mesafelerini görüyor. Başka drone'ların pozisyonuna doğrudan erişimi yok; koordinasyon yalnızca öğrenilmiş politikadan çıkıyor.
- **İletişim gecikmesi:** Sürü içi bilgi paylaşımında iletişimden kaynaklı gecikme veya bant genişliği kaybı **yok** kabul edilir; gözlemler anlık ve senkron sayılır.

### Algılama ve gözlem uzayı

- **Engel konum bilgisi:** Gerçek uygulamada drone'ların **lidar** ve **radar** benzeri sensörlerle engellerin konum/mesafe bilgisini elde edebileceği varsayılır. Buna uygun olarak gözlem uzayında her drone için belirli yönlerde **ray tabanlı mesafe** (engel/duvar uzaklığı) girişleri kullanılmıştır; böylece simülasyon, gerçek sensör çıktılarına indirgenmiş bir temsil ile uyumlu tutulur.

### Hedefleme ve harita modellemesi

- **Ara hedefler metaforu:** Eğitimde ve görselleştirmede kullanılan tek hedef noktası, gerçekte daha uzun bir rotanın (örneğin Ankara–Antalya arası bir rota) üzerindeki **ara hedefler** gibi düşünülmüştür. Drone sürüsü, bu büyük rotanın bir segmentini temsil eden 50×50'lik bir grid üzerinde uçmaktadır.
- **Harita boyutu ve yapı:** Harita boyutu (50×50), engel yoğunluğu ve hedef yapısı; gerçek bir yolculuğun küçük bir kesitini temsil edecek şekilde seçilmiş ve böylece `env_shared_v3.py` içerisindeki eğitim ortamı tasarımı bu soyutlamaya göre belirlenmiştir.

---

## Problemi Nasıl Parçaladım?

Ana hedef: 4 drone sürüsünün engellerden kaçınarak hedefe gitmesi.

**Alt problemlere böldüm:**
1. **Ortam tasarımı** — Grid, engel, drone hareketi, çarpışma, rota tabanlı engel yerleştirme
2. **Gözlem (observation)** — Agent neyi bilmeli? Lokal pozisyon, hedef yönü, engel mesafeleri, formasyon durumu
3. **Aksiyon (action)** — Her drone bağımsız 2 boyutlu hız; parameter sharing ile tek model
4. **Ödül şekillendirme** — Hedefe yaklaşma, çarpışma cezası, formasyon cezası, yakınlık cezaları
5. **Eğitim** — PPO, hyperparameter'lar, SyncVecNormalize callback
6. **Test** — Rastgele parkur + sabit zorlu parkur

---

## Hangi Yaklaşımları Denedim? Hangisi İşe Yaramadı?

### Mimari: Hybrid2'den Shared Policy'ye geçiş

İlk sürümde **Hybrid2** mimarisi kullanılıyordu: tek merkezi model, 10 boyutlu aksiyon (ortak hız + per-drone offset), 30 boyutlu merkez tabanlı gözlem. Bu yapı belli bir noktaya kadar çalıştı ama önemli sorunlar çıkardı.

En büyük problem gözlem tasarımıydı. Tüm gözlem sürü merkezinden bakıyordu; bireysel drone'ların nerede olduğu yerine merkez pozisyon + formasyon hataları vardı. Bu yüzden model "sürüyü bir bütün olarak" hareket ettirmeye çalışıyordu ama dar geçitlerde veya asimetrik engel düzenlerinde tek tek drone'ların ne yapması gerektiğini ayırt edemiyordu. Ayrıca `VecNormalize` ile birlikte bazı gözlem boyutu uyumsuzlukları ve yeniden başlatma hataları da çıktı; production'da güvenilir çalışmıyordu.

**Shared Policy** mimarisine geçince mimari tamamen tersine döndü: artık her drone kendi lokal gözlemini alıyor, aynı model ağırlıkları 4 drone arasında paylaşılıyor. Gözlem boyutu 30'dan 48'e çıktı (4×12) ama her drone sadece kendi 12 boyutunu görüyor. Hem daha temiz hem de daha gerçekçi bir yapı oldu.

### Ortam yapısı

- **İlk tasarım:** Drone sürüsü her seferinde aynı sabit noktadan başlıyor, hedef de sabit bir konumda duruyordu. Engeller bu sabit start–target hattı etrafında rastgele yerleştiriliyordu. Bu yapı, modeli belirli bir "koridoru" ezberlemeye teşvik etti; ajanlar, genel bir navigasyon stratejisi yerine bu sabit senaryoya özel yollar öğrendi.
- **Güncel tasarım:** Başlangıç ve hedef konumları her episode'da harita üzerinde rastgele seçiliyor ve aralarındaki mesafe için minimum bir alt sınır uygulanıyor. Engeller de bu yeni start–target çiftine göre yeniden örnekleniyor. Bunun yanında `obstacles_on_route=True` ile engellerin %60'ı artık start→target koridoruna yerleştiriliyor; modelin çözmesi gereken engel düzeni daha gerçekçi ama hem koridorda hem rastgele engel olduğu için aşırı zor da değil.

### Observation tasarımı

- **Denenen:** Başta sadece merkez + hedef. Engel bilgisi yoktu.
- **Sonuç:** Engelden kaçınmayı öğrenemedi, sık çarpışıyordu.
- **Çözüm:** Her drone için 4 yönlü ray-casting (engel/duvar mesafesi) eklendi. Shared policy'de bu her drone'un kendi local obs'una girdi, dolayısıyla her drone kendi etrafındaki engelleri görüyor.

### Engel sayısı

- **Sabit 5 engel:** Model belirli sayıya alışıyor, genelleme zayıf.
- **Yapılan:** `n_obstacles_range=(7, 9)` ile her episode farklı sayıda engel. Daha robust sonuç.

### Ödül şekillendirme

- **İlk deneme:** Sadece hedefe mesafe. Formation bozuluyordu.
- **İkinci deneme:** Formation cezası çok yüksek — drone'lar çok sıkı kümeleniyordu, engelden kaçınamıyordu.
- **Üçüncü deneme:** Drone-drone çarpışmaları sık görününce `proximity_penalty` ve `min_separation_penalty` eklendi. İki ayrı mekanizma: eşik altında mesafeye orantılı ceza + minimum mesafe altında sabit ağır ceza.
- **Mevcut:** `formation_coef=0.3` — dengeli. Ödül bileşenleri:

  - **Hedefe mesafe cezası:** `r_dist = -0.01 * ||center - target||`
  - **Çarpışma cezası:** `r_collision = -10.0 * N_collisions`
  - **Zaman cezası:** `r_step = -0.01`
  - **Formation cezası:** `r_form = -formation_coef * ||positions - ideal_positions||`
  - **Yakınlık cezası:** `r_prox = -proximity_penalty_coef * (threshold - dist)` (eşik altındaki çiftler için)
  - **İç içe girme cezası:** `r_sep = -min_drone_separation_penalty` (min_drone_separation altına girilirse)
  - **Başarı ödülü:** `+1000.0` (dist < 3.0)
  - **Ağır çarpışma:** `-20.0` + episode sonu (aynı anda ≥2 drone çarpışırsa)

---

## Sinir Ağı Mimarileri Üzerine Notlar

### [256, 256] MLP (bu projedeki yapı)

Bu çözüm, kapasite ve kararlılık arasında pratik bir denge sağladığı için tercih edildi. Hem eğitim süresi makul kaldı hem de reward eğrileri, daha derin mimarilere göre daha öngörülebilir bir şekilde ilerledi. Shared policy setup'ında giriş boyutu 48, çıkış 8; bu boyutlar için [256, 256] fazlasıyla yeterli kapasitede.

### LSTM + MLP politikalar

Bazı önceki denemelerde, geçmiş hareketlerden öğrenebilmek için 256 boyutlu LSTM katmanı + `[256, 256]` MLP gövdesi kullanıldı. Bu yapı, özellikle kısmi gözlem (partial observability) içeren senaryolarda faydalı olsa da, bu projede kullanılan observation yapısı (lokal pozisyon, hız, engel ray'ları, vs.) zaten anlık durumu yeterince temsil ettiği için ek karmaşıklığa gerek duyulmadı.

### DQN / DDQN mimarileri

- **Dueling DQN:** Obs → 256 → 256 → 128 gövde, ardından value ve advantage stream'leri (her biri 128 → 128 → output) içeren yapılar denendi. Ayrık aksiyon uzayı için makul sonuçlar verse de, bu projede ihtiyaç duyulan **sürekli / ince ayarlı kontrol** için PPO + continuous action uzayı daha uygun bulundu.
- **DDQN (256 tabanlı MLP):** Obs → 256 → 256 → 128 → action_dim yapısı ile discrete kontrol denemeleri yapıldı; ancak yine continuous kontrol gereksinimi ve sürü koordinasyonu nedeniyle ana çözüm olarak tercih edilmedi.

### Daha karmaşık özel mimariler

Bazı projelerde özel feature extractor'lar (ör. toplam feature_dim ≈ 176, ardından actor/critic head'lerde [256, 256, 128, 64]) kullanıldı. Bu mimariler temsil gücü açısından güçlü olsa da, tuning maliyeti yüksek ve davranış analizi daha zordu; bu case study'de odak noktası **ortam tasarımı + reward + mimari seçim** olduğu için, daha sade bir MLP yapısı tercih edildi.

**Sonuç:** Bu proje özelinde `[256, 256]` MLP'li PPO politikası, hem önceki deneyimlerden gelen bilgiye dayanarak hem de pratik gözlemlerle **"yeterince güçlü ama yönetilebilir karmaşıklıkta"** bir tercih olarak belirlendi. Daha büyük haritalar veya daha zengin sensör setleri hedeflendiğinde, yukarıda bahsedilen daha derin veya LSTM'li mimariler yeniden değerlendirilebilir.

---

## Kritik Karar Noktaları

### 1. Rastgele başlangıç/hedef mi, sabit mi?
- **Sabit (sol alt → sağ üst):** Kolay öğrenir ama sadece o yönü bilir.
- **Rastgele:** Daha zor ama tüm yönlerde genelleme. Seçildi: rastgele, min 15 birim mesafe.

### 2. Engeller nereye yerleştirilmeli?
- **Tam rastgele (grid'e):** Model farklı engel düzenlerini görür ama zorlu senaryolar az çıkıyor; drone doğrudan hedefe gidip çoğu engeli atlayabiliyor.
- **Rotaya yerleştirilmiş (`obstacles_on_route=True`):** Engellerin %60'ı start→target koridoruna denk geliyor. Drone önce rotayı temizlemek zorunda; daha gerçekçi ve eğitici bir senaryo. Seçildi: varsayılan `route_obstacle_ratio=0.6`.

### 3. VecNormalize kullanımı
- **Kullanmadan:** Gözlemler ölçek farklılıkları nedeniyle eğitimi zorlaştırıyordu.
- **Kullanarak:** `norm_obs=True`, `clip_obs=10` — stabil ve daha hızlı öğrenme. Ek olarak `SyncVecNormalizeCallback` ile eval ortamı, train ortamının obs istatistiklerini otomatik takip ediyor.

### 4. Wall sliding açık mı kapalı mı?
- **Kapalı:** Duvara çarpınca durur; bazı senaryolarda mantıklı.
- **Açık (varsayılan):** Duvar boyunca kayar, köşeden dönebilir. Daha az takılma. Varsayılan: açık.

---

## Parametre Gerekçelendirmesi (Mülakatta Gelebilecek Sorular)

### safety_radius = 2.0, obstacle_radius = 3.0
- **safety_radius (2.0):** Drone–engel veya drone–duvar mesafesi bu değerin altına düştüğünde çarpışma sayılır. Formation offset'ler ~2 birim; drone'lar birbirine 3-4 birim civarı yaklaşabiliyor. safety_radius=2.0, engel boyutu (obstacle_radius=3) ile uyumlu: engelden en az ~1 birim boşluk kalacak şekilde çarpışma tetiklenir. Daha küçük (1.0) seçilse geç algılama, daha büyük (3.0+) seçilse gereksiz erken ceza riski var.
- **obstacle_radius (3.0):** Grid 50×50, formation ~4 birim genişliğinde. 3.0 ile engeller dar geçitler oluşturabilecek kadar büyük ama aşılamaz değil. Grid ölçeğine göre makul; 2.0 çok küçük, 5.0 çok büyük kalırdı.

### Observation: 48 boyut (4×12), 4 ray neden?
- **48 boyut yapısı:** Her drone için 2 (pozisyon) + 2 (hız) + 2 (hedef vektörü) + 2 (formasyon hatası) + 4 (ray mesafeleri) = 12 boyut → 4 drone × 12 = 48. Boyut sayısı bu bileşenlerin toplamı; tasarım kararı önce "her drone ne bilmeli?" sorusuna cevap, sonra bunların concat'i.
- **4 ray neden?** Başlangıçta sadece pozisyon+hedef vardı; engel bilgisi yoktu, çarpışma çoktu. 4 yön (0°, 90°, 180°, 270°) — ön, sağ, arka, sol — temel engel algılama için yeterli bulundu. 8 ray daha iyi algılama sağlayabilir ama obs boyutu 4×16=64'e çıkar; bu case study'de 4 ray ile çalışan bir çözüm elde edildi, trade-off olarak kabul edildi.

### Eğitim süresi: 2M timestep nasıl belirlendi?
- **1M default, 2M önerilen:** `train_v2.py` varsayılanı 2M. Bu tercih; reward eğrisinin saturasyona yaklaşması ve eval skorunun stabilize olması için yeterli süre. 500K'da erken durursa genelleme zayıf kalabiliyor; 2M ile hem rastgele parkurda hem hard course'ta tutarlı davranış gözlemlendi. Kesin bir grid search yapılmadı; deneyimsel olarak "1M'den iyi, 3M'de marginal fayda" gözlemi.

### Formation cezası ile çarpışma cezası çakışırsa ne olur?
- **Öncelik:** Çarpışma cezası (-10/çarpışma) formation cezasından (-0.3×formation_err) **çok daha büyük**. Dolayısıyla politika önce çarpışmadan kaçınmayı öğrenir; formation ikincil kalır. Dar geçitten geçerken formation bozulması kabul edilebilir çünkü çarpışma cezası daha ağır.
- **Çakışma senaryosu:** "Sıkı formation koru" ile "engele çarpma" çelişirse (dar geçit), politika formation'dan taviz verip geçitten geçmeyi tercih eder — bu istenen davranış. Katsayılar (`formation_coef=0.3`, collision=-10) bu önceliği yansıtacak şekilde seçildi.

---

## Nerede Takıldım? Nasıl Çözdüm?

### Hybrid2'den Shared Policy'ye geçişteki gözlem boyutu uyumsuzluğu
- **Sorun:** `VecNormalize` ile birlikte `obs_rms` boyutu eski modelin 30 boyutlu obs'una kilitleniyordu; yeni 48 boyutlu obs yüklenince boyut uyumsuzluğu patlıyordu.
- **Çözüm:** `SyncVecNormalizeCallback` yazıldı — eval ortamı, train ortamının `obs_rms`'ini doğrudan referans alıyor; ayrı normalizasyon istatistiği tutmuyor. Bu hem boyut sorununu çözdü hem de eval sırasında eski istatistiklerin kayması problemini ortadan kaldırdı.

### Import / path hataları
- **Sorun:** `from shared.env_shared import` — proje köküne göre path, farklı çalışma dizinlerinde hata.
- **Çözüm:** `sys.path.insert` ile script'in bulunduğu dizin eklendi; local import'a geçildi.

### Hard course ortamı ile uyum
- **Sorun:** Eğitim rastgele parkurda, test sabit `hard_course_config` ile. Observation/action aynı olmalı.
- **Çözüm:** `DroneSwarmSharedHardCourseEnv`, `DroneSwarmSharedEnv`'den türetildi; sadece `reset` override, start/target sabit config'den.

### Zorlu parkurda başlangıç/hedef yönü
- **İstek:** Hem normal yön hem swap test edilebilmeli.
- **Çözüm:** `swap_start_target` parametresi eklendi; `--swap` ile CLI'dan seçilebilir.

---

## Zaman Nasıl Harcandı?

- Ortam + observation/action tasarımı (Hybrid2'den Shared Policy'ye migration dahil): ~%35
- Ödül şekillendirme + hyperparameter denemeleri: ~%20
- PPO eğitimi (2M step): ~%15 (bekleme)
- Görselleştirme (Pygame), test script'leri, README/DEVLOG: ~%20
- Import/path düzeltmeleri, CLI, callback'ler, .gitignore vb.: ~%10

---

## Baştan Başlasam Neyi Farklı Yapardım?

1. **Shared Policy ile başlamak:** Hybrid2 mimarisini geliştirip sonra geçmek yerine, baştan parameter sharing + lokal obs tasarlamak zaman kazandırırdı. Merkez tabanlı gözlem dar geçitlerde yetersiz kalıyor; bunu erken fark etseydim migration maliyeti olmayacaktı.
2. **Curriculum learning:** Engel sayısını 2 → 5 → 9 gibi kademeli artırmak; önce basit senaryolarda öğretip sonra zorlaştırmak.
3. **Reward logging:** Ödül bileşenlerini (mesafe, çarpışma, formation, proximity) ayrı ayrı log'lamak; hangi bileşenin baskın olduğunu görmek. Hangi cezanın ne zaman devreye girdiğini görmek debug süresini kısaltırdı.
4. **Daha erken test:** İlk 100K step'te bile hard course'ta periyodik test; overfitting veya yanlış yöne gidişi erkenden fark etmek.
5. **Ray sayısı:** 4 ray yerine 8 ray denemek; daha iyi engel algılama sağlayabilir. Şu an köşegen yönlerdeki engeller için bilgi yok.

---

## İleriki Çalışmalar İçin Öneriler

Bu projede tek bir PPO politikası, tüm start–target konfigürasyonları ve yönler için ortak bir çözüm öğrenmeye çalışıyor. İleride, bu karmaşık problemi daha yönetilebilir alt problemlere bölmek için **yüksek seviyeli bir "yön/hattı seçici" (router) katmanı** düşünmek faydalı olabilir:

- Harita, örneğin 8 bölge/konfigürasyona ayrılabilir: sol üst ↔ sağ alt, sağ üst ↔ sol alt, orta üst ↔ orta alt, vb.
- Üstte, sadece **hangi bölgesel senaryoda olduğuna** karar veren küçük bir politika (veya basit kural tabanlı router) bulunur.
- Altta ise her bölge için ayrı veya paylaşılmış ama direction-aware alt politikalar (sub-policy) eğitilir; bu alt politikalar, kendi senaryolarında daha dar bir dağılımı öğrenmek zorunda oldukları için PPO açısından daha kolay optimize edilebilir.
- Bu yapının overfit'e kaçmaması için, her bölge içinde hedef noktası tek bir sabit koordinat değil, o bölge etrafındaki bir alan içinde rastgele seçilmelidir (bölge içi rastgelelik korunur).

Böyle bir hiyerarşik/pieces-wise yaklaşım, özellikle daha büyük haritalarda ve çok daha uzun rotalarda **"önce hangi koridor/yön, sonra ince manevra"** ayrımını netleştirerek hem eğitim süresini kısaltabilir hem de genelleme kabiliyetini artırabilir.