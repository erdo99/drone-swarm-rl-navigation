# DEVLOG.md — Geliştirme Süreci Kaydı

Bu dosya projenin geliştirme sürecini kronolojik olarak belgeler. Kararlar, denemeler ve öğrenilenler.

---

## Tasarlayanın Notu

Tek bir drone’a sabit bir rota öğretmeye kıyasla, çoklu drone sürüsü için ortak bir politika öğrenmek daha zorlu bir problemdir. Sinir ağlarını girdi → çıktı arasında bir fonksiyon öğrenmek olarak düşünürsek, bu projede:

- **Öğrenilecek davranış sayısı** (engel kaçınma, hedefe yönelme, sürü formasyonunu koruma, birbirine çarpmama),
- **Durum uzayının boyutu** (4 drone, engeller, duvarlar, hedef konumu)

klasik tek-agent navigasyon görevlerine göre anlamlı biçimde daha yüksektir. Dolayısıyla, tek bir eğitim sürecinde hem engellerden kaçmayı, hem hedefe ulaşmayı, hem de sürü içi çarpışmalardan kaçınmayı aynı anda öğretmek gerekir.

Bu karmaşıklığı yönetebilmek için, aşağıda ayrıntılı olarak açıklanan bazı **basitleştirici kabuller** yapılmıştır (merkezi politika, ideal formasyon, iletişimsiz gecikme olmaması, vb.). Bu kabuller, problemi yapay olarak kolaylaştırmaktan çok, odaklanmak istediğimiz asıl özelliği — **çoklu agent koordinasyonu ve engel kaçınma** — öne çıkarmak amacıyla seçilmiştir.

Problem çözümü sırasında farklı RL mimarileri, eğitim stratejileri ve giriş/çıkış tasarımları da araştırılmıştır. Bunların detayları, aşağıdaki “Hangi Yaklaşımları Denedim?” ve “Sinir Ağı Mimarileri Üzerine Notlar” bölümlerinde özetlenmiştir.

## Başta Yapılan Kabuller

Proje kapsamında aşağıdaki kabuller benimsenmiştir; mimari ve giriş/çıkış parametreleri bu kabullere göre tasarlanmıştır.

### Merkezi politika ve sürü koordinasyonu

- **Tek merkezi sinir ağı (PPO):** Tüm sürü için tek bir politika (policy) eğitilir. Bu tercih, drone’lar arasında **eşler arası (peer-to-peer) iletişim** olduğu ve her birimin kendi sensör verileriyle sürü merkezini (barycenter) hesaplayıp karar alabileceği varsayımına dayanır.
- **Hibrit aksiyon yapısı:** Her drone için ortak bir hız bileşeni ile bireysel bir hız düzeltmesi (offset) kullanılır. Politika çıktısında yalnızca bu offset’lerin uygulanacağı; ortak hızın sürü koordinasyonu ile zaten belirlendiği kabul edilir.
- **İletişim gecikmesi:** Sürü içi bilgi paylaşımında iletişimden kaynaklı gecikme veya bant genişliği kaybı **yok** kabul edilir; gözlemler anlık ve senkron sayılır.

### Algılama ve gözlem uzayı

- **Engel konum bilgisi:** Gerçek uygulamada drone’ların **lidar** ve **radar** benzeri sensörlerle engellerin konum/mesafe bilgisini elde edebileceği varsayılır. Buna uygun olarak gözlem uzayında her drone için belirli yönlerde **ray tabanlı mesafe** (engel/duvar uzaklığı) girişleri kullanılmıştır; böylece simülasyon, gerçek sensör çıktılarına indirgenmiş bir temsil ile uyumlu tutulur.

### Hedefleme ve harita modellemesi

- **Ara hedefler metaforu:** Eğitimde ve görselleştirmede kullanılan tek hedef noktası, gerçekte daha uzun bir rotanın (örneğin Ankara–Antalya arası bir rota) üzerindeki **ara hedefler** gibi düşünülmüştür. Drone sürüsü, bu büyük rotanın bir segmentini temsil eden 50×50’lik bir grid üzerinde uçmaktadır.
- **Harita boyutu ve yapı:** Harita boyutu (50×50), engel yoğunluğu ve hedef yapısı; gerçek bir yolculuğun küçük bir kesitini temsil edecek şekilde seçilmiş ve böylece `env.py` içerisindeki eğitim ortamı tasarımı bu soyutlamaya göre belirlenmiştir.

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

### Ortam yapısı

- **İlk tasarım:** Drone sürüsü her seferinde aynı sabit noktadan başlıyor, hedef de sabit bir konumda duruyordu. Engeller bu sabit start–target hattı etrafında rastgele yerleştiriliyordu. Bu yapı, modeli belirli bir “koridoru” ezberlemeye teşvik etti; ajanlar, genel bir navigasyon stratejisi yerine bu sabit senaryoya özel yollar öğrendi.
- **Güncel tasarım:** Başlangıç ve hedef konumları her episode’da harita üzerinde rastgele seçiliyor ve aralarındaki mesafe için minimum bir alt sınır uygulanıyor. Engeller de bu yeni start–target çiftine göre yeniden örnekleniyor. Böylece model, sadece “sol alt → sağ üst” gibi tek bir rotaya değil, haritanın farklı bölgelerinde, farklı yönlerde ve farklı engel düzenlerinde genelleme yapmak zorunda kalıyor.

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

Mevcut ödül fonksiyonu, kodda yaklaşık olarak şu şekilde tanımlıdır (sadeleştirilmiş gösterimle):

- **Hedefe mesafe cezası:**  
  `r_dist = -0.01 * ||center - target||`
- **Çarpışma cezası:**  
  `r_collision = -10.0 * N_collisions`  
  Her drone–engel veya drone–duvar çarpışması için ek ceza.
- **Zaman cezası:**  
  `r_step = -0.01`  
  Her adımda küçük negatif ödül; daha kısa sürede hedefe gitmeye teşvik eder.
- **Formation cezası:**  
  `r_form = -formation_coef * ||positions - ideal_positions||`  (varsayılan `formation_coef = 0.3`)  
  Sürü ideal kare formasyonundan ne kadar saparsa ceza artar.
- **Başarı ödülü (hedefe varış):**  
  `r_success = +1000`  (merkez–hedef mesafesi `< 3.0` olduğunda)
- **Ağır çarpışma cezası (episode sonu):**  
  `r_heavy_collision = -20.0`  (aynı anda `N_collisions >= 2` ise)

Toplam anlık ödül:

`r = r_dist + r_collision + r_step + r_form + r_success + r_heavy_collision`

### İlk modeller ve davranışsal düzeltmeler

- **Tek “vücut” benzetimi ve durup kalma sorunu:** İlk modellerde sürü, tek bir büyük cisim gibi düşünülen basit bir yapı ile eğitildi. Yüksek çarpışma cezası nedeniyle ajan, engelleri gördüğünde belirli bir noktaya kadar ilerleyip sonra tamamen durma eğilimi gösterdi; “hiç hareket etmemek” çarpışmaktan daha az maliyetliydi.
- **Sabit konum cezası ve titreme davranışı:** Bunu çözmek için konum olarak sabit kalmaya doğrudan ceza eklendi. Ancak bu kez ajan, sabit kalmamak için ekranda titreme (çok küçük ileri–geri hareketler) benzeri davranışlar sergilemeye başladı.
- **Zaman eşiği ve hedef ödülünün güçlendirilmesi:** Titremeyi engellemek için, “aynı konumda kalma” cezası yalnızca pozisyon ≈3 saniye boyunca anlamlı biçimde değişmediğinde uygulanacak şekilde yeniden tanımlandı. Aynı zamanda hedefe varış ödülü 10 kat artırılarak, ajanın “hedefe varmak” davranışının göreli önemi yükseltildi.
- **Çarpışma sonrası episode’un hemen bitmesi:** Erken sürümlerde çarpışmalar episode’u hemen sonlandırdığı için, ajan engellerle ilgili yeterince deneyim toplayamıyor ve uzun vadede sağlıklı kaçınma stratejileri öğrenemiyordu.
- **Geri sekme (bounce) ve çarpışma sonrası öğrenme:** Bu nedenle, çarpışma anında yüksek ceza verilmeye devam edilip, ancak episode sonlandırılmayacak şekilde bir “geri sekme” davranışı eklendi. Drone’lar çarpışma sonrası yavaşlayıp geri itilseler de, episode devam ettiği için engellerden kaçınmayı aktif olarak öğrenebiliyorlar.
- **Stage’li yapı ve pratikteki sorunu:** Bir aşamada, stage’li eğitim kullanarak önce yalnızca birlikte uçma/hedefe gitme, sonraki stage’lerde engellerden kaçma öğretme fikri denendi. Ancak bazı stage’lerde uzun süre hiç engel temasının görülmemesi, sonradan eklenecek kaçınma davranışının yeterince örnek görememesine ve toplam öğrenmeye “balta vurmasına” neden oldu.
- **Güncel denge:** Mevcut tasarımda ajan, hem engellere temas etmemeyi hem de hedefe yönelmeyi **aynı eğitim akışı içinde** öğreniyor. Ortak hız bileşeni sayesinde sürünün birlikte hareket etmesi görece kolay öğrenilirken, per-drone offset’ler engellerin etrafından dolaşma ve formasyonu koruma davranışlarını ince ayar düzeyinde mümkün kılıyor.

---

## Sinir Ağı Mimarileri Üzerine Notlar

Bu projede kullanılan son mimari, Stable-Baselines3 PPO’nun **MLP tabanlı politikası**dır; gövde yapısı `[256, 256]` olacak şekilde seçilmiştir. Buna giden yolda, farklı projelerde ve senaryolarda çeşitli ağ mimarileri denendi:

### PPO tabanlı MLP politikalar

- **[256, 256, 128] MLP (en yaygın yapı):**  
  - Çeşitli drone projelerinde (ör. Drone-New_4, Drone-New-5, Drone-New-8/9/10/11, Drone_New-6, Drone-New-15-ppo-discrete) hem policy hem value için ortak veya ayrık head’lerle kullanıldı.  
  - Daha fazla katman ve nöron sayısı sayesinde karmaşık davranışları temsil edebilse de, eğitim kararlılığı ve tuning maliyeti arttı; bazı senaryolarda overfit’e ve hassas hyperparametre bağımlılığına yol açtı.
- **Daha küçük MLP’ler ([128, 128, 64], [64, 64]):**  
  - Daha küçük observation uzayına sahip deneysel ortamlarda ve hızlı “smoke test” script’lerinde kullanıldı.  
  - Hafif olmalarına rağmen, bu projedeki gibi yüksek boyutlu observation + çoklu drone davranışı için genellikle yetersiz kaldılar (hedef ve engel kombinasyonlarını tam öğrenemediler).
- **[256, 256] MLP (bu projedeki yapı):**  
  - Bu çözüm, kapasite ve kararlılık arasında pratik bir denge sağladığı için tercih edildi.  
  - Hem eğitim süresi makul kaldı hem de reward eğrileri, daha derin mimarilere göre daha öngörülebilir bir şekilde ilerledi.

### LSTM + MLP politikalar

- Bazı önceki denemelerde, geçmiş hareketlerden öğrenebilmek için 256 boyutlu LSTM katmanı + `[256, 256]` MLP gövdesi kullanıldı.  
- Bu yapı, özellikle kısmi gözlem (partial observability) içeren senaryolarda faydalı olsa da, bu projede kullanılan observation yapısı (merkez, hız, engel ray’ları, vs.) zaten anlık durumu yeterince temsil ettiği için ek karmaşıklığa gerek duyulmadı.

### DQN / DDQN mimarileri

- **Dueling DQN:**  
  - Obs → 256 → 256 → 128 gövde, ardından value ve advantage stream’leri (her biri 128 → 128 → output) içeren yapılar denendi.  
  - Ayrık aksiyon uzayı için makul sonuçlar verse de, bu projede ihtiyaç duyulan **sürekli / ince ayarlı kontrol** için PPO + continuous action uzayı daha uygun bulundu.
- **DDQN (256 tabanlı MLP):**  
  - Obs → 256 → 256 → 128 → action_dim yapısı ile discrete kontrol denemeleri yapıldı; ancak yine continuous kontrol gereksinimi ve sürü koordinasyonu nedeniyle ana çözüm olarak tercih edilmedi.

### Daha karmaşık özel mimariler

- Bazı projelerde özel feature extractor’lar (ör. toplam feature_dim ≈ 176, ardından actor/critic head’lerde [256, 256, 128, 64]) kullanıldı.  
- Bu mimariler temsil gücü açısından güçlü olsa da, tuning maliyeti yüksek ve davranış analizi daha zordu; bu case study’de odak noktası **ortam tasarımı + reward + kontrol mimarisi** olduğu için, daha sade bir MLP yapısı tercih edildi.

**Sonuç:** Bu proje özelinde `[256, 256]` MLP’li PPO politikası, hem önceki deneyimlerden gelen bilgiye dayanarak hem de pratik gözlemlerle **“yeterince güçlü ama yönetilebilir karmaşıklıkta”** bir tercih olarak belirlendi. Daha büyük haritalar veya daha zengin sensör setleri hedeflendiğinde, yukarıda bahsedilen daha derin veya LSTM’li mimariler yeniden değerlendirilebilir.

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

---

## İleriki Çalışmalar İçin Öneriler

Bu projede tek bir PPO politikası, tüm start–target konfigürasyonları ve yönler için ortak bir çözüm öğrenmeye çalışıyor. İleride, bu karmaşık problemi daha yönetilebilir alt problemlere bölmek için **yüksek seviyeli bir “yön/hattı seçici” (router) katmanı** düşünmek faydalı olabilir:

- Harita, örneğin 8 bölge/konfigürasyona ayrılabilir: sol üst ↔ sağ alt, sağ üst ↔ sol alt, orta üst ↔ orta alt, vb.
- Üstte, sadece **hangi bölgesel senaryoda olduğuna** karar veren küçük bir politika (veya basit kural tabanlı router) bulunur.
- Altta ise her bölge için ayrı veya paylaşılmış ama direction-aware alt politikalar (sub-policy) eğitilir; bu alt politikalar, kendi senaryolarında daha dar bir dağılımı öğrenmek zorunda oldukları için PPO açısından daha kolay optimize edilebilir.
- Bu yapının overfit’e kaçmaması için, her bölge içinde hedef noktası tek bir sabit koordinat değil, o bölge etrafındaki bir alan içinde rastgele seçilmelidir (bölge içi rastgelelik korunur).

Böyle bir hiyerarşik/pieces-wise yaklaşım, özellikle daha büyük haritalarda ve çok daha uzun rotalarda **“önce hangi koridor/yön, sonra ince manevra”** ayrımını netleştirerek hem eğitim süresini kısaltabilir hem de genelleme kabiliyetini artırabilir.
