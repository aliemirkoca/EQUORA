# EQUORA — Faz 1: Elde Kalacak Ürün Tahmini (ML) Mimarisi

Bu doküman, **EQUORA** projesinin birinci aşaması olan *son tüketim tarihi (SKT) yaklaşan ve satılamayacak ürünlerin önceden tahmin edilmesi* probleminin teorik çerçevesini, veri modelini, matematiksel formülasyonunu ve doğrulama stratejisini tanımlar.

## 1. Problemin Tanımı ve Gözlem Birimi

Klasik perakende modelleri *"Yarın* $X$ *ürününden kaç adet satılır?"* sorusuna odaklanır. EQUORA ise problemi **sonlu ömürlü bir partinin (lot) sağkalım ve kuyruk problemi** olarak ele alır.

Veritabanımızdaki `products` ve `stock_movements` (`IN`, `OUT`, `WASTE`) tablolarından **zamanda geriye dönük sızıntısız kesitler (point-in-time snapshots)** üretilir. Modelin karar birimi tek bir ürün değil, **(Parti, Gün)** ikilisidir:

$$
(b, t)
$$

**Değişkenler:**

* $b$: Belirli bir mağaza (`store_id`), ürün (`product_id`) ve son tüketim tarihine (`expiry_date`) sahip tekil ürün partisi.

* $t$: Tahminin yapıldığı gün (gün sonu değerlendirme anı).

Her $t$ günü akşamında, rafta hâlâ stoğu bulunan her aktif parti $b$ için şu iki temel soruya yanıt aranır:

1. Bu parti son tüketim tarihine kadar tamamen tükenecek mi, yoksa elde fire kalacak mı?

2. Eğer elde kalacaksa, SKT günü geldiğinde tahmini kaç adet ürün çöpe gidecek?

## 2. Anlık Stok ve FEFO Kuyruk Fiziği

### 2.1. Anlık Kalan Stok Hesabı

`stock_movements` tablosundaki hareket kayıtlarından, herhangi bir $t$ gününün sonundaki kalan stok şu şekilde yeniden inşa edilir:

$$
I_{b,t} = \sum_{\tau \le t} Q^{\text{IN}}_{b,\tau} - \sum_{\tau \le t} Q^{\text{OUT}}_{b,\tau}
$$

**Değişkenler:**

* $I_{b,t}$: Parti $b$'nin $t$ günü sonundaki kalan stok adedi (`remaining_qty`).

* $\tau$: $t$ gününe kadar (ve $t$ günü dahil) gerçekleşmiş hareket zamanları.

* $Q^{\text{IN}}_{b,\tau}$: Parti $b$'ye $\tau$ zamanında giren (`IN`) ürün adedi.

* $Q^{\text{OUT}}_{b,\tau}$: Parti $b$'den $\tau$ zamanında satılan (`OUT`) ürün adedi.

### 2.2. Üst Kuyruk (Upstream FEFO Queue) ve Efektif Stok

Market operasyonları **FEFO (First Expired, First Out)** prensibiyle çalıştığı için, aynı ürünün daha yakın SKT'li bir partisi tükenmeden yeni partiden satış gerçekleşmez. Bu nedenle bir partinin gerçek satış baskısını ölçmek için önündeki kuyruk stoğu da hesaba katılır:

$$
I^{\text{eff}}_{b,t} = I_{b,t} + \sum_{k \in \mathcal{U}(b,t)} I_{k,t}
$$

**Değişkenler:**

* $I^{\text{eff}}_{b,t}$: Parti $b$'nin $t$ günündeki efektif (kuyruk dahil) stok adedi.

* $I_{b,t}$: Parti $b$'nin kendi kalan stoğu.

* $\mathcal{U}(b,t)$: Aynı mağaza ve aynı ürüne ait olup, $t$ gününde hâlâ aktif olan ve son tüketim tarihi parti $b$'den daha erken olan ($T_k < T_b$) partiler kümesi.

* $I_{k,t}$: Ön kuyruktaki $k$ partisinin $t$ günündeki kalan stok adedi.

## 3. Model Mimarisi: İki Aşamalı (Hurdle) Yapı

Veritabanımızdaki dağılımda kapanmış partilerin yaklaşık yarısında fire sıfır ($W_b = 0$), diğer yarısında ise pozitif bir tam sayıdır ($W_b > 0$). İstatistikte bu yapıya **sıfır şişkinlikli (zero-inflated)** veri denir.

Tek bir klasik regresyon modeli bu dağılımda sıfır ile pozitif değerler arasında ortalama almaya çalışıp yanlı tahmin üretir. Bu nedenle **İki Aşamalı Hurdle Mimarisi** (LightGBM / XGBoost tabanlı) planlanmıştır:

### Aşama 1: Fire Olasılığı Sınıflandırıcısı (Classification)

Bir partinin SKT günü geldiğinde elde kalma olasılığını tahmin eder:

$$
\hat{p}_{b,t} = P(W_b > 0 \mid \mathbf{x}_{b,t})
$$

**Değişkenler:**

* $\hat{p}_{b,t}$: Parti $b$'nin $t$ günü itibarıyla fire verme olasılığı ($0 \le \hat{p}_{b,t} \le 1$).

* $W_b$: Parti $b$'nin SKT sonunda gerçekleşen gerçek fire adedi (`WASTE`).

* $\mathbf{x}_{b,t}$: Parti $b$'nin $t$ gününe kadar bilinen tüm durum ve geçmiş özellik vektörü.

### Aşama 2: Koşullu Fire Miktarı Regresörü (Conditional Regression)

Yalnızca gerçekten fire vermiş ($W_b > 0$) partiler üzerinde eğitilir ve *"eğer fire oluşursa kaç adet kalır?"* sorusunu yanıtlar:

$$
\hat{q}_{b,t} = \mathbb{E}[W_b \mid W_b > 0, \mathbf{x}_{b,t}]
$$

**Değişkenler:**

* $\hat{q}_{b,t}$: Fire oluşması koşuluyla parti $b$'den elde kalması beklenen ürün adedi.

* $\mathbb{E}[\cdot]$: Koşullu beklenen değer operatörü.

* $W_b$: Parti $b$'nin gerçek fire adedi.

* $\mathbf{x}_{b,t}$: Parti $b$'nin $t$ günündeki özellik vektörü.

### Birleşik Beklenen Fire Tahmini

İki modelin çıktısı birleştirilerek partinin beklenen fire adedi elde edilir; ayrıca $\hat{p}_{b,t}$ ve $\hat{q}_{b,t}$ değerleri Faz 2'deki RL ajanına ayrı ayrı sinyal olarak beslenir:

$$
\hat{W}_{b,t} = \hat{p}_{b,t} \times \hat{q}_{b,t}
$$

**Değişkenler:**

* $\hat{W}_{b,t}$: Parti $b$ için $t$ gününde hesaplanan beklenen fire adedi.

* $\hat{p}_{b,t}$: Aşama 1 modelinden gelen fire olasılığı.

* $\hat{q}_{b,t}$: Aşama 2 modelinden gelen koşullu fire miktarı tahmini.

> **Alternatif Kıyaslama (Benchmark):** Hurdle mimarisine ek olarak, sıfır yığılmasını tek dağılımda modelleyebilen **Tweedie Kayıp Fonksiyonlu (**$1 < p < 2$**)** tekil bir LightGBM modeli de kıyaslama (ablation) amacıyla test edilecektir.

## 4. Alan Bilgisine Dayalı Özellik Mühendisliği (Feature Engineering)

Ağaç tabanlı modeller oranları ve fiziksel envanter ilişkilerini doğrudan ham sütunlardan öğrenmekte zorlanır. Bu nedenle aşağıdaki temel özellikler Polars ile türetilecektir:

### 4.1. SKT'ye Kalan Gün (Days to Expiry)

$$
\text{DTE}_{b,t} = T_b - t
$$

**Değişkenler:**

* $\text{DTE}_{b,t}$: Parti $b$'nin $t$ gününden itibaren son tüketim tarihine kalan gün sayısı.

* $T_b$: Parti $b$'nin son tüketim tarihi (`expiry_date`).

* $t$: Gözlem / tahmin tarihi.

### 4.2. Kalan Stok Oranı (Remaining Ratio)

$$
r_{b,t} = \frac{I_{b,t}}{Q^{\text{IN}}_{b}}
$$

**Değişkenler:**

* $r_{b,t}$: Parti $b$'nin ilk giriş miktarına göre rafta kalan yüzdelik oranı ($0 \le r_{b,t} \le 1$).

* $I_{b,t}$: Parti $b$'nin $t$ günündeki kalan stoğu.

* $Q^{\text{IN}}_{b}$: Parti $b$'nin markete ilk giriş (`IN`) toplam adedi.

### 4.3. Hareketli Satış Hızı (Rolling Sales Velocity)

Ürün ve mağaza bazında son $w$ günlük ortalama satış hızını ($w \in \{3, 7, 14\}$ gün) ölçer:

$$
v^{(w)}_{s,u,t} = \frac{1}{w} \sum_{j=0}^{w-1} S_{s,u,t-j}
$$

**Değişkenler:**

* $v^{(w)}_{s,u,t}$: Mağaza $s$'deki $u$ ürününün $t$ günü itibarıyla son $w$ günlük ortalama günlük satış hızı.

* $w$: Hareketli pencere uzunluğu (3, 7 veya 14 gün).

* $S_{s,u,t-j}$: Mağaza $s$'de $u$ ürününün $(t-j)$ günündeki toplam `OUT` satış adedi.

### 4.4. Stok Karşılama Süresi (Days of Cover — DoC)

Mevcut satış hızıyla, kuyruk dahil efektif stoğun kaç günde eriyeceğini gösterir:

$$
\text{DoC}_{b,t} = \frac{I^{\text{eff}}_{b,t}}{v^{(7)}_{s,u,t} + \epsilon}
$$

**Değişkenler:**

* $\text{DoC}_{b,t}$: Parti $b$'nin (ve önündeki FEFO kuyruğunun) mevcut talep hızıyla tükenmesi için gereken tahmini gün sayısı.

* $I^{\text{eff}}_{b,t}$: Parti $b$'nin $t$ günündeki kuyruk dahil efektif stok adedi.

* $v^{(7)}_{s,u,t}$: İlgili mağaza ve ürünün son 7 günlük ortalama günlük satış hızı.

* $\epsilon$: Sıfıra bölme hatasını önleyen çok küçük pozitif sabittir (örneğin $\epsilon = 10^{-5}$).

### 4.5. Boyutsuz Fire Risk İndeksi (Cover-to-Expiry Ratio)

Modelin en güçlü tekil sinyalidir. $\rho_{b,t} > 1$ olması, mevcut satış hızıyla partinin SKT'ye yetişemeyeceğinin matematiksel göstergesidir:

$$
\rho_{b,t} = \frac{\text{DoC}_{b,t}}{\max(\text{DTE}_{b,t}, 1)}
$$

**Değişkenler:**

* $\rho_{b,t}$: Parti $b$'nin $t$ günündeki boyutsuz fire risk indeksi.

* $\text{DoC}_{b,t}$: Efektif stoğun tükenmesi için gereken gün sayısı.

* $\text{DTE}_{b,t}$: SKT'ye kalan gün sayısı (sıfıra bölünmeyi önlemek için alt sınır $1$ alınır).

### 4.6. Talep İvmesi (Demand Momentum)

Ürüne olan talebin son günlerde hızlanıp hızlanmadığını veya şok yaşayıp yaşamadığını yakalar:

$$
m_{s,u,t} = \frac{v^{(3)}_{s,u,t} + \epsilon}{v^{(14)}_{s,u,t} + \epsilon}
$$

**Değişkenler:**

* $m_{s,u,t}$: Mağaza $s$'deki $u$ ürününün $t$ günündeki talep ivme katsayısı ($m < 1$ yavaşlayan talep, $m > 1$ hızlanan talep).

* $v^{(3)}_{s,u,t}$: Son 3 günlük ortalama satış hızı.

* $v^{(14)}_{s,u,t}$: Son 14 günlük ortalama satış hızı.

* $\epsilon$: Sıfıra bölme hatasını önleyen küçük sabittir.

### 4.7. Ek Bağlamsal Özellikler

* **Ürün ve Kategori Bilgileri:** `category`, `shelf_life_days`, `unit_price`, `store_id`.

* **Takvim Sinyalleri:** Haftanın günü (`day_of_week`), hafta sonu bayrağı (`is_weekend`), kalan pencere içinde hafta sonu olup olmadığı (`weekend_days_left`), ayın maaş dönemi bayrağı.

* **Gecikmeli İndirim (Lagged Discount):** Sızıntıyı önlemek için yalnızca $(t-1)$ ve önceki günlerin ortalama `discount_rate` değeri.

## 5. Veri Sızıntısını Önleme ve Doğrulama Stratejisi

Zaman serisi ve envanter verilerinde klasik rastgele bölme (`train_test_split`) yapay olarak yüksek skor üretir. Bunu engellemek için 3 katı kural uygulanır:

1. **Sağdan Sansürleme (Right-Censoring) Filtresi:**
   Veri setindeki son simülasyon tarihi $T_{\max}$ olmak üzere, yalnızca yaşam döngüsü tamamen kapanmış partiler eğitim ve test setine alınır:
   

   $$
   T_b < T_{\max}
   $$

   
   **Değişkenler:**

   * $T_b$: Parti $b$'nin son tüketim tarihi (`expiry_date`).

   * $T_{\max}$: Veritabanındaki en son gözlem/simülasyon günü.

   $T_b \ge T_{\max}$ koşulundaki aktif partiler eğitimden dışlanır ve **canlı risk tahmini (inference)** seti olarak kullanılır.

2. **Parti Bazlı Zamansal Bölme (Group-Temporal Split):**
   Aynı partiye ait farklı gün kesitleri asla hem eğitim hem test setine bölünmez. Kapanmış partiler giriş tarihine göre sıralanır: ilk $\%75$'lik dönem eğitim (train), sonraki $\%10$'luk dönem doğrulama (validation / eşik optimizasyonu), son $\%15$'lik dönem test (out-of-time test) seti olarak ayrılır.

3. **İndirim Sızıntısı (Discount Leakage) Koruması:**
   Tahmin anında ($t$ günü) simülatörün o gün uyguladığı acil indirim oranı özelliğe doğrudan verilmez; yalnızca geçmiş günlerin indirim ortalaması kullanılır.

## 6. Asimetrik Maliyet ve Karar Eşiği Optimizasyonu

Perakende fire tahmininde iki hata türünün finansal ve sosyal maliyeti eşit değildir:

* **False Negative (FN):** Gerçekte çöpe gidecek bir partiyi kaçırmak (ürün maliyeti + imha maliyeti + karbon ayak izi kaybı).

* **False Positive (FP):** Zaten satılacak bir partiye gereksiz erken indirim yapmak (yalnızca kâr marjından sınırlı kayıp).

Bu nedenle sınıflandırma karar eşiği $\theta$ standart $0.50$ yerine beklenen toplam maliyeti minimize edecek şekilde optimize edilir:

$$
\theta^* = \arg\min_{\theta \in (0,1)} \Big( C_{\text{FN}} \cdot \text{FN}(\theta) + C_{\text{FP}} \cdot \text{FP}(\theta) \Big)
$$

**Değişkenler:**

* $\theta^*$: Optimum sınıflandırma karar eşiği (genellikle $0.30 - 0.40$ bandında).

* $\theta$: Aday olasılık eşiği ($\hat{p}_{b,t} \ge \theta$ ise parti "riskli" işaretlenir).

* $C_{\text{FN}}$: Birim fireyi kaçırmanın (False Negative) maliyeti.

* $\text{FN}(\theta)$: Eşik $\theta$ seçildiğinde kaçırılan fire adedi/sayısı.

* $C_{\text{FP}}$: Birim ürüne gereksiz indirim/aksiyon uygulamanın (False Positive) maliyeti.

* $\text{FP}(\theta)$: Eşik $\theta$ seçildiğinde yanlış alarm verilen güvenli parti sayısı.

### EQUORA Aksiyon Fazlarına Yönlendirme (Faz 2 RL Girdisi)

ML modelinin ürettiği $\hat{p}_{b,t}$, $\hat{q}_{b,t}$ ve $\text{DTE}_{b,t}$ çıktıları partiyi aşağıdaki karar fazlarına yönlendirir:

| Koşul | Atanan Faz | Operasyonel Aksiyon | 
| ----- | ----- | ----- | 
| $\hat{p}_{b,t} < \theta^*$ | `HOLD` | Normal raf satışına devam edilir. | 
| $\hat{p}_{b,t} \ge \theta^*$ ve $2 \le \text{DTE}_{b,t} \le 4$ | `DYNAMIC_DISCOUNT` | Dinamik indirim uygulanır (kâr kurtarma + EQUORA komisyonu). | 
| $\hat{p}_{b,t} \ge \theta^*$ ve $\text{DTE}_{b,t} = 1$ | `SOCIAL_AID` | RL ajanı ürünü en yakın ve en uygun ihtiyaç sahibine (anonim QR) atar. | 
| $\hat{p}_{b,t} \ge \theta^*$ ve $\text{DTE}_{b,t} = 0$ | `URGENT_BULK` | Toplu acil kurtarma (aşevi, KYK yurdu veya barınak yönlendirmesi). | 

## 7. Değerlendirme Metrikleri

Modelin başarısı hem teknik doğruluk hem de iş/ESG etkisi açısından ölçülür:

* **Teknik Metrikler:**

  * *Aşama 1 (Sınıflandırma):* PR-AUC, ROC-AUC ve özellikle $\text{DTE} = 2$ (48 saat kala) seviyesindeki **Recall** ve **Precision**.

  * *Aşama 2 (Regresyon):* Gerçek fire adedi üzerinden MAE, RMSE ve toplam hacim sapması (WAPE):
    

    $$
    \text{WAPE} = \frac{\sum_{i=1}^{N} \vert{}W_i - \hat{W}_i\vert{}}{\sum_{i=1}^{N} W_i}
    $$

    
    **Değişkenler:**

    * $\text{WAPE}$: Ağırlıklı Mutlak Yüzde Hata (Weighted Absolute Percentage Error).

    * $N$: Test setindeki toplam gözlem sayısı.

    * $W_i$: $i$-inci gözlemin gerçekleşen gerçek fire adedi.

    * $\hat{W}_i$: $i$-inci gözlemin model tarafından tahmin edilen fire adedi.

* **İş ve Sosyal Etki (ESG) Metrikleri:**

  * 48 saat önceden yakalanan toplam fire yüzdesi.

  * Kurtarılan ürün adedi ve parasal değer ($\text{TL}$).

  * Önlenen tahmini karbon salımı ($\text{kg } \text{CO}_2$).