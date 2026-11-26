# EQUORA Faz 1 — Elde Kalacak Ürün Tahmini (Hurdle ML) Değerlendirme Raporu

## 1. Maliyet Duyarlı Eşik Optimizasyonu (Validation Seti)
- **Aktif Ekonomik Karar Eşiği ($\theta^*_{\text{econ}}$):** `0.54` (Aksiyon penceresi `DTE <= 4`, kapsam: `val_full`)
- **Benzersiz Parti Bazlı Ekonomik Eşik:** `0.69` (`1,357` parti)
- **Legacy (`row_all`) Eşiği:** `0.54` (geriye dönük kıyaslama)
- **Ekonomik Birim Katsayıları:** FN = `0.70x` (`COGS + imha`), FP = `0.20x` (`beklenen pencere satışı marj kaybı`)
- **Validation Pencere Hata Maliyeti (`0.50` -> `0.54`):** `695,205.13 TL` -> **`671,020.33 TL`** (Azalma: **%3.48**)

## 2. Test Seti Sınıflandırma Metrikleri

### Tüm Test Seti (`days_to_expiry >= 0`)
| Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standart (`0.50`) | 0.8496 | 0.9061 | 0.7854 | 0.8820 | 0.8309 |
| **Ekonomik Optimal (`0.54`)** | **0.8496** | **0.9061** | **0.8011** | **0.8544** | **0.8269** |

### Aksiyon Penceresi (`days_to_expiry <= 4`, n=5,069) & 48s+ Ufku (`days_to_expiry >= 2`, n=4,935)
| Kapsam / Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Pencere `DTE <= 4` — Standart (`0.50`) | 0.8669 | 0.9279 | 0.8154 | 0.8920 | 0.8520 |
| **Pencere `DTE <= 4` — Optimal (`0.54`)** | **0.8669** | **0.9279** | **0.8289** | **0.8700** | **0.8490** |
| Erken Ufuk `DTE >= 2` — Standart (`0.50`) | 0.8064 | 0.8264 | 0.7285 | 0.8446 | 0.7823 |
| **Erken Ufuk `DTE >= 2` — Optimal (`0.54`)** | **0.8064** | **0.8264** | **0.7466** | **0.8059** | **0.7751** |

## 3. Test Seti Regresyon Metrikleri (Fire Miktarı Tahmini)
- **Seçilen Koşullu Regresör Stratejisi:** `log1p_l2` (Duan Smearing: `1.12412`)

| Kapsam | Satır Sayısı | MAE (Adet) | RMSE (Adet) | WAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| Koşullu Regresör (`final_waste_qty > 0`, $\hat{q}$) | 4,279 | 5.3004 | 8.2779 | %36.94 |
| Birleşik Beklenen Fire (Tümü, $\hat{W} = \hat{p} \times \hat{q}$) | 6,813 | 5.1948 | 8.2328 | %57.64 |

## 4. Test Seti İş ve Jüri Metrikleri (Benzersiz Parti Bazlı)
- **Toplam Benzersiz Test Partisi:** `1,362` (Fire oluşan: `756`)
- **Aksiyon Penceresinde (`2 <= DTE <= 4`) Yakalanan Fire Oranı:**
  - Parti bazlı: **%89.55** (`677` / `756` parti)
  - Ürün adedi bazlı: **%95.19** (`10,112` / `10,623` adet)
  - TL değeri bazlı: **%97.56** (`1,556,264.40 TL` brüt kurtarma potansiyeli)
- **Baseline (Müdahale Yok) Finansal Kayıp:** `1,595,138.10 TL`
- **EQUORA Tahmini Toplam Finansal Kayıp:** `535,016.12 TL`
  - Kaçırılan Fire (FN, %100): `38,873.70 TL`
  - Erken Kurtarma İndirim Marj Maliyeti (TP, %20): `311,252.88 TL`
  - Yanlış Alarm İndirim Marj Maliyeti (FP, %20): `184,889.54 TL`
- **Tahmini Net Finansal Tasarruf:** **`1,060,121.98 TL`** (**%66.46** zarar azalması)

### ESG / Karbon Ayak İzi (CO2e) Etkisi (Test Seti — Benzersiz Partiler)
- **Müdahalesiz (Baseline) Fire Karbon Salımı:** `54,401.67 kg CO2e` (`54.402 ton`)
- **48 Saat+ Erken Yakalama ile Önlenebilir CO2e:** **`53,429.12 kg CO2e`** (`53.429 ton`, **%98.21**)
- **Çevresel Eşdeğerlik:** ~`314,289 km` binek araç sürüşü / ~`2,454` olgun ağacın 1 yıllık CO2 tutumu

| Kategori | Fire Adedi | Baseline CO2e (kg) | 48s+ Önlenebilir CO2e (kg) |
| :--- | :---: | :---: | :---: |
| Et & Tavuk | 2,709 | 40,973.70 | 40,372.80 |
| Süt & Süt Ürünleri | 1,368 | 6,289.35 | 6,210.60 |
| Şarküteri | 427 | 2,513.02 | 2,488.00 |
| Fırın & Unlu Mamul | 2,838 | 1,775.83 | 1,557.06 |
| Meyve & Sebze | 2,119 | 1,665.80 | 1,638.12 |
| Hazır Gıda | 1,162 | 1,183.97 | 1,162.53 |

## 5. Açıklanabilirlik — Top 15 Feature Importance (Gain)

### Aşama 1: Risk Sınıflandırıcı (Top 15)
| Sıra | Özellik | Gain (%) | Split |
| :---: | :--- | :---: | :---: |
| 1 | `risk_index` | %45.69 | 597 |
| 2 | `life_ratio` | %12.01 | 456 |
| 3 | `days_to_expiry` | %9.68 | 230 |
| 4 | `remaining_ratio` | %9.55 | 506 |
| 5 | `days_of_cover` | %2.79 | 202 |
| 6 | `weekend_days_left` | %2.59 | 273 |
| 7 | `unit_price` | %2.09 | 391 |
| 8 | `initial_in_qty` | %2.08 | 377 |
| 9 | `vel_std_7d` | %1.58 | 293 |
| 10 | `vel_14d` | %1.55 | 302 |
| 11 | `vel_7d` | %1.52 | 281 |
| 12 | `days_since_in` | %1.23 | 142 |
| 13 | `shelf_life_days` | %1.18 | 208 |
| 14 | `effective_stock_qty` | %1.04 | 172 |
| 15 | `rolling_3d_avg_discount` | %0.92 | 171 |

### Aşama 2: Koşullu Fire Regresörü (Top 15)
| Sıra | Özellik | Gain (%) | Split |
| :---: | :--- | :---: | :---: |
| 1 | `risk_index` | %24.78 | 591 |
| 2 | `remaining_qty` | %24.75 | 407 |
| 3 | `effective_stock_qty` | %17.10 | 469 |
| 4 | `life_ratio` | %6.12 | 366 |
| 5 | `initial_in_qty` | %5.33 | 486 |
| 6 | `days_to_expiry` | %3.56 | 161 |
| 7 | `vel_7d` | %3.38 | 402 |
| 8 | `vel_std_7d` | %3.00 | 462 |
| 9 | `vel_14d` | %2.13 | 370 |
| 10 | `unit_price` | %1.78 | 406 |
| 11 | `remaining_ratio` | %1.34 | 246 |
| 12 | `weekend_days_left` | %1.09 | 158 |
| 13 | `rolling_3d_avg_discount` | %0.84 | 206 |
| 14 | `demand_momentum` | %0.78 | 204 |
| 15 | `shelf_life_days` | %0.77 | 165 |

## 6. Hurdle vs. Tweedie (`1 < p < 2`) Ablation Kıyaslaması
- **En İyi Tweedie Varyans Üssü ($p^*$):** `1.4` (`best_iter=353`)
- **Tweedie Validation Izgarası:** `p=1.2` -> WAPE `%59.78`, `p=1.4` -> WAPE `%59.52`, `p=1.6` -> WAPE `%61.55`, `p=1.8` -> WAPE `%68.95`

| Metrik (Out-of-Time Test) | EQUORA Hurdle (`p_hat * q_hat`) | Tweedie Benchmark (`p*`) |
| :--- | :---: | :---: |
| Tüm Satırlar MAE (Adet) | 5.1948 | 5.2174 |
| Tüm Satırlar RMSE (Adet) | 8.2328 | 8.2689 |
| **Tüm Satırlar WAPE (%)** | **%57.64** | **%57.89** |
| Sıfır-Fire (`W=0`) Ort. Yanlış Tahmin (Adet) | 4.2929 | 4.2701 |
| Sıfır-Fire (`W=0`) `>= 1 adet` Tahmin Oranı | %73.64 | %70.21 |
| Pozitif-Fire (`W>0`) WAPE (%) | %39.92 | %40.27 |
| Risk Sıralaması ROC-AUC | 0.8496 | 0.8033 |
| Risk Sıralaması PR-AUC | 0.9061 | 0.8590 |

## 7. Aşama 1 Kalibrasyon (Brier & ECE) ve Aşama 2 Regresör Kayıp Kıyaslaması
- **Kalibrasyon Yöntemi:** `isotonic` | **Aktif mi?** `HAYIR (Ham olasılık korundu)`
- **Validation Kontrolü (Brier / ECE):** Ham `0.16122` / `0.03389` -> Kalibre `0.16446` / `0.05778`
- **Test Seti (Brier / ECE 10-bin):** Ham `0.15015` / `0.01143` -> Kalibre **`0.15015` / `0.01143`**

| Aşama 2 Regresör Stratejisi | Val MAE | Val RMSE | Val WAPE (%) | Duan Smearing | Seçildi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `raw_l2` | 5.4239 | 7.7593 | %38.42 | 1.00000 | - |
| `log1p_l2` | 5.2439 | 7.6706 | %37.14 | 1.12412 | **SEÇİLDİ** |
| `raw_l1` | 5.3472 | 7.7777 | %37.88 | 1.00000 | - |
| `poisson` | 5.3697 | 7.7314 | %38.04 | 1.00000 | - |

## 8. Eşik Optimizasyonu Kıyaslaması: Legacy (`row_all`) vs. Ekonomik (`action_window`)
| Optimizasyon Modu | Kapsam | FN / FP Katsayısı | Optimal $\theta^*$ | Val Maliyet (`0.50`) | Val Maliyet ($\theta^*$) | Azalma (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Legacy (`row_all`) | Tüm günler (`DTE >= 0`) | `1.00` / `0.25` (tüm kalan stok) | `0.54` | 2,311,059.23 TL | 2,237,463.80 TL | %3.18 |
| **Ekonomik (`action_window` satır)** | `DTE <= 4` | `0.70` / `0.20` (pencere satışı) | **`0.54`** | 695,205.13 TL | 671,020.33 TL | **%3.48** |
| Ekonomik (`action_window` parti) | `DTE <= 4` (benzersiz parti) | `0.70` / `0.20` (ilk alarm) | `0.69` | 329,636.91 TL | 240,229.40 TL | %27.12 |
