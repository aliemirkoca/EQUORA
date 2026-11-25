# EQUORA Faz 1 — Elde Kalacak Ürün Tahmini (Hurdle ML) Değerlendirme Raporu

## 1. Maliyet Duyarlı Eşik Optimizasyonu (Validation Seti)
- **Optimal Karar Eşiği ($\theta^*$):** `0.54` (Standart eşik: `0.50`)
- **Asimetrik Maliyet Katsayıları:** FN = `1.00x` (tam ürün kaybı), FP = `0.25x` (gereksiz indirim marj kaybı)
- **Validation Hata Maliyeti (`0.50`):** `2,311,059.23 TL`
- **Validation Hata Maliyeti (`0.54`):** `2,237,463.80 TL` (Azalma: `%3.18`)

## 2. Test Seti Sınıflandırma Metrikleri (Tüm Test Seti vs. 48 Saat+ Ufku)

### Tüm Test Seti (`days_to_expiry >= 0`)
| Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standart (`0.50`) | 0.8496 | 0.9061 | 0.7854 | 0.8820 | 0.8309 |
| **Optimal (`0.54`)** | **0.8496** | **0.9061** | **0.8011** | **0.8544** | **0.8269** |

### Aksiyon Alınabilir Erken Ufuk (`days_to_expiry >= 2`, n=4,935)
| Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standart (`0.50`) | 0.8064 | 0.8264 | 0.7285 | 0.8446 | 0.7823 |
| **Optimal (`0.54`)** | **0.8064** | **0.8264** | **0.7466** | **0.8059** | **0.7751** |

## 3. Test Seti Regresyon Metrikleri (Fire Miktarı Tahmini)
| Kapsam | Satır Sayısı | MAE (Adet) | RMSE (Adet) | WAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| Koşullu Regresör (`final_waste_qty > 0`, $\hat{q}$) | 4,279 | 5.3538 | 8.1417 | %37.31 |
| Birleşik Beklenen Fire (Tümü, $\hat{W} = \hat{p} \times \hat{q}$) | 6,813 | 5.2889 | 8.1655 | %58.68 |

## 4. Test Seti İş ve Jüri Metrikleri (Benzersiz Parti Bazlı)
- **Toplam Benzersiz Test Partisi:** `1,362` (Fire oluşan: `756`)
- **SKT'ye En Az 2 Gün Kala (48s+) Yakalanan Fire Oranı:**
  - Parti bazlı: **%90.08** (`681` / `756` parti)
  - Ürün adedi bazlı: **%95.35** (`10,129` / `10,623` adet)
  - TL değeri bazlı: **%97.75** (`1,559,247.30 TL` brüt kurtarma potansiyeli)
- **Baseline (Müdahale Yok) Finansal Kayıp:** `1,595,138.10 TL`
- **EQUORA Tahmini Toplam Finansal Kayıp:** `897,335.98 TL`
  - Kaçırılan Fire (FN, %100): `35,890.80 TL`
  - Erken Kurtarma İndirim Marj Maliyeti (TP, %25): `389,811.82 TL`
  - Yanlış Alarm İndirim Marj Maliyeti (FP, %25): `471,633.35 TL`
- **Tahmini Net Finansal Tasarruf:** **`697,802.12 TL`** (**%43.75** zarar azalması)

### ESG / Karbon Ayak İzi (CO2e) Etkisi (Test Seti — Benzersiz Partiler)
- **Müdahalesiz (Baseline) Fire Karbon Salımı:** `54,401.67 kg CO2e` (`54.402 ton`)
- **48 Saat+ Erken Yakalama ile Önlenebilir CO2e:** **`53,513.99 kg CO2e`** (`53.514 ton`, **%98.37**)
- **Çevresel Eşdeğerlik:** ~`314,788 km` binek araç sürüşü / ~`2,458` olgun ağacın 1 yıllık CO2 tutumu

| Kategori | Fire Adedi | Baseline CO2e (kg) | 48s+ Önlenebilir CO2e (kg) |
| :--- | :---: | :---: | :---: |
| Et & Tavuk | 2,709 | 40,973.70 | 40,372.80 |
| Süt & Süt Ürünleri | 1,368 | 6,289.35 | 6,270.45 |
| Şarküteri | 427 | 2,513.02 | 2,513.02 |
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
| 1 | `remaining_qty` | %29.32 | 454 |
| 2 | `risk_index` | %19.87 | 508 |
| 3 | `effective_stock_qty` | %13.59 | 482 |
| 4 | `life_ratio` | %8.54 | 357 |
| 5 | `initial_in_qty` | %7.77 | 351 |
| 6 | `days_to_expiry` | %4.62 | 189 |
| 7 | `vel_7d` | %2.60 | 303 |
| 8 | `vel_std_7d` | %2.47 | 318 |
| 9 | `unit_price` | %1.88 | 326 |
| 10 | `weekend_days_left` | %1.48 | 150 |
| 11 | `vel_14d` | %1.46 | 244 |
| 12 | `remaining_ratio` | %1.24 | 200 |
| 13 | `rolling_3d_avg_discount` | %0.95 | 165 |
| 14 | `days_of_cover` | %0.75 | 149 |
| 15 | `shelf_life_days` | %0.64 | 92 |
