# EQUORA Faz 2A — RL Dinamik İndirim Motoru Değerlendirme Raporu

## 1. Eğitim ve Ortam Özeti
- **Birincil Ajan:** `MaskablePPO` (`200,000` adım, `295.9 sn`, cihaz: `cpu`)
- **Kıyaslama Ajanı:** `DQN` (`120,000` adım, `162.9 sn`)
- **Kronolojik Epizot Bölmesi:** Train `6,768` parti (`2026-05-29..2026-10-05`) | Out-of-Time Test `2,257` parti (`2026-10-05..2026-11-21`) | Kesişim: `0`
- **İndirim Penceresi:** `DTE 4 -> 2` | Monotonluk Kısıtı: `True` (İhlal: **`0`**) | Faz 1 Kapısı: `True` (`theta* = 0.54`)
- **Birim Ekonomi:** EQUORA Komisyonu `%8` | Artık Stok Cezası `0.35x` | CO2 Gölge Fiyatı `2.50 TL/kg`

## 2. Out-of-Time Test Politika Yarışı (Ortak Rastgele Sayılar — CRN)
| Politika | Brüt Gelir (TL) | Gelir Artışı | EQUORA Komisyon (TL) | Net Ekonomik Değer (TL) | Artık Fire (Adet) | Fire Azalışı | Önlenen CO2e (kg) | Ort. İndirim | Fiyat Değiş./Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NoDiscountPolicy | 6,207,058.50 | +0.00% | 0.00 | 4,314,824.97 | 38,987 | +0.00% | 0.00 | %0.00 | 0.00 |
| StaticRulePolicy | 6,216,140.02 | +0.15% | 125,008.28 | 4,588,820.94 | 35,692 | +8.45% | 12,603.29 | %5.31 | 0.57 |
| Phase1MLFixedDiscountPolicy | 6,236,560.34 | +0.48% | 345,351.62 | 5,054,268.64 | 32,018 | +17.88% | 35,053.06 | %11.81 | 0.89 |
| MyopicExpectedValuePolicy | 5,433,696.54 | -12.46% | 281,072.55 | 4,438,607.67 | 26,671 | +31.59% | 59,581.88 | %24.90 | 0.68 |
| **MaskablePPO** | 6,158,796.02 | -0.78% | 339,130.38 | **4,959,304.07** | 31,898 | +18.18% | 33,812.88 | %12.34 | 0.67 |
| DQN_Benchmark | 6,143,656.76 | -1.02% | 267,999.09 | 4,897,058.04 | 31,325 | +19.65% | 36,527.79 | %12.97 | 0.76 |

## 3. İstatistiksel Anlamlılık (MaskablePPO vs. NoDiscountPolicy — Eşleştirilmiş t-Test)
| Metrik (Epizot Başına Fark) | Ortalama Fark | %95 Güven Aralığı | t-İstatistiği | p-Değeri | Anlamlı (p<0.01) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `revenue_tl` | -21.3835 | [-72.3988, +29.6319] | -0.82 | `4.114e-01` | HAYIR |
| `net_economic_tl` | +285.5468 | [+212.9179, +358.1757] | 7.71 | `1.932e-14` | EVET |
| `waste_units_reduced` | +3.1409 | [+2.8758, +3.4060] | 23.22 | `3.946e-107` | EVET |
| `co2_kg_prevented` | +14.9813 | [+12.3840, +17.5786] | 11.31 | `7.264e-29` | EVET |

## 4. Hassasiyet ve Stres Testleri (Out-of-Time Test)

### SKT Gün Sonuna Kadar İndirim (DISCOUNT_STOP_DTE=0)
| Politika | Brüt Gelir (TL) | Gelir Artışı | EQUORA Komisyon (TL) | Net Ekonomik Değer (TL) | Artık Fire (Adet) | Fire Azalışı | Önlenen CO2e (kg) | Ort. İndirim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NoDiscountPolicy | 9,709,260.10 | +0.00% | 0.00 | 9,042,797.13 | 12,205 | +0.00% | 0.00 | %0.00 |
| StaticRulePolicy | 8,650,848.90 | -10.90% | 304,928.45 | 8,532,682.30 | 6,896 | +43.50% | 22,870.44 | %13.96 |
| Phase1MLFixedDiscountPolicy | 8,658,915.93 | -10.82% | 531,862.24 | 8,803,320.86 | 6,523 | +46.55% | 27,635.69 | %14.24 |
| MyopicExpectedValuePolicy | 8,391,617.54 | -13.57% | 508,194.36 | 8,581,273.99 | 5,535 | +54.65% | 34,497.55 | %15.51 |
| **MaskablePPO** | 8,756,330.94 | -9.81% | 539,655.32 | **8,885,467.16** | 7,033 | +42.38% | 24,071.58 | %13.48 |

### Stratejik Tüketici Yamyamlaşma Stresi (CANNIBALIZATION=0.15)
| Politika | Brüt Gelir (TL) | Gelir Artışı | EQUORA Komisyon (TL) | Net Ekonomik Değer (TL) | Artık Fire (Adet) | Fire Azalışı | Önlenen CO2e (kg) | Ort. İndirim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NoDiscountPolicy | 6,308,177.30 | +0.00% | 0.00 | 4,451,335.36 | 38,423 | +0.00% | 0.00 | %0.00 |
| StaticRulePolicy | 6,291,454.22 | -0.27% | 130,359.56 | 4,701,699.51 | 35,180 | +8.44% | 12,631.45 | %5.26 |
| Phase1MLFixedDiscountPolicy | 6,170,884.59 | -2.18% | 342,322.96 | 4,964,482.68 | 32,698 | +14.90% | 29,913.49 | %11.81 |
| MyopicExpectedValuePolicy | 5,436,874.17 | -13.81% | 282,919.92 | 4,450,269.37 | 27,871 | +27.46% | 57,780.55 | %24.99 |
| **MaskablePPO** | 6,195,253.21 | -1.79% | 344,272.32 | **5,019,394.08** | 32,273 | +16.01% | 32,507.52 | %12.38 |
