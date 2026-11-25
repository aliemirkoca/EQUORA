"""Faz 1 / ESG Katmanı — Gıda Kaynaklı Karbon Ayak İzi (CO2e) Hesaplayıcısı (CO2ImpactCalculator).

Girdi:
    - `product_name`, `category` ve adet sütunlarını (`expected_waste_qty` veya
      `final_waste_qty`) içeren Polars `DataFrame` / `LazyFrame`.

Çıktı:
    - Her satıra (ürün/partiye) literatür tabanlı tahmini birim ağırlık
      (`unit_weight_kg`), karbon emisyon katsayısı (`co2_factor_kg_per_kg`,
      kg CO2e / kg ürün) ve toplam beklenen/gerçekleşen karbon salımı
      (`estimated_co2_kg` veya `co2_kg`) sütunları eklenmiş Polars tablosu.
    - Jüri/ESG raporları için eşdeğer otomobil km ve olgun ağaç-yıl metrikleri.

Metodoloji ve Referanslar:
    - Emisyon katsayıları (kg CO2e / kg gıda): Poore & Nemecek (2018, Science,
      "Reducing food's environmental impacts through producers and consumers")
      ve FAO Food Wastage Footprint muhafazakâr perakende ortalamalarına dayanır.
    - Ürün birim ağırlığı (`unit_weight_kg`): Önce ürün adından Polars regex ile
      gerçek paket gramajı/hacmi (örn. `500 GR`, `220*3`, `1 LT`, `260GR`) ayrıştırılır;
      gramaj yazmayan ürünlerde kategori/alt-ürün bazlı gerçekçi TR perakende
      varsayılan ağırlıkları kullanılır.
    - Eşdeğerlikler:
      * 1 olgun ağacın yıllık CO2 tutumu: `21.77 kg CO2e / yıl` (US EPA / EEA)
      * Ortalama binek otomobil salımı: `0.17 kg CO2e / km` (küresel karma ortalama)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class ESGReferenceConstants:
    """Akademik literatür (Poore & Nemecek 2018, FAO, EPA) sabitleri."""

    # Eşdeğerlik katsayıları
    TREE_ABSORPTION_KG_CO2_PER_YEAR: float = 21.77
    CAR_EMISSION_KG_CO2_PER_KM: float = 0.17

    # Alt-ürün / kategori bazlı emisyon faktörleri (kg CO2e / kg gıda)
    CO2_BEEF: float = 27.00            # Dana / kırmızı et (süt+besi dengeli muhafazakâr)
    CO2_LAMB: float = 24.00            # Kuzu eti
    CO2_POULTRY: float = 6.90          # Tavuk / piliç
    CO2_PROCESSED_RED_MEAT: float = 18.00   # Dana sucuk / salam / sosis
    CO2_PROCESSED_POULTRY: float = 7.50     # Piliç / hindi şarküteri
    CO2_CHEESE: float = 13.50          # Beyaz peynir / kaşar / lor
    CO2_YOGURT: float = 3.50           # Yoğurt
    CO2_MILK: float = 3.15             # Süt
    CO2_BAKERY: float = 1.57           # Ekmek / unlu mamul
    CO2_NUTS: float = 2.30             # Badem vb. kuruyemiş
    CO2_PRODUCE: float = 0.90          # Taze meyve & sebze
    CO2_READY_MEAL: float = 3.20       # Hazır gıda / sütlü tatlı / meze

    # Ürün adında gramaj yazmayan ürünler için kategori varsayılan birim ağırlığı (kg)
    DEFAULT_WEIGHT_MEAT: float = 1.00
    DEFAULT_WEIGHT_DAIRY_CHEESE: float = 0.50
    DEFAULT_WEIGHT_DAIRY_OTHER: float = 1.00
    DEFAULT_WEIGHT_DELI: float = 0.40
    DEFAULT_WEIGHT_BAKERY: float = 0.40
    DEFAULT_WEIGHT_PRODUCE_HERB: float = 0.25   # Maydanoz, kıvırcık, paket badem
    DEFAULT_WEIGHT_PRODUCE_STD: float = 1.00    # Kg bazlı sebze/meyve
    DEFAULT_WEIGHT_READY_MEAL: float = 0.35


class CO2ImpactCalculator:
    """Polars tabanlı ürün ağırlığı ayrıştırıcı ve karbon ayak izi (CO2e) hesaplayıcı."""

    def __init__(self, constants: ESGReferenceConstants | None = None) -> None:
        self.c = constants or ESGReferenceConstants()

    def _unit_weight_expr(self) -> pl.Expr:
        """Ürün adından (product_name) gramaj/litre bilgisini saf Polars regex ile çıkarır.

        Desteklenen kalıplar:
            - Çarpımlı paketler: `220*3` -> 660 g (0.66 kg)
            - Çoklu gramaj: `4 LÜ ... 110GR` -> 4 * 110 g (0.44 kg)
            - Gram: `250 GR`, `500GR.`, `2000 GR`, `SUCUK90 GR.` -> değer / 1000 kg
            - Litre: `1 LT`, `1LT.`, `1/1 LT` -> 1.0 kg (süt yoğunluğu ~1.03 kg/L)
            - Kilogram: `KG` -> 1.0 kg
            - Gramaj yoksa: kategori + ürün adı tabanlı gerçekçi perakende ağırlığı.
        """
        uname = pl.col("product_name").str.to_uppercase()

        # 1) `220*3` gibi çarpımlı gramaj kalıbı
        mult_g_val = uname.str.extract(r"(\d+)\s*\*\s*(\d+)", 1).cast(pl.Float64)
        mult_g_cnt = uname.str.extract(r"(\d+)\s*\*\s*(\d+)", 2).cast(pl.Float64)
        mult_pack_kg = (mult_g_val * mult_g_cnt) / 1000.0

        # 2) Doğrudan gram (`GR` veya `G`) kalıbı (ör. `500GR`, `250 GR`, `SUCUK90 GR`)
        raw_grams = uname.str.extract(r"(\d+(?:[.,]\d+)?)\s*GR\b", 1).str.replace(",", ".").cast(pl.Float64)

        # Çoklu paket çarpanı (ör. `4 LÜ ... 110GR` -> 4 x 110g)
        pack_mult = (
            uname.str.extract(r"(\d+)\s*(?:LI|Lİ|LU|LÜ)\b", 1)
            .cast(pl.Float64)
            .fill_null(1.0)
        )
        # Eğer gramaj küçük (< 200g) ve çoklu paket etiketi varsa (örn. 4'lü 110gr puding), çarp
        grams_kg = (
            pl.when(raw_grams.is_not_null() & (raw_grams < 200.0) & (pack_mult > 1.0))
            .then((raw_grams * pack_mult) / 1000.0)
            .when(raw_grams.is_not_null())
            .then(raw_grams / 1000.0)
            .otherwise(None)
        )

        # 3) Litre (`1 LT`, `1LT.`, `1/1 LT`) ve mililitre (`ML`) kalıbı
        raw_ml = uname.str.extract(r"(\d+(?:[.,]\d+)?)\s*ML\b", 1).str.replace(",", ".").cast(pl.Float64)
        ml_kg = raw_ml / 1000.0

        raw_lt = uname.str.extract(r"(?:^|\s|/)(\d+(?:[.,]\d+)?)\s*LT\b", 1).str.replace(",", ".").cast(pl.Float64)

        # 4) Explicit `KG` ibaresi (ör. `DANA KIYMALIK KG.`, `GOZDE MACAR SALAM KG`)
        has_kg = uname.str.contains(r"\bKG\b")

        # 5) Kategori + alt ürün varsayılanları
        fallback_kg = (
            pl.when(pl.col("category") == "Et & Tavuk")
            .then(pl.lit(self.c.DEFAULT_WEIGHT_MEAT))
            .when(
                (pl.col("category") == "Süt & Süt Ürünleri")
                & uname.str.contains(r"PEYNIR|PEYNİR|KASAR|KAŞAR|LOR")
            )
            .then(pl.lit(self.c.DEFAULT_WEIGHT_DAIRY_CHEESE))
            .when(pl.col("category") == "Süt & Süt Ürünleri")
            .then(pl.lit(self.c.DEFAULT_WEIGHT_DAIRY_OTHER))
            .when(pl.col("category") == "Şarküteri")
            .then(pl.lit(self.c.DEFAULT_WEIGHT_DELI))
            .when(pl.col("category") == "Fırın & Unlu Mamul")
            .then(pl.lit(self.c.DEFAULT_WEIGHT_BAKERY))
            .when(
                (pl.col("category") == "Meyve & Sebze")
                & uname.str.contains(r"MAYDANOZ|KIVIRCIK|BADEM")
            )
            .then(pl.lit(self.c.DEFAULT_WEIGHT_PRODUCE_HERB))
            .when(pl.col("category") == "Meyve & Sebze")
            .then(pl.lit(self.c.DEFAULT_WEIGHT_PRODUCE_STD))
            .otherwise(pl.lit(self.c.DEFAULT_WEIGHT_READY_MEAL))
        )

        # Öncelik sırası: çarpımlı paket > gram > ml > litre > KG > kategori varsayılanı
        return (
            pl.coalesce(
                [
                    mult_pack_kg,
                    grams_kg,
                    ml_kg,
                    raw_lt,
                    pl.when(has_kg).then(pl.lit(1.0)).otherwise(None),
                    fallback_kg,
                ]
            )
            .clip(lower_bound=0.05, upper_bound=5.0)
            .round(3)
        )

    def _co2_factor_expr(self) -> pl.Expr:
        """Kategori ve alt-ürün türüne göre kg CO2e / kg ürün katsayısını atar."""
        uname = pl.col("product_name").str.to_uppercase()

        return (
            # Et & Tavuk alt kırılımı (Dana vs. Kuzu vs. Piliç/Tavuk — emisyon farkı ~4x!)
            pl.when((pl.col("category") == "Et & Tavuk") & uname.str.contains(r"KUZU"))
            .then(pl.lit(self.c.CO2_LAMB))
            .when(
                (pl.col("category") == "Et & Tavuk")
                & uname.str.contains(r"PILIC|PİLİÇ|SENPILIC|TAVUK|BAGET|KANAT|BONFILE|BUT|CIGER")
            )
            .then(pl.lit(self.c.CO2_POULTRY))
            .when(pl.col("category") == "Et & Tavuk")
            .then(pl.lit(self.c.CO2_BEEF))
            # Şarküteri alt kırılımı (Piliç/Hindi vs. Dana sucuk/salam/sosis)
            .when(
                (pl.col("category") == "Şarküteri")
                & uname.str.contains(r"PILIC|PİLİÇ|SENPILIC|HINDI|HİNDİ")
            )
            .then(pl.lit(self.c.CO2_PROCESSED_POULTRY))
            .when(pl.col("category") == "Şarküteri")
            .then(pl.lit(self.c.CO2_PROCESSED_RED_MEAT))
            # Süt & Süt Ürünleri alt kırılımı (Peynir vs. Yoğurt vs. Süt)
            .when(
                (pl.col("category") == "Süt & Süt Ürünleri")
                & uname.str.contains(r"PEYNIR|PEYNİR|KASAR|KAŞAR|LOR")
            )
            .then(pl.lit(self.c.CO2_CHEESE))
            .when(
                (pl.col("category") == "Süt & Süt Ürünleri")
                & uname.str.contains(r"YOGURT|YOĞURT")
            )
            .then(pl.lit(self.c.CO2_YOGURT))
            .when(pl.col("category") == "Süt & Süt Ürünleri")
            .then(pl.lit(self.c.CO2_MILK))
            # Fırın & Unlu Mamul
            .when(pl.col("category") == "Fırın & Unlu Mamul")
            .then(pl.lit(self.c.CO2_BAKERY))
            # Meyve & Sebze (Badem vb. kuruyemiş vs. taze sebze-meyve)
            .when(
                (pl.col("category") == "Meyve & Sebze")
                & uname.str.contains(r"BADEM|CEVIZ|FINDIK")
            )
            .then(pl.lit(self.c.CO2_NUTS))
            .when(pl.col("category") == "Meyve & Sebze")
            .then(pl.lit(self.c.CO2_PRODUCE))
            # Hazır Gıda
            .otherwise(pl.lit(self.c.CO2_READY_MEAL))
        )

    def add_co2_columns(
        self,
        df: pl.DataFrame,
        qty_col: str = "expected_waste_qty",
        output_col: str = "estimated_co2_kg",
    ) -> pl.DataFrame:
        """Verilen Polars DataFrame'e ağırlık, karbon katsayısı ve toplam kg CO2e sütunu ekler.

        Formül:
            `co2_kg = qty * unit_weight_kg * co2_factor_kg_per_kg`
        """
        return df.with_columns(
            self._unit_weight_expr().alias("unit_weight_kg"),
            self._co2_factor_expr().alias("co2_factor_kg_per_kg"),
        ).with_columns(
            (
                pl.col(qty_col).cast(pl.Float64)
                * pl.col("unit_weight_kg")
                * pl.col("co2_factor_kg_per_kg")
            )
            .round(3)
            .alias(output_col)
        )

    def to_equivalents(self, co2_kg: float) -> dict[str, float]:
        """Toplam kg CO2e değerini ton, araç km ve olgun ağaç-yıl eşdeğerine çevirir."""
        safe_kg = max(0.0, float(co2_kg))
        return {
            "co2_kg": round(safe_kg, 2),
            "co2_tons": round(safe_kg / 1000.0, 3),
            "equivalent_car_km": round(safe_kg / self.c.CAR_EMISSION_KG_CO2_PER_KM, 1),
            "equivalent_tree_years": round(
                safe_kg / self.c.TREE_ABSORPTION_KG_CO2_PER_YEAR, 1
            ),
        }

    def methodology_metadata(self) -> dict[str, Any]:
        """Rapor şeffaflığı için kullanılan literatür katsayılarını sözlük olarak döndürür."""
        return {
            "source": "Poore & Nemecek (2018, Science) + FAO Food Wastage Footprint + US EPA",
            "factors_kg_co2e_per_kg": {
                "Et & Tavuk (Dana)": self.c.CO2_BEEF,
                "Et & Tavuk (Kuzu)": self.c.CO2_LAMB,
                "Et & Tavuk (Tavuk/Piliç)": self.c.CO2_POULTRY,
                "Şarküteri (Kırmızı Et)": self.c.CO2_PROCESSED_RED_MEAT,
                "Şarküteri (Piliç/Hindi)": self.c.CO2_PROCESSED_POULTRY,
                "Süt & Süt Ürünleri (Peynir)": self.c.CO2_CHEESE,
                "Süt & Süt Ürünleri (Yoğurt)": self.c.CO2_YOGURT,
                "Süt & Süt Ürünleri (Süt)": self.c.CO2_MILK,
                "Hazır Gıda": self.c.CO2_READY_MEAL,
                "Fırın & Unlu Mamul": self.c.CO2_BAKERY,
                "Meyve & Sebze (Kuruyemiş)": self.c.CO2_NUTS,
                "Meyve & Sebze (Taze)": self.c.CO2_PRODUCE,
            },
            "tree_absorption_kg_co2_per_year": self.c.TREE_ABSORPTION_KG_CO2_PER_YEAR,
            "car_emission_kg_co2_per_km": self.c.CAR_EMISSION_KG_CO2_PER_KM,
        }
