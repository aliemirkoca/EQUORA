"""Faz 1 / Adım 2 — Sızıntısız FEFO & Talep Özellik Mühendisliği (WasteFeatureEngineer).

Girdi:
    - `data/ml/snapshots.parquet` (kapanmış partilerin günlük kesitleri)
    - `data/ml/active_snapshots.parquet` (aktif partilerin son gün fotoğrafı)
    - `data/ml/daily_store_product_sales.parquet` (tam takvimli şube-ürün satış & indirim serisi)

Çıktı (Ara Checkpoint'ler):
    - `data/ml/features.parquet`: Eğitim/Doğrulama/Test havuzu özellik tablosu.
    - `data/ml/active_features.parquet`: Canlı tahmin seti özellik tablosu.
    - `models/hurdle_v1/feature_columns.json`: Modelin tüketeceği kesin özellik
      listesi, kategorik sütunlar ve deterministik kategori kod haritası.

Rol:
    - Hem kapanmış hem aktif snapshot tablolarına BİREBİR AYNI, yalnızca
      `<= snapshot_date` bilgisini kullanan Polars `LazyFrame` dönüşüm hattını
      uygular. FEFO kuyruk fiziği, hareketli satış hızı, boyutsuz risk indeksi,
      gecikmeli (sızıntısız) indirim ve takvim özelliklerini üretir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import polars as pl

from src.ml_pipeline.config import MLConfig


@dataclass
class FeatureEngineeringResult:
    """WasteFeatureEngineer çalışma özeti ve çıktı boyutları."""

    closed_feature_rows: int
    closed_feature_cols: int
    active_feature_rows: int
    active_feature_cols: int
    feature_columns: list[str]
    categorical_features: list[str]
    category_mapping: dict[str, int]


class WasteFeatureEngineer:
    """Polars LazyFrame tabanlı, sıfır veri sızıntılı özellik mühendisliği sınıfı."""

    # Model eğitim ve canlı tahminde kullanılacak kesin sayısal/kategorik özellikler
    FEATURE_COLUMNS: list[str] = [
        # 1) FEFO Kuyruk Fiziği
        "upstream_fefo_qty",
        "effective_stock_qty",
        "active_batch_count",
        # 2) Stok & Raf Ömrü Durumu ve Oranları
        "remaining_qty",
        "initial_in_qty",
        "daily_out_qty",
        "days_to_expiry",
        "remaining_ratio",
        "life_ratio",
        "days_since_in",
        # 3) Satış Hızı ve Talep İvmesi (Store + Product Bazlı)
        "vel_3d",
        "vel_7d",
        "vel_14d",
        "vel_std_7d",
        "demand_momentum",
        # 4) Boyutsuz Risk İndeksi
        "days_of_cover",
        "risk_index",
        # 5) Sızıntısız Gecikmeli İndirim Özellikleri
        "lag1_discount_rate",
        "rolling_3d_avg_discount",
        # 6) Takvim & Periyot Özellikleri
        "day_of_week",
        "is_weekend",
        "is_payday_window",
        "weekend_days_left",
        # 7) Ürün & Mağaza Özellikleri
        "category_code",
        "store_id",
        "unit_price",
        "shelf_life_days",
    ]

    CATEGORICAL_FEATURES: list[str] = ["category_code", "store_id"]

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.config.ensure_directories()

    # -------------------------------------------------------------------------
    # Yardımcı 1: Şube-Ürün Hareketli Satış Hızı ve Gecikmeli İndirim Serisi
    # -------------------------------------------------------------------------
    def _build_rolling_sales_and_discount_lf(
        self,
        daily_sales_lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Tam takvimli şube-ürün serisinden sızıntısız satış hızı ve gecikmeli indirim üretir.

        Sızıntı Koruması:
            - `rolling_mean` / `rolling_std` Polars'ta geriye dönük `[t-w+1 .. t]`
              penceresinde çalışır; gelecekteki günlere (`> snapshot_date`) asla bakmaz.
            - İndirim özellikleri (`lag1_discount_rate`, `rolling_3d_avg_discount`)
              **mutlaka `shift(1)` sonrası** hesaplanır; dolayısıyla simülatörün
              bugün (`t`) verdiği anlık indirim modele KESİNLİKLE sızmaz.
        """
        group_keys = ["store_id", "product_id"]

        return (
            daily_sales_lf.sort(["store_id", "product_id", "snapshot_date"])
            .with_columns(
                # Geriye dönük 3, 7 ve 14 günlük hareketli ortalama satış hızı (adet/gün)
                pl.col("store_prod_daily_out")
                .rolling_mean(window_size=3, min_samples=1)
                .over(group_keys)
                .alias("vel_3d"),
                pl.col("store_prod_daily_out")
                .rolling_mean(window_size=7, min_samples=1)
                .over(group_keys)
                .alias("vel_7d"),
                pl.col("store_prod_daily_out")
                .rolling_mean(window_size=14, min_samples=1)
                .over(group_keys)
                .alias("vel_14d"),
                # Son 7 günlük satış oynaklığı (standart sapma; ilk günde null -> 0.0)
                pl.col("store_prod_daily_out")
                .rolling_std(window_size=7, min_samples=2)
                .over(group_keys)
                .fill_null(0.0)
                .alias("vel_std_7d"),
                # DİKKAT — DISCOUNT LEAKAGE KORUMASI:
                # Önce 1 gün kaydır (`shift(1)`), böylece yalnızca dünkü (t-1) indirim alınır.
                pl.col("daily_avg_discount")
                .shift(1)
                .over(group_keys)
                .fill_null(0.0)
                .alias("lag1_discount_rate"),
                # Son 3 günün gecikmeli [t-3, t-2, t-1] ortalama indirimi (bugün t HARİÇ)
                pl.col("daily_avg_discount")
                .shift(1)
                .rolling_mean(window_size=3, min_samples=1)
                .over(group_keys)
                .fill_null(0.0)
                .alias("rolling_3d_avg_discount"),
            )
            .with_columns(
                # Talep ivmesi (kısa vadeli hızın uzun vadeli hıza oranı; 0 bölmeye karşı +1e-5)
                (pl.col("vel_3d") / (pl.col("vel_14d") + 1e-5)).alias("demand_momentum")
            )
            .select(
                [
                    "store_id",
                    "product_id",
                    "snapshot_date",
                    "vel_3d",
                    "vel_7d",
                    "vel_14d",
                    "vel_std_7d",
                    "demand_momentum",
                    "lag1_discount_rate",
                    "rolling_3d_avg_discount",
                ]
            )
        )

    # -------------------------------------------------------------------------
    # Yardımcı 2: Tek Bir Snapshot LazyFrame'ine Birebir Aynı Özellik Hattı
    # -------------------------------------------------------------------------
    def transform_lazy(
        self,
        snapshots_lf: pl.LazyFrame,
        rolling_sales_lf: pl.LazyFrame,
        category_mapping: dict[str, int],
    ) -> pl.LazyFrame:
        """Verilen snapshot LazyFrame'i üzerine tüm 7 özellik grubunu uygular."""
        # 1) FEFO Kuyruk Fiziği:
        #    Aynı (store_id, product_id, snapshot_date) grubunda her partinin `expiry_date`
        #    değeri benzersizdir. Grupları `expiry_date` artan sıralayıp `remaining_qty`
        #    üzerinde kümülatif toplam alıp kendisini çıkardığımızda, matematiksel olarak
        #    birebir `SUM(other.remaining_qty WHERE other.expiry_date < self.expiry_date)`
        #    elde edilir (O(N log N) vektörel, sıfır self-join maliyeti).
        fefo_keys = ["store_id", "product_id", "snapshot_date"]

        lf = (
            snapshots_lf.sort(["store_id", "product_id", "snapshot_date", "expiry_date"])
            .with_columns(
                (
                    pl.col("remaining_qty").cum_sum().over(fefo_keys)
                    - pl.col("remaining_qty")
                )
                .cast(pl.Int32)
                .alias("upstream_fefo_qty"),
                # Aynı gün rafta bulunan aynı ürüne ait aktif parti sayısı
                pl.len().over(fefo_keys).cast(pl.Int32).alias("active_batch_count"),
            )
            .with_columns(
                # Bu partinin satılabilmesi için önce öndeki FEFO kuyruğunun erimesi gerekir:
                # Etkin eritilmesi gereken stok = kendi kalan stoğu + öndeki daha acil partiler
                (pl.col("remaining_qty") + pl.col("upstream_fefo_qty"))
                .cast(pl.Int32)
                .alias("effective_stock_qty"),
                # 2) Stok & Raf Ömrü Oranları
                (
                    pl.col("remaining_qty").cast(pl.Float64)
                    / pl.col("initial_in_qty").cast(pl.Float64).clip(lower_bound=1.0)
                ).alias("remaining_ratio"),
                (
                    pl.col("days_to_expiry").cast(pl.Float64)
                    / pl.col("shelf_life_days").cast(pl.Float64).clip(lower_bound=1.0)
                ).alias("life_ratio"),
                (pl.col("snapshot_date") - pl.col("first_in_date"))
                .dt.total_days()
                .cast(pl.Int32)
                .alias("days_since_in"),
            )
            # 3) Satış hızı ve gecikmeli indirim serisini (store_id, product_id, snapshot_date) ile bağla
            .join(
                rolling_sales_lf,
                on=["store_id", "product_id", "snapshot_date"],
                how="left",
            )
            .with_columns(
                pl.col("vel_3d").fill_null(0.0),
                pl.col("vel_7d").fill_null(0.0),
                pl.col("vel_14d").fill_null(0.0),
                pl.col("vel_std_7d").fill_null(0.0),
                pl.col("demand_momentum").fill_null(0.0),
                pl.col("lag1_discount_rate").fill_null(0.0),
                pl.col("rolling_3d_avg_discount").fill_null(0.0),
            )
            # 4) Boyutsuz Risk İndeksi
            .with_columns(
                # Etkin stoğun 7 günlük satış hızıyla kaç günde eriyeceği (Days of Cover)
                (
                    pl.col("effective_stock_qty").cast(pl.Float64)
                    / (pl.col("vel_7d") + 1e-5)
                ).alias("days_of_cover")
            )
            .with_columns(
                # Risk İndeksi > 1.0 => mevcut satış hızıyla SKT'ye kadar stok erimeyecek sinyali!
                # days_to_expiry == 0 durumunda 0'a bölmeyi önlemek için alt sınır 0.5 gün alınır.
                (
                    pl.col("days_of_cover")
                    / pl.max_horizontal(
                        pl.col("days_to_expiry").cast(pl.Float64),
                        pl.lit(0.5),
                    )
                ).alias("risk_index")
            )
            # 6) Takvim & Periyot Özellikleri
            .with_columns(
                # Polars ISO weekday: 1=Pazartesi .. 6=Cumartesi, 7=Pazar
                pl.col("snapshot_date").dt.weekday().cast(pl.Int8).alias("day_of_week"),
                (pl.col("snapshot_date").dt.weekday() >= 6)
                .cast(pl.Int8)
                .alias("is_weekend"),
                # Maaş / yoğun alışveriş pencereleri: ayın 1-3 ve 15-17. günleri
                pl.col("snapshot_date")
                .dt.day()
                .is_in([1, 2, 3, 15, 16, 17])
                .cast(pl.Int8)
                .alias("is_payday_window"),
            )
            .with_columns(
                # Bugün (snapshot_date) ile expiry_date arasında kalan satış günlerindeki
                # `(snapshot_date, expiry_date]` Cumartesi + Pazar sayısı (kapalı formül):
                #   n = max(days_to_expiry, 0), full_weeks = n // 7, rem = n % 7
                #   d_sat = ((6 - dow + 6) % 7) + 1  (bir sonraki Cumartesi'ye kalan gün)
                #   d_sun = ((7 - dow + 6) % 7) + 1  (bir sonraki Pazar'a kalan gün)
                #   weekend_days_left = 2 * full_weeks + (rem >= d_sat) + (rem >= d_sun)
                (
                    2 * (pl.col("days_to_expiry").clip(lower_bound=0) // 7)
                    + (
                        (pl.col("days_to_expiry").clip(lower_bound=0) % 7)
                        >= (((6 - pl.col("day_of_week").cast(pl.Int32) + 6) % 7) + 1)
                    ).cast(pl.Int32)
                    + (
                        (pl.col("days_to_expiry").clip(lower_bound=0) % 7)
                        >= (((7 - pl.col("day_of_week").cast(pl.Int32) + 6) % 7) + 1)
                    ).cast(pl.Int32)
                )
                .cast(pl.Int32)
                .alias("weekend_days_left"),
                # 7) Deterministik kategori tamsayı kodlaması
                pl.col("category")
                .replace_strict(category_mapping, default=-1, return_dtype=pl.Int32)
                .alias("category_code"),
            )
        )
        return lf

    # -------------------------------------------------------------------------
    # Ana Özellik Üretim Akışı
    # -------------------------------------------------------------------------
    def build(self) -> FeatureEngineeringResult:
        """Kapanmış ve aktif snapshot'lardan özellik tablolarını üretir ve kaydeder."""
        closed_snap_lf = pl.scan_parquet(self.config.snapshots_path)
        active_snap_lf = pl.scan_parquet(self.config.active_snapshots_path)
        daily_sales_lf = pl.scan_parquet(self.config.daily_sales_path)

        # Deterministik kategori -> integer haritası (alfabetik sıralı)
        categories = (
            pl.concat(
                [
                    closed_snap_lf.select("category"),
                    active_snap_lf.select("category"),
                ]
            )
            .unique()
            .sort("category")
            .collect()["category"]
            .to_list()
        )
        category_mapping: dict[str, int] = {
            str(cat): idx for idx, cat in enumerate(categories)
        }

        # Şube-ürün hareketli satış hızı ve gecikmeli indirim LazyFrame'i
        rolling_sales_lf = self._build_rolling_sales_and_discount_lf(daily_sales_lf)

        # Her iki veri setine BİREBİR AYNI sızıntısız dönüşüm hattını uygula
        closed_features_df = self.transform_lazy(
            snapshots_lf=closed_snap_lf,
            rolling_sales_lf=rolling_sales_lf,
            category_mapping=category_mapping,
        ).collect()

        active_features_df = self.transform_lazy(
            snapshots_lf=active_snap_lf,
            rolling_sales_lf=rolling_sales_lf,
            category_mapping=category_mapping,
        ).collect()

        # Doğrulama 1: FEFO anahtar benzersizliği ve negatif upstream/etkin stok yokluğu
        assert closed_features_df.filter(pl.col("upstream_fefo_qty") < 0).height == 0
        assert active_features_df.filter(pl.col("upstream_fefo_qty") < 0).height == 0

        # Doğrulama 2: Tüm model özelliklerinin eksiksiz ve null/NaN içermediğini kontrol et
        for name, df in (
            ("features.parquet", closed_features_df),
            ("active_features.parquet", active_features_df),
        ):
            missing_cols = [c for c in self.FEATURE_COLUMNS if c not in df.columns]
            if missing_cols:
                raise ValueError(f"{name} içinde eksik özellik sütunları: {missing_cols}")
            null_counts = df.select(
                [pl.col(c).null_count().alias(c) for c in self.FEATURE_COLUMNS]
            ).to_dicts()[0]
            bad_nulls = {k: v for k, v in null_counts.items() if v > 0}
            if bad_nulls:
                raise AssertionError(f"{name} içinde beklenmeyen null değerler: {bad_nulls}")

        # Checkpoint Parquet dosyalarını kaydet
        closed_features_df.write_parquet(self.config.features_path, compression="zstd")
        active_features_df.write_parquet(
            self.config.active_features_path, compression="zstd"
        )

        # Modelin tüketeceği kesin özellik şemasını JSON olarak kaydet
        schema_payload: dict[str, Any] = {
            "feature_columns": self.FEATURE_COLUMNS,
            "categorical_features": self.CATEGORICAL_FEATURES,
            "categorical_feature_indices": [
                self.FEATURE_COLUMNS.index(c) for c in self.CATEGORICAL_FEATURES
            ],
            "category_mapping": category_mapping,
        }
        self.config.feature_columns_path.write_text(
            json.dumps(schema_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return FeatureEngineeringResult(
            closed_feature_rows=closed_features_df.height,
            closed_feature_cols=closed_features_df.width,
            active_feature_rows=active_features_df.height,
            active_feature_cols=active_features_df.width,
            feature_columns=list(self.FEATURE_COLUMNS),
            categorical_features=list(self.CATEGORICAL_FEATURES),
            category_mapping=category_mapping,
        )
