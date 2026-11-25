"""Faz 1 / Adım 1 — Günlük Parti Zaman Kesiti Üreticisi (DailySnapshotBuilder).

Girdi:
    - PostgreSQL `equora_db` veritabanındaki `products` ve `stock_movements`
      tabloları (`MLConfig.USE_POSTGRES = True`) VEYA
      `data/processed/products.parquet` ve `data/processed/stock_movements.parquet`.

Çıktı (Ara Checkpoint'ler):
    - `data/ml/snapshots.parquet`:
        Sağdan sansürlemeden arındırılmış (`expiry_date < max_date`) kapanmış
        partilerin, raf üzerinde aktif kaldığı günlere ait sızıntısız günlük
        zaman kesitleri + kesin gerçekleşen hedef etiketler (`final_waste_qty`,
        `will_expire_unsold`).
    - `data/ml/active_snapshots.parquet`:
        `expiry_date >= max_date` ve son günde (`max_date`) `remaining_qty > 0`
        olan aktif partilerin son gün fotoğrafı (asla eğitime girmez; canlı
        tahmin kümesidir).
    - `data/ml/daily_store_product_sales.parquet`:
        Özellik mühendisliğinde satış hızı (`vel_3d`, `vel_7d`, `vel_14d`) ve
        gecikmeli indirim hesabı için tüm takvim günlerini içeren şube-ürün
        günlük satış ve indirim özeti.

Rol:
    - Her `(store_id, product_id, expiry_date)` partisine deterministik `batch_id`
      atar, zaman damgalarını Türkiye iş günü sınırına (`Europe/Istanbul`) hizalar,
      yalnızca `<= snapshot_date` hareketleriyle gün sonu stok bakiyesini kurar
      ve `remaining_qty < 0` olmadığını garanti eder.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl
import psycopg2

from src.data_pipeline.db_loader import PostgresConnectionConfig
from src.ml_pipeline.config import MLConfig


@dataclass
class SnapshotBuildResult:
    """DailySnapshotBuilder çalıştırma özeti ve doğrulama istatistikleri."""

    source: str
    min_date: date
    max_date: date
    total_batches: int
    closed_batches: int
    active_batches: int
    closed_snapshot_rows: int
    active_snapshot_rows: int
    negative_stock_full_count: int
    negative_stock_closed_count: int
    negative_stock_active_count: int
    closed_positive_waste_rate: float


class DailySnapshotBuilder:
    """Polars LazyFrame tabanlı, sızıntısız günlük parti zaman kesiti üreticisi."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.config.ensure_directories()

    # -------------------------------------------------------------------------
    # Veri Okuma (PostgreSQL veya Parquet — Sıfır Pandas)
    # -------------------------------------------------------------------------
    def _load_from_postgres(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """PostgreSQL `equora_db` üzerinden COPY TO STDOUT ile hızlı Polars okuma yapar."""
        db_cfg = PostgresConnectionConfig.from_env(self.config.ENV_PATH)
        conn = psycopg2.connect(**db_cfg.as_kwargs())
        try:
            with conn.cursor() as cur:
                # 1) products tablosunu CSV tamponu üzerinden Polars'a al
                prod_buf = io.BytesIO()
                cur.copy_expert(
                    """
                    COPY (
                        SELECT
                            product_id,
                            barcode,
                            product_name,
                            category,
                            unit_price::DOUBLE PRECISION AS unit_price,
                            shelf_life_days
                        FROM products
                        ORDER BY product_id
                    ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE);
                    """,
                    prod_buf,
                )
                prod_buf.seek(0)
                products_df = pl.read_csv(
                    prod_buf,
                    schema={
                        "product_id": pl.Int32,
                        "barcode": pl.String,
                        "product_name": pl.String,
                        "category": pl.String,
                        "unit_price": pl.Float64,
                        "shelf_life_days": pl.Int32,
                    },
                )

                # 2) stock_movements tablosunu UTC ISO-8601 formatında dışa aktar
                #    Böylece Parquet yoluyla birebir aynı timezone dönüşümü uygulanır.
                mov_buf = io.BytesIO()
                cur.copy_expert(
                    """
                    COPY (
                        SELECT
                            store_id,
                            product_id,
                            expiry_date,
                            movement_type,
                            quantity,
                            to_char(
                                movement_date AT TIME ZONE 'UTC',
                                'YYYY-MM-DD"T"HH24:MI:SS"Z"'
                            ) AS movement_date,
                            discount_rate::DOUBLE PRECISION AS discount_rate
                        FROM stock_movements
                    ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE);
                    """,
                    mov_buf,
                )
                mov_buf.seek(0)
                movements_df = pl.read_csv(
                    mov_buf,
                    schema_overrides={
                        "store_id": pl.Int32,
                        "product_id": pl.Int32,
                        "expiry_date": pl.Date,
                        "movement_type": pl.String,
                        "quantity": pl.Int32,
                        "movement_date": pl.Datetime(time_unit="us", time_zone="UTC"),
                        "discount_rate": pl.Float64,
                    },
                )
        finally:
            conn.close()

        return products_df, movements_df

    def _load_source_lazy(self) -> tuple[pl.LazyFrame, pl.LazyFrame, str]:
        """Config bayrağına göre PostgreSQL veya Parquet kaynaklarını LazyFrame döndürür."""
        if self.config.USE_POSTGRES:
            products_df, movements_df = self._load_from_postgres()
            return products_df.lazy(), movements_df.lazy(), "PostgreSQL (equora_db)"

        products_lf = pl.scan_parquet(self.config.products_parquet_path)
        movements_lf = pl.scan_parquet(self.config.movements_parquet_path)
        return products_lf, movements_lf, "Parquet (data/processed)"

    # -------------------------------------------------------------------------
    # Günlük Şube-Ürün Satış & İndirim Serisi (Feature Engineer Girdisi)
    # -------------------------------------------------------------------------
    def _build_daily_store_product_sales(
        self,
        movements_norm_lf: pl.LazyFrame,
        min_date: date,
        max_date: date,
    ) -> pl.DataFrame:
        """Tüm takvim günleri için (store_id, product_id, snapshot_date) satış/indirim tablosu kurar.

        Sızıntı ve Eksik Gün Koruması:
            - Satış olmayan günlerde rolling ortalamanın yapay şişmemesi için
              `min_date..max_date` arası eksiksiz günlük takvim ızgarası üretilir.
            - `daily_avg_discount` yalnızca o günün OUT satışlarının adet ağırlıklı
              ortalama indirimidir; FeatureEngineer aşamasında mutlaka `shift(1)` ile
              gecikmeli kullanılacaktır.
        """
        # Mevcut tüm (store_id, product_id) çiftlerini bul
        store_products_lf = (
            movements_norm_lf.select(["store_id", "product_id"]).unique()
        )

        # Eksiksiz takvim günleri LazyFrame'i
        calendar_lf = pl.DataFrame(
            {
                "snapshot_date": pl.date_range(
                    start=min_date,
                    end=max_date,
                    interval="1d",
                    eager=True,
                )
            }
        ).lazy()

        # Her (store_id, product_id) x her takvim günü kartezyen ızgarası
        grid_lf = store_products_lf.join(calendar_lf, how="cross")

        # Günlük toplam satış (OUT) ve adet ağırlıklı ortalama indirim oranı
        daily_out_lf = (
            movements_norm_lf.filter(pl.col("movement_type") == "OUT")
            .group_by(["store_id", "product_id", "movement_day"])
            .agg(
                pl.col("quantity").sum().cast(pl.Float64).alias("store_prod_daily_out"),
                (
                    (pl.col("quantity").cast(pl.Float64) * pl.col("discount_rate")).sum()
                    / pl.col("quantity").sum().cast(pl.Float64)
                ).alias("daily_avg_discount"),
            )
            .rename({"movement_day": "snapshot_date"})
        )

        daily_sales_df = (
            grid_lf.join(
                daily_out_lf,
                on=["store_id", "product_id", "snapshot_date"],
                how="left",
            )
            .with_columns(
                pl.col("store_prod_daily_out").fill_null(0.0),
                pl.col("daily_avg_discount").fill_null(0.0),
            )
            .sort(["store_id", "product_id", "snapshot_date"])
            .collect()
        )
        return daily_sales_df

    # -------------------------------------------------------------------------
    # Ana Snapshot İnşa Akışı
    # -------------------------------------------------------------------------
    def build(self) -> SnapshotBuildResult:
        """Günlük zaman kesitlerini oluşturur, doğrular ve Parquet checkpoint'lere yazar."""
        products_lf, movements_lf, source_label = self._load_source_lazy()

        # 1) Zaman damgalarını Türkiye yerel iş gününe (Europe/Istanbul, UTC+3) çevir.
        #    Önemli: Simülatör 00:01+03'te WASTE yazar (UTC'de bir önceki gün 21:01 görünür).
        #    Yerel iş gününe çevirmeden .dt.date() almak WASTE'i SKT gününe kaydırarak
        #    gün sonu kalan stoğu hatalı sıfırlar!
        movements_norm_lf = movements_lf.with_columns(
            pl.col("store_id").cast(pl.Int32),
            pl.col("product_id").cast(pl.Int32),
            pl.col("expiry_date").cast(pl.Date),
            pl.col("quantity").cast(pl.Int32),
            pl.col("discount_rate").cast(pl.Float64),
            pl.col("movement_date")
            .dt.convert_time_zone(self.config.TIMEZONE)
            .dt.date()
            .alias("movement_day"),
            # Her (store_id, product_id, expiry_date) partisi için okunabilir benzersiz ID
            pl.format(
                "S{}_P{}_{}",
                pl.col("store_id"),
                pl.col("product_id"),
                pl.col("expiry_date").dt.strftime("%Y%m%d"),
            ).alias("batch_id"),
        )

        # Veri setinin kapsadığı ilk ve son iş günü (`max_date` sağdan sansürleme sınırıdır)
        date_bounds = movements_norm_lf.select(
            pl.col("movement_day").min().alias("min_date"),
            pl.col("movement_day").max().alias("max_date"),
        ).collect()
        min_date: date = date_bounds["min_date"][0]
        max_date: date = date_bounds["max_date"][0]

        # 2) Şube-ürün tam takvimli günlük satış & indirim özetini kaydet (FeatureEngineer için)
        daily_sales_df = self._build_daily_store_product_sales(
            movements_norm_lf=movements_norm_lf,
            min_date=min_date,
            max_date=max_date,
        )
        daily_sales_df.write_parquet(self.config.daily_sales_path, compression="zstd")

        # 3) Parti düzeyinde özet ve kesinleşmiş hedefler (targets)
        #    Not: final_waste_qty yalnızca `expiry_date < max_date` (kapanmış) partilerde
        #    eğitim hedefi olarak kullanılır; aktif partilerde hedef bilinmediğinden null tutulur.
        batch_meta_lf = (
            movements_norm_lf.group_by(["batch_id", "store_id", "product_id", "expiry_date"])
            .agg(
                pl.col("movement_day")
                .filter(pl.col("movement_type") == "IN")
                .min()
                .alias("first_in_date"),
                pl.col("quantity")
                .filter(pl.col("movement_type") == "IN")
                .sum()
                .fill_null(0)
                .cast(pl.Int32)
                .alias("total_batch_in_qty"),
                pl.col("quantity")
                .filter(pl.col("movement_type") == "WASTE")
                .sum()
                .fill_null(0)
                .cast(pl.Int32)
                .alias("final_waste_qty"),
            )
            .with_columns(
                # Snapshot takviminin son günü: partinin SKT'si ile veri seti son gününün küçüğü
                pl.min_horizontal(
                    pl.col("expiry_date"),
                    pl.lit(max_date),
                ).alias("snapshot_end_date"),
                # Sağdan sansürleme bayrağı: SKT'si max_date'ten kesin küçük olan partiler kapanmıştır
                (pl.col("expiry_date") < pl.lit(max_date)).alias("is_closed_batch"),
                (pl.col("final_waste_qty") > 0).cast(pl.Int8).alias("will_expire_unsold"),
            )
            .filter(pl.col("first_in_date").is_not_null())
        )

        # 4) Her parti için [first_in_date .. snapshot_end_date] günlük zaman kesitleri üret
        #    Polars `date_ranges` + `explode` ile döngüsüz ve tamamen vektörel genişletme
        batch_calendar_lf = (
            batch_meta_lf.with_columns(
                pl.date_ranges(
                    start=pl.col("first_in_date"),
                    end=pl.col("snapshot_end_date"),
                    interval="1d",
                    closed="both",
                ).alias("snapshot_date")
            )
            .explode("snapshot_date")
        )

        # 5) Parti + Gün bazında yalnızca o gün gerçekleşen IN, OUT ve WASTE toplamları
        daily_batch_mov_lf = (
            movements_norm_lf.group_by(["batch_id", "movement_day"])
            .agg(
                pl.col("quantity")
                .filter(pl.col("movement_type") == "IN")
                .sum()
                .fill_null(0)
                .cast(pl.Int32)
                .alias("daily_in_qty"),
                pl.col("quantity")
                .filter(pl.col("movement_type") == "OUT")
                .sum()
                .fill_null(0)
                .cast(pl.Int32)
                .alias("daily_out_qty"),
                pl.col("quantity")
                .filter(pl.col("movement_type") == "WASTE")
                .sum()
                .fill_null(0)
                .cast(pl.Int32)
                .alias("daily_waste_qty"),
            )
            .rename({"movement_day": "snapshot_date"})
        )

        # 6) Kümülatif bakiye hesabı — YALNIZCA `<= snapshot_date` hareketleri!
        #    `cum_sum().over("batch_id")` sıralı (batch_id, snapshot_date) üzerinde çalıştığı
        #    için gelecekteki hiçbir hareket bugünün stoğuna sızamaz.
        products_clean_lf = products_lf.select(
            pl.col("product_id").cast(pl.Int32),
            pl.col("barcode").cast(pl.String),
            pl.col("product_name").cast(pl.String),
            pl.col("category").cast(pl.String),
            pl.col("unit_price").cast(pl.Float64),
            pl.col("shelf_life_days").cast(pl.Int32),
        )

        full_daily_lf = (
            batch_calendar_lf.join(
                daily_batch_mov_lf,
                on=["batch_id", "snapshot_date"],
                how="left",
            )
            .with_columns(
                pl.col("daily_in_qty").fill_null(0),
                pl.col("daily_out_qty").fill_null(0),
                pl.col("daily_waste_qty").fill_null(0),
            )
            .sort(["batch_id", "snapshot_date"])
            .with_columns(
                # O gün sonuna kadar partiye giren toplam stok (çoğu partide tek IN vardır)
                pl.col("daily_in_qty")
                .cum_sum()
                .over("batch_id")
                .alias("initial_in_qty"),
                # O gün sonuna kadar partiden kümülatif satış
                pl.col("daily_out_qty")
                .cum_sum()
                .over("batch_id")
                .alias("cum_out_qty"),
                # O gün sonuna kadar partiden kümülatif fire (<= expiry_date için 0 olmalıdır)
                pl.col("daily_waste_qty")
                .cum_sum()
                .over("batch_id")
                .alias("cum_waste_qty"),
            )
            .with_columns(
                # Gün sonu rafta kalan gerçek stok miktarı: Giren - Satılan - Fire
                (
                    pl.col("initial_in_qty")
                    - pl.col("cum_out_qty")
                    - pl.col("cum_waste_qty")
                )
                .cast(pl.Int32)
                .alias("remaining_qty"),
                # SKT'ye kalan gün sayısı (expiry_date - snapshot_date)
                (pl.col("expiry_date") - pl.col("snapshot_date"))
                .dt.total_days()
                .cast(pl.Int32)
                .alias("days_to_expiry"),
            )
            .join(products_clean_lf, on="product_id", how="left")
        )

        full_daily_df = full_daily_lf.collect()

        # 7) SANITY CHECK #1: Tüm günlük tarihçe üzerinde `remaining_qty < 0` kontrolü
        neg_full = full_daily_df.filter(pl.col("remaining_qty") < 0).height
        if neg_full != 0:
            raise AssertionError(
                f"Stok tutarlılık hatası: full_daily_df içinde {neg_full} satırda remaining_qty < 0!"
            )

        # 8) Kapanmış Partiler (`expiry_date < max_date`) — Eğitim / Doğrulama / Test Havuzu
        #    Not: Gün sonunda tamamen tükenmiş (`remaining_qty == 0`) günler rafta artık
        #    risk altında ürün kalmadığı ve modele trivial kısayol (rem=0 -> fire=0)
        #    vereceği için filtrelenir; yalnızca aktif raf günleri (`remaining_qty > 0`) tutulur.
        closed_cols = [
            "batch_id",
            "store_id",
            "product_id",
            "barcode",
            "product_name",
            "category",
            "unit_price",
            "shelf_life_days",
            "first_in_date",
            "expiry_date",
            "snapshot_date",
            "days_to_expiry",
            "initial_in_qty",
            "cum_out_qty",
            "daily_out_qty",
            "remaining_qty",
            "final_waste_qty",
            "will_expire_unsold",
        ]
        closed_snapshots_df = (
            full_daily_df.lazy()
            .filter(pl.col("is_closed_batch") & (pl.col("remaining_qty") > 0))
            .select(closed_cols)
            .sort(["snapshot_date", "store_id", "product_id", "expiry_date"])
            .collect()
        )

        # 9) Aktif Partiler (`expiry_date >= max_date` ve son günde `remaining_qty > 0`)
        #    Canlı tahmin kümesi: güncel gün (`max_date`) itibarıyla rafta kalan stok fotoğrafı.
        #    Eğitim hedefleri bu sette yer almaz (sağdan sansürleme koruması).
        active_cols = [
            "batch_id",
            "store_id",
            "product_id",
            "barcode",
            "product_name",
            "category",
            "unit_price",
            "shelf_life_days",
            "first_in_date",
            "expiry_date",
            "snapshot_date",
            "days_to_expiry",
            "initial_in_qty",
            "cum_out_qty",
            "daily_out_qty",
            "remaining_qty",
        ]
        active_snapshots_df = (
            full_daily_df.lazy()
            .filter(
                (~pl.col("is_closed_batch"))
                & (pl.col("snapshot_date") == pl.lit(max_date))
                & (pl.col("remaining_qty") > 0)
            )
            .select(active_cols)
            .sort(["store_id", "product_id", "expiry_date"])
            .collect()
        )

        # 10) SANITY CHECK #2: Çıktı tablolarında sıfır negatif stok & sıfır parti sızıntısı
        neg_closed = closed_snapshots_df.filter(pl.col("remaining_qty") < 0).height
        neg_active = active_snapshots_df.filter(pl.col("remaining_qty") < 0).height
        assert neg_closed == 0, f"Kapanmış snapshot'larda {neg_closed} negatif stok bulundu!"
        assert neg_active == 0, f"Aktif snapshot'larda {neg_active} negatif stok bulundu!"

        closed_ids = set(closed_snapshots_df["batch_id"].unique().to_list())
        active_ids = set(active_snapshots_df["batch_id"].unique().to_list())
        assert len(closed_ids & active_ids) == 0, (
            "Sağdan sansürleme hatası: kapanmış ve aktif partiler kesişiyor!"
        )

        # 11) Ara Çıktıları (Checkpoint) Parquet olarak kaydet
        closed_snapshots_df.write_parquet(self.config.snapshots_path, compression="zstd")
        active_snapshots_df.write_parquet(
            self.config.active_snapshots_path, compression="zstd"
        )

        total_batches = batch_meta_lf.select(pl.len()).collect().item()
        pos_rate = (
            float(closed_snapshots_df["will_expire_unsold"].mean() or 0.0)
            if closed_snapshots_df.height > 0
            else 0.0
        )

        return SnapshotBuildResult(
            source=source_label,
            min_date=min_date,
            max_date=max_date,
            total_batches=int(total_batches),
            closed_batches=len(closed_ids),
            active_batches=len(active_ids),
            closed_snapshot_rows=closed_snapshots_df.height,
            active_snapshot_rows=active_snapshots_df.height,
            negative_stock_full_count=neg_full,
            negative_stock_closed_count=neg_closed,
            negative_stock_active_count=neg_active,
            closed_positive_waste_rate=round(pos_rate, 4),
        )
