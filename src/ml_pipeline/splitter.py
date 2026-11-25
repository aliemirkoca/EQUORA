"""Faz 1 / Adım 3 — Sızıntısız Grup-Zaman Bölmesi (GroupTemporalSplitter).

Girdi:
    - `data/ml/features.parquet` (kapanmış partilerin günlük özellik tablosu)
      veya doğrudan bir Polars `LazyFrame` / `DataFrame`.

Çıktı:
    - Kronolojik parti sırasına göre ayrılmış `train_df`, `val_df`, `test_df`
      Polars `DataFrame`'leri ve bölme doğrulama özetini içeren `SplitResult`.

Rol:
    - Aynı `batch_id`'ye ait tüm günlük zaman kesitlerinin kesinlikle tek bir
      kümede kalmasını sağlayarak parti içi zamansal sızıntıyı (group leakage)
      engeller. Partileri `first_in_date` tarihine göre kronolojik sıralayıp
      `%70 Train / %15 Validation / %15 Test` oranlarıyla böler ve sıfır
      kesişim kuralını `assert` ile doğrular.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.ml_pipeline.config import MLConfig


@dataclass
class SplitStats:
    """Tek bir veri kümesi diliminin (Train / Val / Test) özet istatistikleri."""

    name: str
    batch_count: int
    row_count: int
    first_in_min: date
    first_in_max: date
    snapshot_min: date
    snapshot_max: date
    positive_rate: float


@dataclass
class SplitResult:
    """Grup-zaman bölmesinin tüm çıktıları ve doğrulama metrikleri."""

    train_df: pl.DataFrame
    val_df: pl.DataFrame
    test_df: pl.DataFrame
    train_stats: SplitStats
    val_stats: SplitStats
    test_stats: SplitStats
    overlap_train_val: int
    overlap_train_test: int
    overlap_val_test: int


class GroupTemporalSplitter:
    """Parti (batch_id) bazlı kronolojik %70 / %15 / %15 bölücü."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config

    @staticmethod
    def _compute_stats(name: str, df: pl.DataFrame) -> SplitStats:
        """Bir split için parti sayısı, satır sayısı, tarih aralıkları ve fire oranını hesaplar."""
        agg = df.select(
            pl.col("batch_id").n_unique().alias("batch_count"),
            pl.len().alias("row_count"),
            pl.col("first_in_date").min().alias("first_in_min"),
            pl.col("first_in_date").max().alias("first_in_max"),
            pl.col("snapshot_date").min().alias("snapshot_min"),
            pl.col("snapshot_date").max().alias("snapshot_max"),
            pl.col("will_expire_unsold").mean().alias("positive_rate"),
        ).to_dicts()[0]

        return SplitStats(
            name=name,
            batch_count=int(agg["batch_count"]),
            row_count=int(agg["row_count"]),
            first_in_min=agg["first_in_min"],
            first_in_max=agg["first_in_max"],
            snapshot_min=agg["snapshot_min"],
            snapshot_max=agg["snapshot_max"],
            positive_rate=round(float(agg["positive_rate"] or 0.0), 4),
        )

    def split(self, features_lf: pl.LazyFrame | None = None) -> SplitResult:
        """Özellik tablosunu `batch_id` bazında kronolojik olarak böler ve doğrular."""
        lf = (
            features_lf
            if features_lf is not None
            else pl.scan_parquet(self.config.features_path)
        )

        # 1) Her partiyi tek satıra indirge ve `first_in_date` kronolojik sırasına diz
        #    Eşit giriş tarihlerinde deterministik sıra için `batch_id` ikincil anahtardır.
        batches_df = (
            lf.select(["batch_id", "first_in_date"])
            .unique()
            .sort(["first_in_date", "batch_id"])
            .collect()
        )

        total_batches = batches_df.height
        if total_batches < 10:
            raise ValueError(
                f"Grup-zaman bölmesi için yetersiz parti sayısı: {total_batches}"
            )

        # 2) %70 Train / %15 Validation / %15 Test parti sınır indeksleri
        train_end = int(total_batches * self.config.TRAIN_RATIO)
        val_end = int(total_batches * (self.config.TRAIN_RATIO + self.config.VAL_RATIO))

        batch_list = batches_df["batch_id"].to_list()
        train_batch_ids = batch_list[:train_end]
        val_batch_ids = batch_list[train_end:val_end]
        test_batch_ids = batch_list[val_end:]

        # 3) SIZINTI KORUMASI — Parti kimliği kesişim doğrulaması (KESİNLİKLE 0 OLMALI)
        train_set = set(train_batch_ids)
        val_set = set(val_batch_ids)
        test_set = set(test_batch_ids)

        overlap_tv = len(train_set & val_set)
        overlap_tt = len(train_set & test_set)
        overlap_vt = len(val_set & test_set)

        assert overlap_tv == 0, f"Train-Val batch_id kesişimi sıfır değil: {overlap_tv}"
        assert overlap_tt == 0, f"Train-Test batch_id kesişimi sıfır değil: {overlap_tt}"
        assert overlap_vt == 0, f"Val-Test batch_id kesişimi sıfır değil: {overlap_vt}"
        assert (
            len(train_set) + len(val_set) + len(test_set) == total_batches
        ), "Split toplam parti sayısı kaynak parti sayısına eşit değil!"

        # 4) Parti kimliklerini LazyFrame üzerinde semi-join ile filtrele (sıfır Pandas)
        train_ids_lf = pl.DataFrame({"batch_id": train_batch_ids}).lazy()
        val_ids_lf = pl.DataFrame({"batch_id": val_batch_ids}).lazy()
        test_ids_lf = pl.DataFrame({"batch_id": test_batch_ids}).lazy()

        train_df = lf.join(train_ids_lf, on="batch_id", how="semi").collect()
        val_df = lf.join(val_ids_lf, on="batch_id", how="semi").collect()
        test_df = lf.join(test_ids_lf, on="batch_id", how="semi").collect()

        # Satır düzeyinde de batch_id kesişiminin kesinlikle boş küme olduğunu tekrar doğrula
        df_train_ids = set(train_df["batch_id"].unique().to_list())
        df_val_ids = set(val_df["batch_id"].unique().to_list())
        df_test_ids = set(test_df["batch_id"].unique().to_list())
        assert len(df_train_ids & df_val_ids) == 0, "DataFrame Train-Val kesişimi bulundu!"
        assert len(df_train_ids & df_test_ids) == 0, "DataFrame Train-Test kesişimi bulundu!"
        assert len(df_val_ids & df_test_ids) == 0, "DataFrame Val-Test kesişimi bulundu!"

        train_stats = self._compute_stats("TRAIN", train_df)
        val_stats = self._compute_stats("VAL", val_df)
        test_stats = self._compute_stats("TEST", test_df)

        self.print_summary(train_stats, val_stats, test_stats, overlap_tv, overlap_tt, overlap_vt)

        return SplitResult(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_stats=train_stats,
            val_stats=val_stats,
            test_stats=test_stats,
            overlap_train_val=overlap_tv,
            overlap_train_test=overlap_tt,
            overlap_val_test=overlap_vt,
        )

    @staticmethod
    def print_summary(
        train_stats: SplitStats,
        val_stats: SplitStats,
        test_stats: SplitStats,
        overlap_tv: int,
        overlap_tt: int,
        overlap_vt: int,
    ) -> None:
        """Split tarih aralıklarını ve sızıntı doğrulama sonucunu terminale yazdırır."""
        print("  Grup-Zaman Bölmesi (Batch-Level Chronological Split):")
        for st in (train_stats, val_stats, test_stats):
            print(
                f"    - {st.name:<5}: {st.batch_count:>5,} parti | {st.row_count:>6,} satır | "
                f"first_in: {st.first_in_min} .. {st.first_in_max} | "
                f"snapshot: {st.snapshot_min} .. {st.snapshot_max} | "
                f"fire oranı: %{st.positive_rate * 100:.2f}"
            )
        print(
            f"    - Kesişim Doğrulaması (assert): "
            f"Train∩Val={overlap_tv}, Train∩Test={overlap_tt}, Val∩Test={overlap_vt} [OK]"
        )
