"""Faz 2A — Tarihsel Epizot Havuzu Üreticisi (DiscountEpisodeBuilder).

Girdi:
    - `data/ml/features.parquet` (Faz 1 kapanmış partilerinin günlük özellikleri)
    - `CO2ImpactCalculator` (ürün ağırlığı ve kg CO2e / kg katsayısı).

Çıktı:
    - Sızıntısız kronolojik olarak ayrılmış `train_episodes` (%75) ve
      `test_episodes` (%25) `BatchEpisode` listeleri (`EpisodePool`).

Rol:
    - Her benzersiz parti için indirim penceresine giriş anındaki
      (`DTE <= DISCOUNT_START_MAX_DTE` içindeki ilk gözlem günü) gerçek durum
      kesitini Polars `LazyFrame` ile çıkarır; partileri `first_in_date`
      sırasına göre kronolojik bölerek RL ajanının eğitimde hiç görmediği
      gelecek dönem partilerinde (Out-of-Time Test) değerlendirilmesini sağlar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.ml_pipeline.config import MLConfig
from src.ml_pipeline.esg_calculator import CO2ImpactCalculator
from src.rl_pricing.config import RLPricingConfig


@dataclass(frozen=True)
class BatchEpisode:
    """Tek bir ürün partisinin indirim penceresine giriş başlangıç durumu."""

    batch_id: str
    store_id: int
    product_id: int
    product_name: str
    category: str
    category_code: int
    unit_price: float
    shelf_life_days: int
    first_in_date: date
    expiry_date: date
    snapshot_date: date
    days_to_expiry: int
    initial_in_qty: int
    remaining_qty: int
    upstream_fefo_qty: int
    active_batch_count: int
    daily_out_qty: int
    vel_3d: float
    vel_7d: float
    vel_14d: float
    vel_std_7d: float
    lag1_discount_rate: float
    rolling_3d_avg_discount: float
    unit_weight_kg: float
    co2_factor_kg_per_kg: float


@dataclass
class EpisodePool:
    """Kronolojik Train / Out-of-Time Test epizot havuzu ve doğrulama istatistikleri."""

    train_episodes: list[BatchEpisode]
    test_episodes: list[BatchEpisode]
    train_date_range: tuple[date, date]
    test_date_range: tuple[date, date]
    batch_overlap: int


class DiscountEpisodeBuilder:
    """Polars LazyFrame tabanlı, kronolojik Train/Test RL epizot havuzu üreticisi."""

    def __init__(
        self,
        rl_config: RLPricingConfig,
        ml_config: MLConfig | None = None,
    ) -> None:
        self.rl_config = rl_config
        self.ml_config = ml_config or MLConfig()
        self.co2_calculator = CO2ImpactCalculator()

    def build(self) -> EpisodePool:
        """`data/ml/features.parquet` üzerinden indirim penceresi başlangıç epizotlarını çıkarır."""
        lf = pl.scan_parquet(self.ml_config.features_path)

        # İndirim penceresi [DISCOUNT_STOP_DTE .. DISCOUNT_START_MAX_DTE] içindeki
        # her partinin EN ERKEN gözlem gününü (pencereye giriş fotoğrafını) seç
        entry_df = (
            lf.filter(
                (pl.col("days_to_expiry") <= self.rl_config.DISCOUNT_START_MAX_DTE)
                & (pl.col("days_to_expiry") >= self.rl_config.DISCOUNT_STOP_DTE)
                & (pl.col("remaining_qty") > 0)
            )
            .sort(["batch_id", "snapshot_date"])
            .group_by("batch_id", maintain_order=True)
            .first()
            .collect()
        )

        # Her epizota ürün ağırlığı ve CO2 katsayısını ekle (saf Polars)
        entry_df = (
            self.co2_calculator.add_co2_columns(
                entry_df,
                qty_col="remaining_qty",
                output_col="_init_co2_unused",
            )
            .drop("_init_co2_unused")
            .sort(["first_in_date", "batch_id"])
        )

        episodes: list[BatchEpisode] = [
            BatchEpisode(
                batch_id=str(r["batch_id"]),
                store_id=int(r["store_id"]),
                product_id=int(r["product_id"]),
                product_name=str(r["product_name"]),
                category=str(r["category"]),
                category_code=int(r["category_code"]),
                unit_price=float(r["unit_price"]),
                shelf_life_days=int(r["shelf_life_days"]),
                first_in_date=r["first_in_date"],
                expiry_date=r["expiry_date"],
                snapshot_date=r["snapshot_date"],
                days_to_expiry=int(r["days_to_expiry"]),
                initial_in_qty=int(r["initial_in_qty"]),
                remaining_qty=int(r["remaining_qty"]),
                upstream_fefo_qty=int(r["upstream_fefo_qty"]),
                active_batch_count=int(r["active_batch_count"]),
                daily_out_qty=int(r["daily_out_qty"]),
                vel_3d=float(r["vel_3d"]),
                vel_7d=float(r["vel_7d"]),
                vel_14d=float(r["vel_14d"]),
                vel_std_7d=float(r["vel_std_7d"]),
                lag1_discount_rate=float(r["lag1_discount_rate"]),
                rolling_3d_avg_discount=float(r["rolling_3d_avg_discount"]),
                unit_weight_kg=float(r["unit_weight_kg"]),
                co2_factor_kg_per_kg=float(r["co2_factor_kg_per_kg"]),
            )
            for r in entry_df.iter_rows(named=True)
        ]

        n_total = len(episodes)
        if n_total < 20:
            raise ValueError(f"RL eğitimi için yetersiz epizot sayısı: {n_total}")

        split_idx = int(n_total * self.rl_config.TRAIN_EPISODE_RATIO)
        train_eps = episodes[:split_idx]
        test_eps = episodes[split_idx:]

        # Kronolojik sızıntısızlık doğrulaması
        train_ids = {ep.batch_id for ep in train_eps}
        test_ids = {ep.batch_id for ep in test_eps}
        overlap = len(train_ids & test_ids)
        assert overlap == 0, f"RL Train/Test batch_id kesişimi sıfır değil: {overlap}"

        return EpisodePool(
            train_episodes=train_eps,
            test_episodes=test_eps,
            train_date_range=(train_eps[0].first_in_date, train_eps[-1].first_in_date),
            test_date_range=(test_eps[0].first_in_date, test_eps[-1].first_in_date),
            batch_overlap=overlap,
        )
