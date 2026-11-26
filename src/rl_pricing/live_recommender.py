r"""Faz 2A — Canlı Dinamik İndirim Öneri Servisi (LiveDiscountRecommender).

Girdi:
    - Eğitilmiş `MaskablePPO` modeli (`models/rl_pricing_v1/maskable_ppo.zip`),
    - Dondurulmuş Faz 1 `EquoraHurdleModel` ve `models/hurdle_v1/threshold.json`,
    - Canlı aktif stok özellikleri (`data/ml/active_features.parquet`) ve
      Faz 1 canlı tahminleri (`data/ml/live_predictions.parquet`).

Çıktı:
    - `data/rl/live_discount_recommendations.parquet`: Her aktif parti için
      optimal indirim oranı (`recommended_discount_rate`), beklenen talep artışı,
      beklenen satış, indirim sonrası artık tahmini (`expected_residual_qty` —
      Faz 2B Sosyal Yardım arz sinyali), beklenen gelir, EQUORA komisyonu,
      önlenen CO2e ve devir kanalı (`handoff_channel`).
    - PostgreSQL `equora_db` içinde `rl_discount_recommendations` tablosu
      (mevcut `ml_waste_predictions` tablosuna dokunmadan idempotent yazım).

Rol:
    - Faz 1 tahminlerini ve Faz 2A RL politikasını birleştirerek şube bazlı
      somut aksiyon ve fiyat etiketini üretir; ayrıca ileride bağlanacak Faz 2B
      Sosyal Yardım & Kurum Eşleştirme katmanı için standart veri sözleşmesini
      hazırlar.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import polars as pl
import psycopg2
from psycopg2.extras import execute_values
from sb3_contrib import MaskablePPO

from src.data_pipeline.db_loader import PostgresConnectionConfig
from src.ml_pipeline.config import MLConfig
from src.ml_pipeline.esg_calculator import CO2ImpactCalculator
from src.ml_pipeline.hurdle_model import EquoraHurdleModel
from src.rl_pricing.config import RLPricingConfig
from src.rl_pricing.demand_model import CounterfactualDemandModel
from src.rl_pricing.discount_env import EquoraDiscountEnv
from src.rl_pricing.episode_builder import BatchEpisode


CREATE_RL_RECOMMENDATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rl_discount_recommendations (
    recommendation_id           BIGSERIAL PRIMARY KEY,
    batch_id                    VARCHAR(64) NOT NULL,
    store_id                    INT NOT NULL,
    product_id                  INT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    snapshot_date               DATE NOT NULL,
    expiry_date                 DATE NOT NULL,
    days_to_expiry              INT NOT NULL,
    remaining_qty               INT NOT NULL CHECK (remaining_qty >= 0),
    unit_price                  NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    waste_probability           NUMERIC(6, 4) NOT NULL,
    recommended_action          VARCHAR(24) NOT NULL,
    recommended_discount_rate   NUMERIC(5, 4) NOT NULL CHECK (
        recommended_discount_rate >= 0 AND recommended_discount_rate <= 0.50
    ),
    expected_demand_lift        NUMERIC(6, 4) NOT NULL,
    expected_sales_qty          NUMERIC(10, 2) NOT NULL CHECK (expected_sales_qty >= 0),
    expected_residual_qty       NUMERIC(10, 2) NOT NULL CHECK (expected_residual_qty >= 0),
    expected_revenue_tl         NUMERIC(12, 2) NOT NULL CHECK (expected_revenue_tl >= 0),
    expected_commission_tl      NUMERIC(12, 2) NOT NULL CHECK (expected_commission_tl >= 0),
    expected_prevented_co2_kg   NUMERIC(12, 3) NOT NULL CHECK (expected_prevented_co2_kg >= 0),
    handoff_channel             VARCHAR(24) NOT NULL DEFAULT 'NONE',
    policy_version              VARCHAR(32) NOT NULL DEFAULT 'maskable_ppo_v1',
    recommended_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE rl_discount_recommendations
    ADD COLUMN IF NOT EXISTS handoff_channel VARCHAR(24) NOT NULL DEFAULT 'NONE';

CREATE INDEX IF NOT EXISTS idx_rl_rec_snapshot_action
    ON rl_discount_recommendations (snapshot_date, recommended_action, recommended_discount_rate);

CREATE INDEX IF NOT EXISTS idx_rl_rec_store_prod_exp
    ON rl_discount_recommendations (store_id, product_id, expiry_date);
"""


@dataclass
class LiveDiscountRecommendationResult:
    """Canlı RL indirim öneri servisinin çıktı özeti."""

    recommendations_df: pl.DataFrame
    total_active_batches: int
    discounted_batches: int
    total_expected_revenue_tl: float
    total_expected_commission_tl: float
    total_expected_residual_qty: float
    total_expected_prevented_co2_kg: float
    discount_tier_counts: dict[str, int]
    postgres_rows_written: int


class LiveDiscountRecommender:
    """Eğitilmiş MaskablePPO ile canlı aktif partilere optimal indirim önerici."""

    OUTPUT_COLUMNS: list[str] = [
        "batch_id",
        "store_id",
        "product_id",
        "product_name",
        "category",
        "snapshot_date",
        "expiry_date",
        "days_to_expiry",
        "remaining_qty",
        "unit_price",
        "waste_probability",
        "recommended_action",
        "recommended_discount_rate",
        "expected_demand_lift",
        "expected_sales_qty",
        "expected_residual_qty",
        "expected_revenue_tl",
        "expected_commission_tl",
        "expected_prevented_co2_kg",
        "handoff_channel",
        "policy_version",
        "recommended_at",
    ]

    def __init__(
        self,
        rl_config: RLPricingConfig,
        ml_config: MLConfig | None = None,
    ) -> None:
        self.rl_config = rl_config
        self.ml_config = ml_config or MLConfig()
        self.rl_config.ensure_directories()
        self.co2_calculator = CO2ImpactCalculator()

    def _write_to_postgres(self, rec_df: pl.DataFrame) -> int:
        """PostgreSQL `equora_db` içine `rl_discount_recommendations` tablosunu yazar."""
        if rec_df.height == 0:
            return 0

        db_cfg = PostgresConnectionConfig.from_env(self.rl_config.ENV_PATH)
        conn = psycopg2.connect(**db_cfg.as_kwargs())
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_RL_RECOMMENDATIONS_TABLE_SQL)
                    snap_dates = rec_df["snapshot_date"].unique().to_list()
                    cur.execute(
                        "DELETE FROM rl_discount_recommendations WHERE snapshot_date = ANY(%s);",
                        (snap_dates,),
                    )

                    rows = [
                        (
                            str(r["batch_id"]),
                            int(r["store_id"]),
                            int(r["product_id"]),
                            r["snapshot_date"],
                            r["expiry_date"],
                            int(r["days_to_expiry"]),
                            int(r["remaining_qty"]),
                            round(float(r["unit_price"]), 2),
                            round(float(r["waste_probability"]), 4),
                            str(r["recommended_action"]),
                            round(float(r["recommended_discount_rate"]), 4),
                            round(float(r["expected_demand_lift"]), 4),
                            round(float(r["expected_sales_qty"]), 2),
                            round(float(r["expected_residual_qty"]), 2),
                            round(float(r["expected_revenue_tl"]), 2),
                            round(float(r["expected_commission_tl"]), 2),
                            round(float(r["expected_prevented_co2_kg"]), 3),
                            str(r["handoff_channel"]),
                            str(r["policy_version"]),
                            r["recommended_at"],
                        )
                        for r in rec_df.iter_rows(named=True)
                    ]

                    execute_values(
                        cur,
                        """
                        INSERT INTO rl_discount_recommendations (
                            batch_id,
                            store_id,
                            product_id,
                            snapshot_date,
                            expiry_date,
                            days_to_expiry,
                            remaining_qty,
                            unit_price,
                            waste_probability,
                            recommended_action,
                            recommended_discount_rate,
                            expected_demand_lift,
                            expected_sales_qty,
                            expected_residual_qty,
                            expected_revenue_tl,
                            expected_commission_tl,
                            expected_prevented_co2_kg,
                            handoff_channel,
                            policy_version,
                            recommended_at
                        )
                        VALUES %s;
                        """,
                        rows,
                    )
            return len(rows)
        finally:
            conn.close()

    def recommend(
        self,
        ppo_model: MaskablePPO | None = None,
        hurdle_model: EquoraHurdleModel | None = None,
        demand_model: CounterfactualDemandModel | None = None,
        write_to_postgres: bool | None = None,
    ) -> LiveDiscountRecommendationResult:
        """Canlı aktif partiler için RL indirim oranını, artık stoğu ve CO2 kazancını üretir."""
        model = ppo_model or MaskablePPO.load(str(self.rl_config.ppo_model_path))
        hurdle = hurdle_model or EquoraHurdleModel.load(self.ml_config)
        dm = demand_model or CounterfactualDemandModel(self.rl_config)

        th_payload = json.loads(
            self.ml_config.threshold_path.read_text(encoding="utf-8")
        )
        phase1_theta = float(th_payload["optimal_threshold"])

        # Aktif özellikler + Faz 1 canlı tahminlerini Polars LazyFrame ile birleştir
        active_feat_lf = pl.scan_parquet(self.ml_config.active_features_path)
        live_pred_lf = pl.scan_parquet(self.ml_config.live_predictions_path).select(
            [
                "batch_id",
                "waste_probability",
                "predicted_waste_qty",
                "expected_waste_qty",
                "recommended_action",
                "unit_weight_kg",
                "co2_factor_kg_per_kg",
            ]
        )
        joined_df = (
            active_feat_lf.join(live_pred_lf, on="batch_id", how="inner")
            .sort(["waste_probability", "days_to_expiry"], descending=[True, False])
            .collect()
        )

        # Her aktif partiyi BatchEpisode'a çevir ve EquoraDiscountEnv gözlem/maskesini kullan
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
            for r in joined_df.iter_rows(named=True)
        ]

        live_env = EquoraDiscountEnv(
            episodes=episodes,
            config=self.rl_config,
            demand_model=dm,
            hurdle_model=hurdle,
            phase1_threshold=phase1_theta,
            seed=self.rl_config.SEED,
        )

        rec_at_utc = datetime.now(timezone.utc)
        policy_version = "maskable_ppo_v1"

        out_rows: list[dict[str, Any]] = []
        joined_dicts = joined_df.to_dicts()

        for idx, (ep, raw_row) in enumerate(zip(episodes, joined_dicts)):
            obs, _ = live_env.reset(
                seed=self.rl_config.SEED + idx,
                options={"episode_idx": idx},
            )
            p1_action = str(raw_row["recommended_action"])
            p_hat = float(raw_row["waste_probability"])

            # Yalnızca DYNAMIC_DISCOUNT kanalında VE indirim penceresi içindeyken
            # (DISCOUNT_STOP_DTE <= DTE <= DISCOUNT_START_MAX_DTE, yani 2..4 gün) RL indirim seçer.
            # DTE > 4 olan riskli partiler henüz erken indirim penceresine girmediği için %0 (izleme),
            # SOCIAL_AID / URGENT_BULK -> %0 + doğrudan Faz 2B devri!
            in_discount_window = (
                p1_action == "DYNAMIC_DISCOUNT"
                and self.rl_config.DISCOUNT_STOP_DTE
                <= ep.days_to_expiry
                <= self.rl_config.DISCOUNT_START_MAX_DTE
            )
            if in_discount_window:
                mask = live_env.action_masks()
                act_idx, _ = model.predict(obs, action_masks=mask, deterministic=True)
                discount_rate = float(self.rl_config.DISCOUNT_GRID[int(act_idx)])
            else:
                discount_rate = 0.0

            # Analitik beklenen talep, satış, artık stok ve önlenen CO2e hesabı (güncel gün için)
            is_weekend = ep.snapshot_date.isoweekday() >= 6
            is_payday = ep.snapshot_date.day in (1, 2, 3, 15, 16, 17)
            base_mu = live_env.current_base_mu

            exp_sales_disc, lift_disc = dm.expected_batch_sales(
                base_mu=base_mu,
                category=ep.category,
                discount_rate=discount_rate,
                days_to_expiry=ep.days_to_expiry,
                is_weekend=is_weekend,
                is_payday_window=is_payday,
                remaining_qty=ep.remaining_qty,
                upstream_fefo_qty=ep.upstream_fefo_qty,
                prev_discount_rate=ep.lag1_discount_rate,
            )
            exp_sales_base, _ = dm.expected_batch_sales(
                base_mu=base_mu,
                category=ep.category,
                discount_rate=0.0,
                days_to_expiry=ep.days_to_expiry,
                is_weekend=is_weekend,
                is_payday_window=is_payday,
                remaining_qty=ep.remaining_qty,
                upstream_fefo_qty=ep.upstream_fefo_qty,
                prev_discount_rate=ep.lag1_discount_rate,
            )

            # Pencere kalan günleri boyunca tahmini toplam satış ve Faz 2B artık arz sinyali
            if in_discount_window:
                rem_days = max(1, ep.days_to_expiry - self.rl_config.DISCOUNT_STOP_DTE + 1)
                total_exp_sales = min(float(ep.remaining_qty), exp_sales_disc * rem_days)
                total_base_sales = min(float(ep.remaining_qty), exp_sales_base * rem_days)
                exp_residual = max(0.0, float(ep.remaining_qty) - total_exp_sales)
                extra_rescued_units = max(0.0, total_exp_sales - total_base_sales)
                handoff = "SOCIAL_AID" if exp_residual >= 1.0 else "NONE"
            elif p1_action in ("SOCIAL_AID", "URGENT_BULK"):
                total_exp_sales = min(float(ep.remaining_qty), exp_sales_base)
                exp_residual = max(0.0, float(ep.remaining_qty) - total_exp_sales)
                extra_rescued_units = 0.0
                handoff = p1_action
            else:
                # HOLD veya DTE > DISCOUNT_START_MAX_DTE (henüz indirim penceresine girmemiş parti)
                total_exp_sales = min(float(ep.remaining_qty), exp_sales_base * max(1, ep.days_to_expiry))
                exp_residual = max(0.0, float(raw_row["expected_waste_qty"]))
                extra_rescued_units = 0.0
                handoff = "WAIT_WINDOW" if p1_action == "DYNAMIC_DISCOUNT" else "NONE"

            exp_rev_tl = total_exp_sales * ep.unit_price * (1.0 - discount_rate)
            exp_comm_tl = (
                exp_rev_tl * self.rl_config.EQUORA_COMMISSION_RATE
                if discount_rate > 1e-9
                else 0.0
            )
            unit_co2 = ep.unit_weight_kg * ep.co2_factor_kg_per_kg
            exp_prevented_co2 = extra_rescued_units * unit_co2

            out_rows.append(
                {
                    "batch_id": ep.batch_id,
                    "store_id": ep.store_id,
                    "product_id": ep.product_id,
                    "product_name": ep.product_name,
                    "category": ep.category,
                    "snapshot_date": ep.snapshot_date,
                    "expiry_date": ep.expiry_date,
                    "days_to_expiry": ep.days_to_expiry,
                    "remaining_qty": ep.remaining_qty,
                    "unit_price": round(ep.unit_price, 2),
                    "waste_probability": round(p_hat, 4),
                    "recommended_action": p1_action,
                    "recommended_discount_rate": round(discount_rate, 4),
                    "expected_demand_lift": round(lift_disc, 4),
                    "expected_sales_qty": round(total_exp_sales, 2),
                    "expected_residual_qty": round(exp_residual, 2),
                    "expected_revenue_tl": round(exp_rev_tl, 2),
                    "expected_commission_tl": round(exp_comm_tl, 2),
                    "expected_prevented_co2_kg": round(exp_prevented_co2, 3),
                    "handoff_channel": handoff,
                    "policy_version": policy_version,
                    "recommended_at": rec_at_utc,
                }
            )

        rec_df = (
            pl.DataFrame(out_rows)
            .select(self.OUTPUT_COLUMNS)
            .sort(
                ["recommended_discount_rate", "waste_probability", "expected_revenue_tl"],
                descending=[True, True, True],
            )
        )

        # Kısıt doğrulaması (Plan Kabul Kriteri)
        assert rec_df.filter(pl.col("expected_residual_qty") < 0).height == 0
        assert (
            rec_df.filter(
                (pl.col("recommended_discount_rate") < 0.0)
                | (pl.col("recommended_discount_rate") > 0.50)
            ).height
            == 0
        )

        # Parquet checkpoint'e kaydet
        rec_df.write_parquet(
            self.rl_config.live_recommendations_path,
            compression="zstd",
        )

        # PostgreSQL `rl_discount_recommendations` tablosuna yaz
        should_write = (
            self.rl_config.USE_POSTGRES
            if write_to_postgres is None
            else write_to_postgres
        )
        pg_written = self._write_to_postgres(rec_df) if should_write else 0

        tier_rows = (
            rec_df.group_by("recommended_discount_rate")
            .len()
            .sort("recommended_discount_rate")
            .to_dicts()
        )
        tier_counts = {
            f"%{int(round(float(r['recommended_discount_rate']) * 100))}": int(r["len"])
            for r in tier_rows
        }

        discounted_cnt = rec_df.filter(pl.col("recommended_discount_rate") > 0.0).height
        return LiveDiscountRecommendationResult(
            recommendations_df=rec_df,
            total_active_batches=rec_df.height,
            discounted_batches=discounted_cnt,
            total_expected_revenue_tl=round(float(rec_df["expected_revenue_tl"].sum() or 0.0), 2),
            total_expected_commission_tl=round(float(rec_df["expected_commission_tl"].sum() or 0.0), 2),
            total_expected_residual_qty=round(float(rec_df["expected_residual_qty"].sum() or 0.0), 2),
            total_expected_prevented_co2_kg=round(
                float(rec_df["expected_prevented_co2_kg"].sum() or 0.0), 2
            ),
            discount_tier_counts=tier_counts,
            postgres_rows_written=pg_written,
        )
