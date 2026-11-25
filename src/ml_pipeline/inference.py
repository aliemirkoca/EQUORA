"""Faz 1 / Adım 6 — Canlı Risk Tahminleyici ve Aksiyon Yönlendirici (LiveRiskPredictor).

Girdi:
    - Kaydedilmiş Hurdle modeli (`models/hurdle_v1/classifier.pkl`, `regressor.pkl`,
      `model_metadata.json`).
    - Optimal karar eşiği (`models/hurdle_v1/threshold.json`).
    - Güncel aktif stok özellik tablosu (`data/ml/active_features.parquet`).

Çıktı:
    - `data/ml/live_predictions.parquet`: Aktif partilerin canlı fire olasılığı,
      tahmini fire adedi, beklenen fire, tahmini TL zararı ve aksiyon önerileri.
    - PostgreSQL `equora_db` içinde `ml_waste_predictions` tablosu (yoksa
      oluşturulur ve güncel `snapshot_date` tahminleri idempotent olarak yazılır).

Rol:
    - Sağdan sansürleme ile eğitimden tamamen izole tutulmuş güncel aktif partiler
      üzerinde canlı risk tahmini yapar ve her partiyi optimal eşik ($\theta^*$)
      ile kalan raf ömrüne (`days_to_expiry`) göre operasyonel aksiyon kanallarına
      (`HOLD`, `DYNAMIC_DISCOUNT`, `SOCIAL_AID`, `URGENT_BULK`) yönlendirir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import polars as pl
import psycopg2
from psycopg2.extras import execute_values

from src.data_pipeline.db_loader import PostgresConnectionConfig
from src.ml_pipeline.config import MLConfig
from src.ml_pipeline.esg_calculator import CO2ImpactCalculator
from src.ml_pipeline.hurdle_model import EquoraHurdleModel


CREATE_PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ml_waste_predictions (
    prediction_id       BIGSERIAL PRIMARY KEY,
    batch_id            VARCHAR(64) NOT NULL,
    store_id            INT NOT NULL,
    product_id          INT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    barcode             VARCHAR(32) NOT NULL,
    product_name        VARCHAR(150) NOT NULL,
    category            VARCHAR(64) NOT NULL,
    snapshot_date       DATE NOT NULL,
    expiry_date         DATE NOT NULL,
    days_to_expiry      INT NOT NULL,
    remaining_qty       INT NOT NULL CHECK (remaining_qty >= 0),
    unit_price          NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    unit_weight_kg      NUMERIC(8, 3) NOT NULL DEFAULT 0.500 CHECK (unit_weight_kg > 0),
    co2_factor_kg_per_kg NUMERIC(8, 3) NOT NULL DEFAULT 0.000 CHECK (co2_factor_kg_per_kg >= 0),
    waste_probability   NUMERIC(6, 4) NOT NULL CHECK (waste_probability >= 0 AND waste_probability <= 1),
    predicted_waste_qty NUMERIC(10, 2) NOT NULL CHECK (predicted_waste_qty >= 0),
    expected_waste_qty  NUMERIC(10, 2) NOT NULL CHECK (expected_waste_qty >= 0),
    estimated_loss_tl   NUMERIC(12, 2) NOT NULL CHECK (estimated_loss_tl >= 0),
    estimated_co2_kg    NUMERIC(12, 3) NOT NULL DEFAULT 0.000 CHECK (estimated_co2_kg >= 0),
    threshold_used      NUMERIC(5, 4) NOT NULL,
    recommended_action  VARCHAR(24) NOT NULL CHECK (
        recommended_action IN ('HOLD', 'DYNAMIC_DISCOUNT', 'SOCIAL_AID', 'URGENT_BULK')
    ),
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Mevcut tabloyu bozmadan ESG/CO2 sütunlarını ekle (şema göçü koruması)
ALTER TABLE ml_waste_predictions
    ADD COLUMN IF NOT EXISTS unit_weight_kg NUMERIC(8, 3) NOT NULL DEFAULT 0.500,
    ADD COLUMN IF NOT EXISTS co2_factor_kg_per_kg NUMERIC(8, 3) NOT NULL DEFAULT 0.000,
    ADD COLUMN IF NOT EXISTS estimated_co2_kg NUMERIC(12, 3) NOT NULL DEFAULT 0.000;

CREATE INDEX IF NOT EXISTS idx_ml_pred_snapshot_action
    ON ml_waste_predictions (snapshot_date, recommended_action);

CREATE INDEX IF NOT EXISTS idx_ml_pred_store_prod_exp
    ON ml_waste_predictions (store_id, product_id, expiry_date);
"""


@dataclass
class LiveInferenceResult:
    """Canlı tahmin çalıştırma özeti, aksiyon ve ESG/CO2 dağılımı."""

    predictions_df: pl.DataFrame
    threshold_used: float
    total_active_batches: int
    action_counts: dict[str, int]
    total_expected_waste_qty: float
    total_estimated_loss_tl: float
    total_estimated_co2_kg: float
    actionable_estimated_co2_kg: float
    co2_equivalents: dict[str, float]
    postgres_rows_written: int


class LiveRiskPredictor:
    """Kaydedilmiş Hurdle modeliyle canlı aktif stok tahmini ve PostgreSQL kayıt sınıfı."""

    OUTPUT_COLUMNS: list[str] = [
        "batch_id",
        "store_id",
        "product_id",
        "barcode",
        "product_name",
        "category",
        "snapshot_date",
        "expiry_date",
        "days_to_expiry",
        "remaining_qty",
        "unit_price",
        "unit_weight_kg",
        "co2_factor_kg_per_kg",
        "waste_probability",
        "predicted_waste_qty",
        "expected_waste_qty",
        "estimated_loss_tl",
        "estimated_co2_kg",
        "threshold_used",
        "recommended_action",
        "predicted_at",
    ]

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.co2_calculator = CO2ImpactCalculator()

    def _load_threshold(self) -> float:
        """`models/hurdle_v1/threshold.json` dosyasından optimal eşiği okur."""
        payload = json.loads(
            self.config.threshold_path.read_text(encoding="utf-8")
        )
        return float(payload["optimal_threshold"])

    def _write_to_postgres(self, predictions_df: pl.DataFrame) -> int:
        """PostgreSQL `equora_db` içine `ml_waste_predictions` tablosunu kurar ve yazar."""
        if predictions_df.height == 0:
            return 0

        db_cfg = PostgresConnectionConfig.from_env(self.config.ENV_PATH)
        conn = psycopg2.connect(**db_cfg.as_kwargs())
        try:
            with conn:
                with conn.cursor() as cur:
                    # 1) Tablo ve indeksler yoksa oluştur
                    cur.execute(CREATE_PREDICTIONS_TABLE_SQL)

                    # 2) Aynı snapshot_date için önceki kayıtları temizle (idempotent yeniden çalıştırma)
                    snap_dates = predictions_df["snapshot_date"].unique().to_list()
                    cur.execute(
                        "DELETE FROM ml_waste_predictions WHERE snapshot_date = ANY(%s);",
                        (snap_dates,),
                    )

                    # 3) Toplu ekleme (execute_values)
                    rows = [
                        (
                            str(r["batch_id"]),
                            int(r["store_id"]),
                            int(r["product_id"]),
                            str(r["barcode"]),
                            str(r["product_name"]),
                            str(r["category"]),
                            r["snapshot_date"],
                            r["expiry_date"],
                            int(r["days_to_expiry"]),
                            int(r["remaining_qty"]),
                            round(float(r["unit_price"]), 2),
                            round(float(r["unit_weight_kg"]), 3),
                            round(float(r["co2_factor_kg_per_kg"]), 3),
                            round(float(r["waste_probability"]), 4),
                            round(float(r["predicted_waste_qty"]), 2),
                            round(float(r["expected_waste_qty"]), 2),
                            round(float(r["estimated_loss_tl"]), 2),
                            round(float(r["estimated_co2_kg"]), 3),
                            round(float(r["threshold_used"]), 4),
                            str(r["recommended_action"]),
                            r["predicted_at"],
                        )
                        for r in predictions_df.select(self.OUTPUT_COLUMNS).iter_rows(named=True)
                    ]

                    execute_values(
                        cur,
                        """
                        INSERT INTO ml_waste_predictions (
                            batch_id,
                            store_id,
                            product_id,
                            barcode,
                            product_name,
                            category,
                            snapshot_date,
                            expiry_date,
                            days_to_expiry,
                            remaining_qty,
                            unit_price,
                            unit_weight_kg,
                            co2_factor_kg_per_kg,
                            waste_probability,
                            predicted_waste_qty,
                            expected_waste_qty,
                            estimated_loss_tl,
                            estimated_co2_kg,
                            threshold_used,
                            recommended_action,
                            predicted_at
                        )
                        VALUES %s;
                        """,
                        rows,
                    )
            return len(rows)
        finally:
            conn.close()

    def predict(self, write_to_postgres: bool | None = None) -> LiveInferenceResult:
        """Aktif partiler üzerinde canlı Hurdle tahmini yapar, Parquet ve PostgreSQL'e kaydeder."""
        model = EquoraHurdleModel.load(self.config)
        theta_opt = self._load_threshold()

        # Güncel aktif partileri LazyFrame ile oku
        active_df = pl.scan_parquet(self.config.active_features_path).collect()
        preds = model.predict_hurdle(active_df, threshold=theta_opt)

        predicted_at_utc = datetime.now(timezone.utc)

        # Polars vektörel karar motoru (Aksiyon yönlendirme kuralları):
        #   - waste_probability < theta*                        -> 'HOLD'
        #   - waste_probability >= theta* & days_to_expiry >= 2 -> 'DYNAMIC_DISCOUNT'
        #   - waste_probability >= theta* & days_to_expiry == 1 -> 'SOCIAL_AID'
        #   - waste_probability >= theta* & days_to_expiry <= 0 -> 'URGENT_BULK'
        predictions_df = (
            active_df.with_columns(
                pl.Series("waste_probability", preds.waste_probability).round(4),
                pl.Series("predicted_waste_qty", preds.predicted_waste_qty).round(2),
                pl.Series("expected_waste_qty", preds.expected_waste_qty).round(2),
                pl.lit(round(theta_opt, 4), dtype=pl.Float64).alias("threshold_used"),
                pl.lit(predicted_at_utc).alias("predicted_at"),
            )
            .with_columns(
                # Beklenen finansal zarar = beklenen fire adedi * birim fiyat (TL)
                (pl.col("expected_waste_qty") * pl.col("unit_price"))
                .round(2)
                .alias("estimated_loss_tl"),
                pl.when(pl.col("waste_probability") < pl.lit(theta_opt))
                .then(pl.lit("HOLD"))
                .when(pl.col("days_to_expiry") >= pl.lit(self.config.DYNAMIC_DISCOUNT_MIN_DTE))
                .then(pl.lit("DYNAMIC_DISCOUNT"))
                .when(pl.col("days_to_expiry") == pl.lit(self.config.SOCIAL_AID_DTE))
                .then(pl.lit("SOCIAL_AID"))
                .otherwise(pl.lit("URGENT_BULK"))
                .alias("recommended_action"),
            )
        )

        # Literatür tabanlı karbon ayak izi (CO2e) sütunlarını ekle (Polars vektörel)
        predictions_df = (
            self.co2_calculator.add_co2_columns(
                predictions_df,
                qty_col="expected_waste_qty",
                output_col="estimated_co2_kg",
            )
            .select(self.OUTPUT_COLUMNS)
            .sort(
                ["waste_probability", "estimated_loss_tl"],
                descending=[True, True],
            )
        )

        # Ara çıktı / canlı tahmin Parquet dosyasını kaydet
        predictions_df.write_parquet(
            self.config.live_predictions_path,
            compression="zstd",
        )

        # PostgreSQL `ml_waste_predictions` tablosuna yaz
        should_write_db = (
            self.config.USE_POSTGRES
            if write_to_postgres is None
            else write_to_postgres
        )
        pg_written = self._write_to_postgres(predictions_df) if should_write_db else 0

        # Aksiyon dağılımı ve ESG/CO2 özeti
        action_rows = (
            predictions_df.group_by("recommended_action")
            .len()
            .sort("recommended_action")
            .to_dicts()
        )
        action_counts = {
            str(r["recommended_action"]): int(r["len"]) for r in action_rows
        }

        total_expected_qty = float(predictions_df["expected_waste_qty"].sum() or 0.0)
        total_est_loss = float(predictions_df["estimated_loss_tl"].sum() or 0.0)
        total_est_co2 = float(predictions_df["estimated_co2_kg"].sum() or 0.0)
        actionable_co2 = float(
            predictions_df.filter(pl.col("recommended_action") != "HOLD")[
                "estimated_co2_kg"
            ].sum()
            or 0.0
        )

        return LiveInferenceResult(
            predictions_df=predictions_df,
            threshold_used=theta_opt,
            total_active_batches=predictions_df.height,
            action_counts=action_counts,
            total_expected_waste_qty=round(total_expected_qty, 2),
            total_estimated_loss_tl=round(total_est_loss, 2),
            total_estimated_co2_kg=round(total_est_co2, 2),
            actionable_estimated_co2_kg=round(actionable_co2, 2),
            co2_equivalents=self.co2_calculator.to_equivalents(actionable_co2),
            postgres_rows_written=pg_written,
        )
