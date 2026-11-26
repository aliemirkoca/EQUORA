r"""Faz 1 / Adım 5 — Maliyet Duyarlı Eşik Optimizasyonu ve Değerlendirme (ModelEvaluator).

Girdi:
    - Eğitilmiş `EquoraHurdleModel`, opsiyonel `TweedieBenchmarkResult`,
      Validation (`val_df`) ve Test (`test_df`) Polars DataFrame'leri.

Çıktı:
    - `models/hurdle_v1/threshold.json`: Hem ekonomik-tutarlı aksiyon penceresi
      (`DTE <= 4`) optimal karar eşiği (theta*_econ), hem geriye dönük legacy
      (`row_all`) eşiği.
    - `reports/phase1_eval/evaluation_report.json`: Tüm teknik metrikler,
      iş/ESG metrikleri, Top-15 özellik önemleri, Hurdle vs. Tweedie ablation,
      kalibrasyon (Brier/ECE) & regresör kayıp kıyaslaması ve eşik kıyaslaması.
    - `reports/phase1_eval/summary.md`: Jüri/yönetici sunumuna hazır 8 bölümlü
      Türkçe değerlendirme raporu (Markdown).

Rol:
    - Validation üzerinde theta in [0.10, 0.90] aralığını 0.01 adımlarla tarar:
      * Mod 1 (Legacy — `row_all`): Önceki sürümle birebir kıyaslama için korunur.
      * Mod 2 (Ekonomik-Tutarlı — `DTE <= 4`): Birim ekonomi (`COGS + imha = 0.70`)
        ve beklenen pencere satışı (`min(remaining_qty, vel_7d * max(DTE, 1))`)
        üzerinden hem satır hem benzersiz parti bazlı optimal theta*_econ bulur.
    - Test setinde standart `0.50` vs. optimal `theta*` kıyaslamasını, regresyon
      ve sıfır-fire doğruluğunu, Hurdle vs. Tweedie farkını, kalibrasyon kalitesini
      ve benzersiz parti bazlı finansal + CO2e (ESG) etkisini raporlar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.ml_pipeline.config import MLConfig
from src.ml_pipeline.esg_calculator import CO2ImpactCalculator
from src.ml_pipeline.hurdle_model import EquoraHurdleModel
from src.ml_pipeline.tweedie_benchmark import TweedieBenchmarkResult


@dataclass
class EvaluationOutput:
    """ModelEvaluator tam çıktı paketi."""

    optimal_threshold: float
    legacy_threshold: float
    report: dict[str, Any]
    threshold_path: str
    report_json_path: str
    summary_md_path: str


class ModelEvaluator:
    """Maliyet duyarlı eşik optimizasyonu, ablation ve teknik/iş/ESG değerlendiricisi."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.co2_calculator = CO2ImpactCalculator()

    # -------------------------------------------------------------------------
    # Yardımcı 1: Mod 1 (Legacy — Satır Bazlı Tümü) Finansal Hata Maliyeti
    # -------------------------------------------------------------------------
    def _compute_row_financial_error_legacy(
        self,
        y_true: np.ndarray,
        p_hat: np.ndarray,
        final_waste_qty: np.ndarray,
        remaining_qty: np.ndarray,
        unit_price: np.ndarray,
        threshold: float,
    ) -> tuple[float, float, float]:
        """Legacy maliyet matrisi (tüm satırlar, FN=1.0*W*P, FP=0.25*R*P)."""
        y_pred = (p_hat >= threshold).astype(np.int32)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)

        fn_cost_tl = float(
            np.sum(self.config.FN_COST_MULT * final_waste_qty[fn_mask] * unit_price[fn_mask])
        )
        fp_cost_tl = float(
            np.sum(self.config.FP_COST_MULT * remaining_qty[fp_mask] * unit_price[fp_mask])
        )
        return fn_cost_tl + fp_cost_tl, fn_cost_tl, fp_cost_tl

    # -------------------------------------------------------------------------
    # Yardımcı 2: Mod 2 (Ekonomik-Tutarlı — Aksiyon Penceresi DTE <= 4)
    # -------------------------------------------------------------------------
    def _compute_economic_window_row_error(
        self,
        y_true: np.ndarray,
        p_hat: np.ndarray,
        final_waste_qty: np.ndarray,
        expected_window_sales: np.ndarray,
        unit_price: np.ndarray,
        threshold: float,
    ) -> tuple[float, float, float]:
        """Ekonomik-tutarlı satır bazlı maliyet matrisi (DTE <= THRESHOLD_OPT_MAX_DTE).

        Ekonomik Tutarlılık:
            - FN = (UNIT_COST_RATIO + DISPOSAL_COST_RATIO) * final_waste_qty * unit_price
              (varsayılan 0.70 * W * P: gerçek satın alma maliyeti + imha gideri).
            - FP = FP_DISCOUNT_MARGIN_LOSS * min(remaining_qty, vel_7d * max(DTE, 1)) * unit_price
              (yalnızca pencere boyunca gerçekten satılacak kısma gereksiz indirim marj kaybı;
              erken günlerdeki şişirilmiş remaining_qty yanlılığını giderir).
        """
        y_pred = (p_hat >= threshold).astype(np.int32)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)

        fn_mult = self.config.UNIT_COST_RATIO + self.config.DISPOSAL_COST_RATIO
        fp_mult = self.config.FP_DISCOUNT_MARGIN_LOSS

        fn_cost_tl = float(np.sum(fn_mult * final_waste_qty[fn_mask] * unit_price[fn_mask]))
        fp_cost_tl = float(
            np.sum(fp_mult * expected_window_sales[fp_mask] * unit_price[fp_mask])
        )
        return fn_cost_tl + fp_cost_tl, fn_cost_tl, fp_cost_tl

    def _compute_economic_window_batch_error(
        self,
        window_df: pl.DataFrame,
        p_hat_window: np.ndarray,
        threshold: float,
    ) -> tuple[float, float, float]:
        """Benzersiz parti bazlı (sıfır çift sayım) ekonomik-tutarlı maliyet hesabı (DTE <= 4)."""
        fn_mult = self.config.UNIT_COST_RATIO + self.config.DISPOSAL_COST_RATIO
        fp_mult = self.config.FP_DISCOUNT_MARGIN_LOSS

        scored = window_df.with_columns(
            pl.Series("p_hat", p_hat_window),
            pl.min_horizontal(
                pl.col("remaining_qty").cast(pl.Float64),
                pl.col("vel_7d").cast(pl.Float64)
                * pl.max_horizontal(pl.col("days_to_expiry").cast(pl.Float64), pl.lit(1.0)),
            ).alias("expected_window_sales"),
        )

        # Her benzersiz partinin pencere boyunca max risk olasılığı ve ilk alarm anındaki maruziyeti
        batch_agg = (
            scored.sort(["batch_id", "snapshot_date"])
            .group_by("batch_id")
            .agg(
                pl.col("will_expire_unsold").first().alias("y_true"),
                pl.col("final_waste_qty").first().cast(pl.Float64).alias("w_qty"),
                pl.col("unit_price").first().cast(pl.Float64).alias("u_price"),
                (pl.col("p_hat").max() >= threshold).cast(pl.Int8).alias("is_alert"),
                pl.col("expected_window_sales")
                .filter(pl.col("p_hat") >= threshold)
                .first()
                .fill_null(0.0)
                .alias("first_alert_exp_sales"),
            )
        )

        y_t = batch_agg["y_true"].to_numpy().astype(np.int32)
        y_p = batch_agg["is_alert"].to_numpy().astype(np.int32)
        w_q = batch_agg["w_qty"].to_numpy().astype(np.float64)
        u_p = batch_agg["u_price"].to_numpy().astype(np.float64)
        exp_s = batch_agg["first_alert_exp_sales"].to_numpy().astype(np.float64)

        fn_mask = (y_t == 1) & (y_p == 0)
        fp_mask = (y_t == 0) & (y_p == 1)

        fn_cost_tl = float(np.sum(fn_mult * w_q[fn_mask] * u_p[fn_mask]))
        fp_cost_tl = float(np.sum(fp_mult * exp_s[fp_mask] * u_p[fp_mask]))
        return fn_cost_tl + fp_cost_tl, fn_cost_tl, fp_cost_tl

    # -------------------------------------------------------------------------
    # 1) Çift Modlu Eşik Optimizasyonu (Legacy row_all vs. Ekonomik action_window)
    # -------------------------------------------------------------------------
    def optimize_threshold(
        self,
        model: EquoraHurdleModel,
        val_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Validation üzerinde Legacy (`row_all`) ve Ekonomik (`DTE <= 4`) eşikleri tarar.

        Sızıntı Koruması:
            - Kalibratör Validation'ın ilk yarısında (`val_cal`) fit edildiyse, eşik
              taraması kalibratörün hiç görmediği ikinci yarı (`val_thresh`) üzerinde
              yapılır; böylece kalibrasyon->eşik aşırı uyumu engellenir.
        """
        if model.calibrator_active:
            _, val_eval_df = EquoraHurdleModel.split_val_chronologically(val_df)
            val_scope_note = "val_second_half_leakage_free"
        else:
            val_eval_df = val_df
            val_scope_note = "val_full"

        thresholds = np.round(
            np.arange(
                self.config.THRESHOLD_MIN,
                self.config.THRESHOLD_MAX + 1e-9,
                self.config.THRESHOLD_STEP,
            ),
            2,
        )

        # --- MOD 1: Legacy (Tüm Validation satırları, FN=1.0*W*P, FP=0.25*R*P) ---
        p_all = model.predict_proba(val_eval_df)
        y_all = val_eval_df["will_expire_unsold"].to_numpy().astype(np.int32)
        w_all = val_eval_df["final_waste_qty"].to_numpy().astype(np.float64)
        r_all = val_eval_df["remaining_qty"].to_numpy().astype(np.float64)
        u_all = val_eval_df["unit_price"].to_numpy().astype(np.float64)

        legacy_best_theta = 0.50
        legacy_best_loss = float("inf")
        legacy_best_fn = 0.0
        legacy_best_fp = 0.0
        for th in thresholds:
            tot, fn_l, fp_l = self._compute_row_financial_error_legacy(
                y_all, p_all, w_all, r_all, u_all, float(th)
            )
            if tot < legacy_best_loss:
                legacy_best_loss, legacy_best_fn, legacy_best_fp = tot, fn_l, fp_l
                legacy_best_theta = float(th)

        legacy_050_tot, legacy_050_fn, legacy_050_fp = self._compute_row_financial_error_legacy(
            y_all, p_all, w_all, r_all, u_all, 0.50
        )

        # --- MOD 2: Ekonomik-Tutarlı Aksiyon Penceresi (DTE <= THRESHOLD_OPT_MAX_DTE) ---
        win_df = val_eval_df.filter(
            pl.col("days_to_expiry") <= self.config.THRESHOLD_OPT_MAX_DTE
        )
        p_win = model.predict_proba(win_df)
        y_win = win_df["will_expire_unsold"].to_numpy().astype(np.int32)
        w_win = win_df["final_waste_qty"].to_numpy().astype(np.float64)
        r_win = win_df["remaining_qty"].to_numpy().astype(np.float64)
        v_win = win_df["vel_7d"].to_numpy().astype(np.float64)
        dte_win = win_df["days_to_expiry"].to_numpy().astype(np.float64)
        u_win = win_df["unit_price"].to_numpy().astype(np.float64)
        exp_win_sales = np.minimum(r_win, v_win * np.maximum(dte_win, 1.0))

        # 2a) Pencere içi satır bazlı ekonomik tarama
        econ_row_theta = 0.50
        econ_row_loss = float("inf")
        econ_row_fn = 0.0
        econ_row_fp = 0.0
        scan_points: list[dict[str, float]] = []

        for th in thresholds:
            tot, fn_l, fp_l = self._compute_economic_window_row_error(
                y_win, p_win, w_win, exp_win_sales, u_win, float(th)
            )
            scan_points.append(
                {
                    "threshold": float(th),
                    "econ_row_loss_tl": round(tot, 2),
                    "econ_row_fn_tl": round(fn_l, 2),
                    "econ_row_fp_tl": round(fp_l, 2),
                }
            )
            if tot < econ_row_loss:
                econ_row_loss, econ_row_fn, econ_row_fp = tot, fn_l, fp_l
                econ_row_theta = float(th)

        econ_row_050_tot, _, _ = self._compute_economic_window_row_error(
            y_win, p_win, w_win, exp_win_sales, u_win, 0.50
        )

        # 2b) Pencere içi benzersiz parti bazlı ekonomik tarama (erken aksiyon: 2 <= DTE <= 4)
        #     Not: DTE=0..1 dahil edilirse max(p_hat) son gün ~0.997'ye ulaştığından eşik
        #     yapay olarak 0.90'a kayar; gerçek erken aksiyon penceresi 2 <= DTE <= 4'tür.
        early_win_df = win_df.filter(pl.col("days_to_expiry") >= 2)
        p_early_win = model.predict_proba(early_win_df)
        econ_batch_theta = 0.50
        econ_batch_loss = float("inf")
        econ_batch_fn = 0.0
        econ_batch_fp = 0.0
        for th in thresholds:
            tot, fn_l, fp_l = self._compute_economic_window_batch_error(
                early_win_df, p_early_win, float(th)
            )
            if tot < econ_batch_loss:
                econ_batch_loss, econ_batch_fn, econ_batch_fp = tot, fn_l, fp_l
                econ_batch_theta = float(th)

        econ_batch_050_tot, econ_batch_050_fn, econ_batch_050_fp = (
            self._compute_economic_window_batch_error(early_win_df, p_early_win, 0.50)
        )

        # Aktif üretim karar eşiği: Ekonomik-tutarlı pencere satır/parti eşiği
        # (Satır bazlı pencere eşiği günlük snapshot kararıyla birebir örtüştüğü için
        # birincil operasyonel theta* olarak seçilir; parti bazlı theta* de raporlanır.)
        active_theta = econ_row_theta

        threshold_payload: dict[str, Any] = {
            "optimal_threshold": active_theta,
            "optimal_threshold_mode": "economic_action_window_row",
            "default_threshold": 0.50,
            "val_eval_scope": val_scope_note,
            "val_eval_rows_all": int(val_eval_df.height),
            "val_eval_rows_action_window": int(win_df.height),
            "action_window_max_dte": self.config.THRESHOLD_OPT_MAX_DTE,
            # Legacy Mod 1 (geriye dönük kıyaslama)
            "legacy_row_all": {
                "optimal_threshold": legacy_best_theta,
                "fn_cost_mult": self.config.FN_COST_MULT,
                "fp_cost_mult": self.config.FP_COST_MULT,
                "val_optimal_total_loss_tl": round(legacy_best_loss, 2),
                "val_optimal_fn_loss_tl": round(legacy_best_fn, 2),
                "val_optimal_fp_loss_tl": round(legacy_best_fp, 2),
                "val_default_050_total_loss_tl": round(legacy_050_tot, 2),
                "val_default_050_fn_loss_tl": round(legacy_050_fn, 2),
                "val_default_050_fp_loss_tl": round(legacy_050_fp, 2),
                "val_loss_reduction_vs_050_pct": round(
                    100.0 * (legacy_050_tot - legacy_best_loss) / max(legacy_050_tot, 1e-6),
                    2,
                ),
            },
            # Ekonomik Mod 2a (Aksiyon Penceresi DTE <= 4, satır bazlı)
            "economic_action_window_row": {
                "optimal_threshold": econ_row_theta,
                "fn_cost_mult": round(
                    self.config.UNIT_COST_RATIO + self.config.DISPOSAL_COST_RATIO, 4
                ),
                "fp_cost_mult": self.config.FP_DISCOUNT_MARGIN_LOSS,
                "val_optimal_total_loss_tl": round(econ_row_loss, 2),
                "val_optimal_fn_loss_tl": round(econ_row_fn, 2),
                "val_optimal_fp_loss_tl": round(econ_row_fp, 2),
                "val_default_050_total_loss_tl": round(econ_row_050_tot, 2),
                "val_loss_reduction_vs_050_pct": round(
                    100.0 * (econ_row_050_tot - econ_row_loss) / max(econ_row_050_tot, 1e-6),
                    2,
                ),
            },
            # Ekonomik Mod 2b (Erken Aksiyon Penceresi 2 <= DTE <= 4, benzersiz parti bazlı)
            "economic_action_window_batch": {
                "optimal_threshold": econ_batch_theta,
                "unique_batches": int(early_win_df["batch_id"].n_unique()),
                "val_optimal_total_loss_tl": round(econ_batch_loss, 2),
                "val_optimal_fn_loss_tl": round(econ_batch_fn, 2),
                "val_optimal_fp_loss_tl": round(econ_batch_fp, 2),
                "val_default_050_total_loss_tl": round(econ_batch_050_tot, 2),
                "val_default_050_fn_loss_tl": round(econ_batch_050_fn, 2),
                "val_default_050_fp_loss_tl": round(econ_batch_050_fp, 2),
                "val_loss_reduction_vs_050_pct": round(
                    100.0 * (econ_batch_050_tot - econ_batch_loss) / max(econ_batch_050_tot, 1e-6),
                    2,
                ),
            },
            # Geriye dönük uyumluluk alanları (eski okuyucular için)
            "fn_cost_mult": round(
                self.config.UNIT_COST_RATIO + self.config.DISPOSAL_COST_RATIO, 4
            ),
            "fp_cost_mult": self.config.FP_DISCOUNT_MARGIN_LOSS,
            "val_optimal_total_loss_tl": round(econ_row_loss, 2),
            "val_default_050_total_loss_tl": round(econ_row_050_tot, 2),
            "val_loss_reduction_vs_050_pct": round(
                100.0 * (econ_row_050_tot - econ_row_loss) / max(econ_row_050_tot, 1e-6),
                2,
            ),
            "scan_summary": [
                pt
                for pt in scan_points
                if abs((pt["threshold"] * 100) % 5) < 1e-6
                or pt["threshold"] in (econ_row_theta, econ_batch_theta, legacy_best_theta)
            ],
        }

        self.config.threshold_path.write_text(
            json.dumps(threshold_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return threshold_payload

    # -------------------------------------------------------------------------
    # 2) Teknik Metrikler (Sınıflandırma + Regresyon)
    # -------------------------------------------------------------------------
    @staticmethod
    def _classification_metrics(
        y_true: np.ndarray,
        p_hat: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        """Verilen eşikte ROC-AUC, PR-AUC, Precision, Recall ve F1-Score hesaplar."""
        y_pred = (p_hat >= threshold).astype(np.int32)
        return {
            "threshold": round(float(threshold), 2),
            "roc_auc": round(float(roc_auc_score(y_true, p_hat)), 4),
            "pr_auc": round(float(average_precision_score(y_true, p_hat)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """MAE, RMSE ve WAPE (Weighted Absolute Percentage Error) metriklerini hesaplar."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        denom = float(np.sum(np.abs(y_true)))
        wape = float(np.sum(np.abs(y_true - y_pred)) / max(denom, 1e-9))
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "wape": round(wape, 4),
            "wape_pct": round(wape * 100.0, 2),
        }

    # -------------------------------------------------------------------------
    # 3) Test Seti İş / Jüri Metrikleri (Parti Bazlı — Sıfır Çift Sayım)
    # -------------------------------------------------------------------------
    def _compute_business_metrics(
        self,
        test_df: pl.DataFrame,
        p_hat: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        """Test setindeki benzersiz partiler üzerinden 48 saat+ erken yakalama, TL ve CO2e ölçer."""
        scored_df = test_df.with_columns(
            pl.Series("waste_probability", p_hat),
            pl.Series("is_alert", (p_hat >= threshold).astype(np.int8)),
        )

        # Her benzersiz test partisinin gerçek fire, ürün ve fiyat özeti
        batch_truth_df = scored_df.group_by("batch_id").agg(
            pl.col("product_name").first().alias("product_name"),
            pl.col("category").first().alias("category"),
            pl.col("will_expire_unsold").first().alias("will_expire_unsold"),
            pl.col("final_waste_qty").first().cast(pl.Float64).alias("final_waste_qty"),
            pl.col("unit_price").first().cast(pl.Float64).alias("unit_price"),
            pl.col("days_to_expiry").max().alias("max_observed_dte"),
        )
        batch_truth_df = self.co2_calculator.add_co2_columns(
            batch_truth_df,
            qty_col="final_waste_qty",
            output_col="waste_co2_kg",
        )

        # Aksiyon penceresi (2 <= DTE <= THRESHOLD_OPT_MAX_DTE) içinde üretilen ilk erken uyarılar
        early_alerts_df = (
            scored_df.filter(
                (pl.col("days_to_expiry") >= 2)
                & (pl.col("days_to_expiry") <= self.config.THRESHOLD_OPT_MAX_DTE)
                & (pl.col("is_alert") == 1)
            )
            .sort(["batch_id", "snapshot_date"])
            .group_by("batch_id")
            .agg(
                pl.lit(1, dtype=pl.Int8).alias("caught_early_48h"),
                pl.col("days_to_expiry").first().alias("first_alert_dte"),
                pl.col("remaining_qty").first().cast(pl.Float64).alias("first_alert_rem_qty"),
                pl.min_horizontal(
                    pl.col("remaining_qty").first().cast(pl.Float64),
                    pl.col("vel_7d").first().cast(pl.Float64)
                    * pl.max_horizontal(pl.col("days_to_expiry").first().cast(pl.Float64), pl.lit(1.0)),
                ).alias("first_alert_exp_sales"),
            )
        )

        eval_batches_df = (
            batch_truth_df.join(early_alerts_df, on="batch_id", how="left")
            .with_columns(
                pl.col("caught_early_48h").fill_null(0),
                pl.col("first_alert_rem_qty").fill_null(0.0),
                pl.col("first_alert_exp_sales").fill_null(0.0),
            )
            .with_columns(
                (pl.col("final_waste_qty") * pl.col("unit_price")).alias("waste_value_tl")
            )
        )

        waste_batches = eval_batches_df.filter(pl.col("will_expire_unsold") == 1)
        non_waste_batches = eval_batches_df.filter(pl.col("will_expire_unsold") == 0)

        total_test_batches = eval_batches_df.height
        total_waste_batches = waste_batches.height
        total_waste_units = float(waste_batches["final_waste_qty"].sum() or 0.0)
        baseline_loss_tl = float(waste_batches["waste_value_tl"].sum() or 0.0)

        caught_waste = waste_batches.filter(pl.col("caught_early_48h") == 1)
        missed_waste = waste_batches.filter(pl.col("caught_early_48h") == 0)
        false_alarm_batches = non_waste_batches.filter(pl.col("caught_early_48h") == 1)

        caught_waste_batch_count = caught_waste.height
        rescueable_units = float(caught_waste["final_waste_qty"].sum() or 0.0)
        rescueable_gross_tl = float(caught_waste["waste_value_tl"].sum() or 0.0)

        early_catch_batch_pct = 100.0 * caught_waste_batch_count / max(total_waste_batches, 1)
        early_catch_unit_pct = 100.0 * rescueable_units / max(total_waste_units, 1e-6)
        early_catch_tl_pct = 100.0 * rescueable_gross_tl / max(baseline_loss_tl, 1e-6)

        missed_waste_loss_tl = float(missed_waste["waste_value_tl"].sum() or 0.0) * self.config.FN_COST_MULT
        tp_markdown_cost_tl = rescueable_gross_tl * self.config.FP_DISCOUNT_MARGIN_LOSS
        fp_markdown_cost_tl = (
            float((false_alarm_batches["first_alert_exp_sales"] * false_alarm_batches["unit_price"]).sum() or 0.0)
            * self.config.FP_DISCOUNT_MARGIN_LOSS
        )

        equora_total_loss_tl = missed_waste_loss_tl + tp_markdown_cost_tl + fp_markdown_cost_tl
        net_savings_tl = baseline_loss_tl - equora_total_loss_tl
        net_loss_reduction_pct = 100.0 * net_savings_tl / max(baseline_loss_tl, 1e-6)

        dte2_df = scored_df.filter(pl.col("days_to_expiry") == 2)
        dte2_waste_df = dte2_df.filter(pl.col("will_expire_unsold") == 1)
        dte2_catch_pct = (
            100.0 * float(dte2_waste_df["is_alert"].mean() or 0.0)
            if dte2_waste_df.height > 0
            else 0.0
        )

        # ESG / CO2e Etki Hesabı (benzersiz parti bazlı)
        baseline_waste_co2_kg = float(waste_batches["waste_co2_kg"].sum() or 0.0)
        prevented_co2_48h_kg = float(caught_waste["waste_co2_kg"].sum() or 0.0)
        prevented_co2_pct = 100.0 * prevented_co2_48h_kg / max(baseline_waste_co2_kg, 1e-6)
        baseline_co2_eq = self.co2_calculator.to_equivalents(baseline_waste_co2_kg)
        prevented_co2_eq = self.co2_calculator.to_equivalents(prevented_co2_48h_kg)

        cat_co2_rows = (
            waste_batches.group_by("category")
            .agg(
                pl.col("final_waste_qty").sum().alias("waste_units"),
                pl.col("waste_value_tl").sum().round(2).alias("waste_tl"),
                pl.col("waste_co2_kg").sum().round(2).alias("baseline_co2_kg"),
                pl.col("waste_co2_kg")
                .filter(pl.col("caught_early_48h") == 1)
                .sum()
                .fill_null(0.0)
                .round(2)
                .alias("prevented_co2_48h_kg"),
            )
            .sort("baseline_co2_kg", descending=True)
            .to_dicts()
        )

        return {
            "total_unique_test_batches": total_test_batches,
            "total_waste_batches": total_waste_batches,
            "caught_early_48h_waste_batches": caught_waste_batch_count,
            "missed_waste_batches": missed_waste.height,
            "false_positive_batches_48h": false_alarm_batches.height,
            "early_catch_rate_by_batch_pct": round(early_catch_batch_pct, 2),
            "early_catch_rate_by_units_pct": round(early_catch_unit_pct, 2),
            "early_catch_rate_by_tl_pct": round(early_catch_tl_pct, 2),
            "exact_dte2_snapshot_recall_pct": round(dte2_catch_pct, 2),
            "total_test_waste_units": int(round(total_waste_units)),
            "rescueable_units_48h": int(round(rescueable_units)),
            "rescueable_gross_value_tl": round(rescueable_gross_tl, 2),
            "baseline_no_action_loss_tl": round(baseline_loss_tl, 2),
            "equora_missed_waste_fn_loss_tl": round(missed_waste_loss_tl, 2),
            "equora_tp_discount_cost_tl": round(tp_markdown_cost_tl, 2),
            "equora_fp_discount_cost_tl": round(fp_markdown_cost_tl, 2),
            "equora_estimated_total_loss_tl": round(equora_total_loss_tl, 2),
            "equora_net_savings_tl": round(net_savings_tl, 2),
            "equora_loss_reduction_pct": round(net_loss_reduction_pct, 2),
            "esg_co2_impact": {
                "baseline_waste_co2_kg": baseline_co2_eq["co2_kg"],
                "baseline_waste_co2_tons": baseline_co2_eq["co2_tons"],
                "prevented_co2_48h_kg": prevented_co2_eq["co2_kg"],
                "prevented_co2_48h_tons": prevented_co2_eq["co2_tons"],
                "prevented_co2_48h_pct": round(prevented_co2_pct, 2),
                "prevented_equivalent_car_km": prevented_co2_eq["equivalent_car_km"],
                "prevented_equivalent_tree_years": prevented_co2_eq["equivalent_tree_years"],
                "category_breakdown": cat_co2_rows,
                "methodology": self.co2_calculator.methodology_metadata(),
            },
        }

    # -------------------------------------------------------------------------
    # 4) Ana Değerlendirme ve 8 Bölümlü Raporlama Akışı
    # -------------------------------------------------------------------------
    def evaluate(
        self,
        model: EquoraHurdleModel,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        tweedie_result: TweedieBenchmarkResult | None = None,
    ) -> EvaluationOutput:
        """Validation eşik optimizasyonu, Test değerlendirmesi ve Ablation raporlarını üretir."""
        threshold_info = self.optimize_threshold(model=model, val_df=val_df)
        theta_opt = float(threshold_info["optimal_threshold"])
        theta_legacy = float(threshold_info["legacy_row_all"]["optimal_threshold"])

        test_preds = model.predict_hurdle(test_df, threshold=theta_opt)
        y_cls_test = test_df["will_expire_unsold"].to_numpy().astype(np.int32)
        y_qty_test = test_df["final_waste_qty"].to_numpy().astype(np.float64)
        rem_test = test_df["remaining_qty"].to_numpy().astype(np.float64)
        vel_test = test_df["vel_7d"].to_numpy().astype(np.float64)
        price_test = test_df["unit_price"].to_numpy().astype(np.float64)
        dte_test = test_df["days_to_expiry"].to_numpy().astype(np.int32)

        # Sınıflandırma metrikleri (tüm test + DTE >= 2 + aksiyon penceresi DTE <= 4)
        cls_default = self._classification_metrics(y_cls_test, test_preds.waste_probability, 0.50)
        cls_optimal = self._classification_metrics(y_cls_test, test_preds.waste_probability, theta_opt)
        cls_legacy = self._classification_metrics(y_cls_test, test_preds.waste_probability, theta_legacy)

        dte2_mask = dte_test >= 2
        cls_dte2_default = self._classification_metrics(
            y_cls_test[dte2_mask], test_preds.waste_probability[dte2_mask], 0.50
        )
        cls_dte2_optimal = self._classification_metrics(
            y_cls_test[dte2_mask], test_preds.waste_probability[dte2_mask], theta_opt
        )

        win_mask = dte_test <= self.config.THRESHOLD_OPT_MAX_DTE
        cls_win_default = self._classification_metrics(
            y_cls_test[win_mask], test_preds.waste_probability[win_mask], 0.50
        )
        cls_win_optimal = self._classification_metrics(
            y_cls_test[win_mask], test_preds.waste_probability[win_mask], theta_opt
        )

        # Test seti maliyet kıyaslaması (Legacy vs Ekonomik Pencere)
        test_legacy_050, _, _ = self._compute_row_financial_error_legacy(
            y_cls_test, test_preds.waste_probability, y_qty_test, rem_test, price_test, 0.50
        )
        test_legacy_opt, _, _ = self._compute_row_financial_error_legacy(
            y_cls_test, test_preds.waste_probability, y_qty_test, rem_test, price_test, theta_legacy
        )
        exp_win_sales_test = np.minimum(
            rem_test[win_mask],
            vel_test[win_mask] * np.maximum(dte_test[win_mask].astype(np.float64), 1.0),
        )
        test_econ_050, _, _ = self._compute_economic_window_row_error(
            y_cls_test[win_mask],
            test_preds.waste_probability[win_mask],
            y_qty_test[win_mask],
            exp_win_sales_test,
            price_test[win_mask],
            0.50,
        )
        test_econ_opt, test_econ_fn, test_econ_fp = self._compute_economic_window_row_error(
            y_cls_test[win_mask],
            test_preds.waste_probability[win_mask],
            y_qty_test[win_mask],
            exp_win_sales_test,
            price_test[win_mask],
            theta_opt,
        )

        # Regresyon metrikleri (Koşullu + Birleşik + Sıfır-Fire FP adedi)
        pos_mask = y_qty_test > 0
        zero_mask = ~pos_mask
        reg_conditional = self._regression_metrics(
            y_qty_test[pos_mask], test_preds.predicted_waste_qty[pos_mask]
        )
        reg_expected_all = self._regression_metrics(
            y_qty_test, test_preds.expected_waste_qty
        )
        reg_expected_pos = self._regression_metrics(
            y_qty_test[pos_mask], test_preds.expected_waste_qty[pos_mask]
        )
        hurdle_zero_preds = test_preds.expected_waste_qty[zero_mask]
        hurdle_zero_metrics = {
            "zero_waste_rows": int(np.sum(zero_mask)),
            "mean_false_positive_qty": round(
                float(np.mean(hurdle_zero_preds)) if len(hurdle_zero_preds) else 0.0, 4
            ),
            "median_false_positive_qty": round(
                float(np.median(hurdle_zero_preds)) if len(hurdle_zero_preds) else 0.0, 4
            ),
            "pct_pred_ge_1_unit": round(
                100.0 * float(np.mean(hurdle_zero_preds >= 1.0)) if len(hurdle_zero_preds) else 0.0,
                2,
            ),
        }

        # Test seti Aşama 1 Kalibrasyon kalitesi (Brier Score & 10-bin ECE)
        test_cal_metrics = EquoraHurdleModel.compute_calibration_metrics(
            y_true=y_cls_test,
            p_raw=test_preds.raw_waste_probability,
            p_cal=test_preds.waste_probability,
            n_bins=10,
        )

        # Hurdle vs. Tweedie Ablation Kıyaslaması
        hurdle_vs_tweedie: dict[str, Any] = {
            "hurdle": {
                "regressor_mode": model.regressor_target_mode,
                "calibrator_active": model.calibrator_active,
                "test_all_metrics": reg_expected_all,
                "test_zero_waste_metrics": hurdle_zero_metrics,
                "test_positive_waste_metrics": reg_expected_pos,
                "test_conditional_q_hat": reg_conditional,
                "test_ranking_metrics": {
                    "roc_auc": cls_optimal["roc_auc"],
                    "pr_auc": cls_optimal["pr_auc"],
                },
            },
            "tweedie": tweedie_result.to_dict() if tweedie_result is not None else None,
        }
        if tweedie_result is not None:
            wape_diff = round(
                tweedie_result.test_all_metrics["wape_pct"] - reg_expected_all["wape_pct"], 2
            )
            hurdle_vs_tweedie["comparison"] = {
                "tweedie_minus_hurdle_wape_pct": wape_diff,
                "lower_wape_winner": (
                    "Hurdle"
                    if reg_expected_all["wape_pct"] <= tweedie_result.test_all_metrics["wape_pct"]
                    else f"Tweedie (p={tweedie_result.best_variance_power})"
                ),
                "higher_pr_auc_winner": (
                    "Hurdle"
                    if cls_optimal["pr_auc"] >= tweedie_result.test_ranking_metrics["pr_auc"]
                    else f"Tweedie (p={tweedie_result.best_variance_power})"
                ),
            }

        # İş / Jüri Metrikleri (benzersiz parti bazlı)
        business_metrics = self._compute_business_metrics(
            test_df=test_df,
            p_hat=test_preds.waste_probability,
            threshold=theta_opt,
        )

        feature_importances = model.get_feature_importances(top_n=15)

        report: dict[str, Any] = {
            "threshold_optimization": threshold_info,
            "test_classification_all": {
                "default_050": cls_default,
                "optimal_theta": cls_optimal,
                "legacy_theta": cls_legacy,
                "row_financial_error_legacy_050_tl": round(test_legacy_050, 2),
                "row_financial_error_legacy_opt_tl": round(test_legacy_opt, 2),
            },
            "test_classification_action_window": {
                "max_dte": self.config.THRESHOLD_OPT_MAX_DTE,
                "rows": int(np.sum(win_mask)),
                "default_050": cls_win_default,
                "optimal_theta": cls_win_optimal,
                "econ_row_error_050_tl": round(test_econ_050, 2),
                "econ_row_error_opt_tl": round(test_econ_opt, 2),
                "econ_row_fn_opt_tl": round(test_econ_fn, 2),
                "econ_row_fp_opt_tl": round(test_econ_fp, 2),
            },
            "test_classification_dte_ge_2": {
                "rows": int(np.sum(dte2_mask)),
                "default_050": cls_dte2_default,
                "optimal_theta": cls_dte2_optimal,
            },
            "test_regression": {
                "regressor_target_mode": model.regressor_target_mode,
                "duan_smearing_factor": round(model.duan_smearing_factor, 5),
                "conditional_positive_waste_rows": int(np.sum(pos_mask)),
                "conditional_q_hat": reg_conditional,
                "overall_all_rows": int(test_df.height),
                "overall_expected_w_hat": reg_expected_all,
                "zero_waste_metrics": hurdle_zero_metrics,
            },
            "calibration_and_regressor_ablation": {
                "calibration_method": model.calibration_method,
                "calibrator_active": model.calibrator_active,
                "val_calibration": model.calibration_val_metrics,
                "test_calibration": test_cal_metrics,
                "regressor_ablation_val": model.regressor_ablation_results,
            },
            "hurdle_vs_tweedie_ablation": hurdle_vs_tweedie,
            "business_metrics_test": business_metrics,
            "feature_importances_top15": feature_importances,
        }

        self.config.eval_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_md = self._render_markdown_summary(report)
        self.config.eval_summary_path.write_text(summary_md, encoding="utf-8")

        return EvaluationOutput(
            optimal_threshold=theta_opt,
            legacy_threshold=theta_legacy,
            report=report,
            threshold_path=str(self.config.threshold_path),
            report_json_path=str(self.config.eval_report_path),
            summary_md_path=str(self.config.eval_summary_path),
        )

    @staticmethod
    def _render_markdown_summary(report: dict[str, Any]) -> str:
        """Jüri ve mühendislik incelemesi için 8 bölümlü Türkçe Markdown raporu üretir."""
        th = report["threshold_optimization"]
        c_all = report["test_classification_all"]
        c_win = report["test_classification_action_window"]
        c_dte2 = report["test_classification_dte_ge_2"]
        reg = report["test_regression"]
        biz = report["business_metrics_test"]
        fi = report["feature_importances_top15"]
        hvt = report["hurdle_vs_tweedie_ablation"]
        cal = report["calibration_and_regressor_ablation"]

        opt_th = th["optimal_threshold"]
        leg = th["legacy_row_all"]
        ec_row = th["economic_action_window_row"]
        ec_bat = th["economic_action_window_batch"]

        d050 = c_all["default_050"]
        dopt = c_all["optimal_theta"]
        d2_050 = c_dte2["default_050"]
        d2_opt = c_dte2["optimal_theta"]

        lines: list[str] = [
            "# EQUORA Faz 1 — Elde Kalacak Ürün Tahmini (Hurdle ML) Değerlendirme Raporu",
            "",
            "## 1. Maliyet Duyarlı Eşik Optimizasyonu (Validation Seti)",
            f"- **Aktif Ekonomik Karar Eşiği ($\\theta^*_{{\\text{{econ}}}}$):** `{opt_th:.2f}` (Aksiyon penceresi `DTE <= {th['action_window_max_dte']}`, kapsam: `{th['val_eval_scope']}`)",
            f"- **Benzersiz Parti Bazlı Ekonomik Eşik:** `{ec_bat['optimal_threshold']:.2f}` (`{ec_bat['unique_batches']:,}` parti)",
            f"- **Legacy (`row_all`) Eşiği:** `{leg['optimal_threshold']:.2f}` (geriye dönük kıyaslama)",
            f"- **Ekonomik Birim Katsayıları:** FN = `{ec_row['fn_cost_mult']:.2f}x` (`COGS + imha`), FP = `{ec_row['fp_cost_mult']:.2f}x` (`beklenen pencere satışı marj kaybı`)",
            f"- **Validation Pencere Hata Maliyeti (`0.50` -> `{opt_th:.2f}`):** `{ec_row['val_default_050_total_loss_tl']:,.2f} TL` -> **`{ec_row['val_optimal_total_loss_tl']:,.2f} TL`** (Azalma: **%{ec_row['val_loss_reduction_vs_050_pct']:.2f}**)",
            "",
            "## 2. Test Seti Sınıflandırma Metrikleri",
            "",
            "### Tüm Test Seti (`days_to_expiry >= 0`)",
            "| Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| Standart (`0.50`) | {d050['roc_auc']:.4f} | {d050['pr_auc']:.4f} | {d050['precision']:.4f} | {d050['recall']:.4f} | {d050['f1_score']:.4f} |",
            f"| **Ekonomik Optimal (`{opt_th:.2f}`)** | **{dopt['roc_auc']:.4f}** | **{dopt['pr_auc']:.4f}** | **{dopt['precision']:.4f}** | **{dopt['recall']:.4f}** | **{dopt['f1_score']:.4f}** |",
            "",
            f"### Aksiyon Penceresi (`days_to_expiry <= {c_win['max_dte']}`, n={c_win['rows']:,}) & 48s+ Ufku (`days_to_expiry >= 2`, n={c_dte2['rows']:,})",
            "| Kapsam / Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| Pencere `DTE <= {c_win['max_dte']}` — Standart (`0.50`) | {c_win['default_050']['roc_auc']:.4f} | {c_win['default_050']['pr_auc']:.4f} | {c_win['default_050']['precision']:.4f} | {c_win['default_050']['recall']:.4f} | {c_win['default_050']['f1_score']:.4f} |",
            f"| **Pencere `DTE <= {c_win['max_dte']}` — Optimal (`{opt_th:.2f}`)** | **{c_win['optimal_theta']['roc_auc']:.4f}** | **{c_win['optimal_theta']['pr_auc']:.4f}** | **{c_win['optimal_theta']['precision']:.4f}** | **{c_win['optimal_theta']['recall']:.4f}** | **{c_win['optimal_theta']['f1_score']:.4f}** |",
            f"| Erken Ufuk `DTE >= 2` — Standart (`0.50`) | {d2_050['roc_auc']:.4f} | {d2_050['pr_auc']:.4f} | {d2_050['precision']:.4f} | {d2_050['recall']:.4f} | {d2_050['f1_score']:.4f} |",
            f"| **Erken Ufuk `DTE >= 2` — Optimal (`{opt_th:.2f}`)** | **{d2_opt['roc_auc']:.4f}** | **{d2_opt['pr_auc']:.4f}** | **{d2_opt['precision']:.4f}** | **{d2_opt['recall']:.4f}** | **{d2_opt['f1_score']:.4f}** |",
            "",
            "## 3. Test Seti Regresyon Metrikleri (Fire Miktarı Tahmini)",
            f"- **Seçilen Koşullu Regresör Stratejisi:** `{reg['regressor_target_mode']}` (Duan Smearing: `{reg['duan_smearing_factor']:.5f}`)",
            "",
            "| Kapsam | Satır Sayısı | MAE (Adet) | RMSE (Adet) | WAPE (%) |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| Koşullu Regresör (`final_waste_qty > 0`, $\\hat{{q}}$) | {reg['conditional_positive_waste_rows']:,} | {reg['conditional_q_hat']['mae']:.4f} | {reg['conditional_q_hat']['rmse']:.4f} | %{reg['conditional_q_hat']['wape_pct']:.2f} |",
            f"| Birleşik Beklenen Fire (Tümü, $\\hat{{W}} = \\hat{{p}} \\times \\hat{{q}}$) | {reg['overall_all_rows']:,} | {reg['overall_expected_w_hat']['mae']:.4f} | {reg['overall_expected_w_hat']['rmse']:.4f} | %{reg['overall_expected_w_hat']['wape_pct']:.2f} |",
            "",
            "## 4. Test Seti İş ve Jüri Metrikleri (Benzersiz Parti Bazlı)",
            f"- **Toplam Benzersiz Test Partisi:** `{biz['total_unique_test_batches']:,}` (Fire oluşan: `{biz['total_waste_batches']:,}`)",
            f"- **Aksiyon Penceresinde (`2 <= DTE <= {th['action_window_max_dte']}`) Yakalanan Fire Oranı:**",
            f"  - Parti bazlı: **%{biz['early_catch_rate_by_batch_pct']:.2f}** (`{biz['caught_early_48h_waste_batches']:,}` / `{biz['total_waste_batches']:,}` parti)",
            f"  - Ürün adedi bazlı: **%{biz['early_catch_rate_by_units_pct']:.2f}** (`{biz['rescueable_units_48h']:,}` / `{biz['total_test_waste_units']:,}` adet)",
            f"  - TL değeri bazlı: **%{biz['early_catch_rate_by_tl_pct']:.2f}** (`{biz['rescueable_gross_value_tl']:,.2f} TL` brüt kurtarma potansiyeli)",
            f"- **Baseline (Müdahale Yok) Finansal Kayıp:** `{biz['baseline_no_action_loss_tl']:,.2f} TL`",
            f"- **EQUORA Tahmini Toplam Finansal Kayıp:** `{biz['equora_estimated_total_loss_tl']:,.2f} TL`",
            f"  - Kaçırılan Fire (FN, %100): `{biz['equora_missed_waste_fn_loss_tl']:,.2f} TL`",
            f"  - Erken Kurtarma İndirim Marj Maliyeti (TP, %{ModelEvaluator.config_fp_pct(ec_row['fp_cost_mult'])}): `{biz['equora_tp_discount_cost_tl']:,.2f} TL`",
            f"  - Yanlış Alarm İndirim Marj Maliyeti (FP, %{ModelEvaluator.config_fp_pct(ec_row['fp_cost_mult'])}): `{biz['equora_fp_discount_cost_tl']:,.2f} TL`",
            f"- **Tahmini Net Finansal Tasarruf:** **`{biz['equora_net_savings_tl']:,.2f} TL`** (**%{biz['equora_loss_reduction_pct']:.2f}** zarar azalması)",
            "",
            "### ESG / Karbon Ayak İzi (CO2e) Etkisi (Test Seti — Benzersiz Partiler)",
            f"- **Müdahalesiz (Baseline) Fire Karbon Salımı:** `{biz['esg_co2_impact']['baseline_waste_co2_kg']:,.2f} kg CO2e` (`{biz['esg_co2_impact']['baseline_waste_co2_tons']:.3f} ton`)",
            f"- **48 Saat+ Erken Yakalama ile Önlenebilir CO2e:** **`{biz['esg_co2_impact']['prevented_co2_48h_kg']:,.2f} kg CO2e`** (`{biz['esg_co2_impact']['prevented_co2_48h_tons']:.3f} ton`, **%{biz['esg_co2_impact']['prevented_co2_48h_pct']:.2f}**)",
            f"- **Çevresel Eşdeğerlik:** ~`{biz['esg_co2_impact']['prevented_equivalent_car_km']:,.0f} km` binek araç sürüşü / ~`{biz['esg_co2_impact']['prevented_equivalent_tree_years']:,.0f}` olgun ağacın 1 yıllık CO2 tutumu",
            "",
            "| Kategori | Fire Adedi | Baseline CO2e (kg) | 48s+ Önlenebilir CO2e (kg) |",
            "| :--- | :---: | :---: | :---: |",
            *[
                f"| {row['category']} | {int(round(row['waste_units'])):,} | {row['baseline_co2_kg']:,.2f} | {row['prevented_co2_48h_kg']:,.2f} |"
                for row in biz["esg_co2_impact"]["category_breakdown"]
            ],
            "",
            "## 5. Açıklanabilirlik — Top 15 Feature Importance (Gain)",
            "",
            "### Aşama 1: Risk Sınıflandırıcı (Top 15)",
            "| Sıra | Özellik | Gain (%) | Split |",
            "| :---: | :--- | :---: | :---: |",
        ]
        for idx, row in enumerate(fi["classifier"], start=1):
            lines.append(f"| {idx} | `{row['feature']}` | %{row['gain_pct']:.2f} | {row['split']:,} |")

        lines.extend(
            [
                "",
                "### Aşama 2: Koşullu Fire Regresörü (Top 15)",
                "| Sıra | Özellik | Gain (%) | Split |",
                "| :---: | :--- | :---: | :---: |",
            ]
        )
        for idx, row in enumerate(fi["regressor"], start=1):
            lines.append(f"| {idx} | `{row['feature']}` | %{row['gain_pct']:.2f} | {row['split']:,} |")

        # --- BÖLÜM 6: Hurdle vs. Tweedie Ablation ---
        lines.extend(["", "## 6. Hurdle vs. Tweedie (`1 < p < 2`) Ablation Kıyaslaması"])
        tw = hvt.get("tweedie")
        hu = hvt["hurdle"]
        if tw is not None:
            lines.extend(
                [
                    f"- **En İyi Tweedie Varyans Üssü ($p^*$):** `{tw['best_variance_power']}` (`best_iter={tw['best_iteration']}`)",
                    f"- **Tweedie Validation Izgarası:** "
                    + ", ".join(
                        f"`p={r['variance_power']}` -> WAPE `%{r['val_wape_pct']:.2f}`"
                        for r in tw["val_grid_results"]
                    ),
                    "",
                    "| Metrik (Out-of-Time Test) | EQUORA Hurdle (`p_hat * q_hat`) | Tweedie Benchmark (`p*`) |",
                    "| :--- | :---: | :---: |",
                    f"| Tüm Satırlar MAE (Adet) | {hu['test_all_metrics']['mae']:.4f} | {tw['test_all_metrics']['mae']:.4f} |",
                    f"| Tüm Satırlar RMSE (Adet) | {hu['test_all_metrics']['rmse']:.4f} | {tw['test_all_metrics']['rmse']:.4f} |",
                    f"| **Tüm Satırlar WAPE (%)** | **%{hu['test_all_metrics']['wape_pct']:.2f}** | **%{tw['test_all_metrics']['wape_pct']:.2f}** |",
                    f"| Sıfır-Fire (`W=0`) Ort. Yanlış Tahmin (Adet) | {hu['test_zero_waste_metrics']['mean_false_positive_qty']:.4f} | {tw['test_zero_waste_metrics']['mean_false_positive_qty']:.4f} |",
                    f"| Sıfır-Fire (`W=0`) `>= 1 adet` Tahmin Oranı | %{hu['test_zero_waste_metrics']['pct_pred_ge_1_unit']:.2f} | %{tw['test_zero_waste_metrics']['pct_pred_ge_1_unit']:.2f} |",
                    f"| Pozitif-Fire (`W>0`) WAPE (%) | %{hu['test_positive_waste_metrics']['wape_pct']:.2f} | %{tw['test_positive_waste_metrics']['wape_pct']:.2f} |",
                    f"| Risk Sıralaması ROC-AUC | {hu['test_ranking_metrics']['roc_auc']:.4f} | {tw['test_ranking_metrics']['roc_auc']:.4f} |",
                    f"| Risk Sıralaması PR-AUC | {hu['test_ranking_metrics']['pr_auc']:.4f} | {tw['test_ranking_metrics']['pr_auc']:.4f} |",
                ]
            )
        else:
            lines.append("- Tweedie kıyaslaması bu çalıştırmada atlandı (`--skip-benchmarks`).")

        # --- BÖLÜM 7: Aşama 1 Kalibrasyon & Aşama 2 Regresör Ablation ---
        tc = cal["test_calibration"]
        vc = cal["val_calibration"]
        lines.extend(
            [
                "",
                "## 7. Aşama 1 Kalibrasyon (Brier & ECE) ve Aşama 2 Regresör Kayıp Kıyaslaması",
                f"- **Kalibrasyon Yöntemi:** `{cal['calibration_method']}` | **Aktif mi?** `{'EVET' if cal['calibrator_active'] else 'HAYIR (Ham olasılık korundu)'}`",
                f"- **Validation Kontrolü (Brier / ECE):** Ham `{vc.get('brier_raw', 0.0):.5f}` / `{vc.get('ece_10bin_raw', 0.0):.5f}` -> Kalibre `{vc.get('brier_calibrated', 0.0):.5f}` / `{vc.get('ece_10bin_calibrated', 0.0):.5f}`",
                f"- **Test Seti (Brier / ECE 10-bin):** Ham `{tc['brier_raw']:.5f}` / `{tc['ece_10bin_raw']:.5f}` -> Kalibre **`{tc['brier_calibrated']:.5f}` / `{tc['ece_10bin_calibrated']:.5f}`**",
                "",
                "| Aşama 2 Regresör Stratejisi | Val MAE | Val RMSE | Val WAPE (%) | Duan Smearing | Seçildi |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for row in cal["regressor_ablation_val"]:
            tag = "**SEÇİLDİ**" if row.get("selected") else "-"
            lines.append(
                f"| `{row['mode']}` | {row['val_mae']:.4f} | {row['val_rmse']:.4f} | %{row['val_wape_pct']:.2f} | {row['duan_smearing']:.5f} | {tag} |"
            )

        # --- BÖLÜM 8: Eski (row_all) vs. Ekonomik-Tutarlı (action_window) Eşik Kıyaslaması ---
        lines.extend(
            [
                "",
                "## 8. Eşik Optimizasyonu Kıyaslaması: Legacy (`row_all`) vs. Ekonomik (`action_window`)",
                "| Optimizasyon Modu | Kapsam | FN / FP Katsayısı | Optimal $\\theta^*$ | Val Maliyet (`0.50`) | Val Maliyet ($\\theta^*$) | Azalma (%) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
                f"| Legacy (`row_all`) | Tüm günler (`DTE >= 0`) | `{leg['fn_cost_mult']:.2f}` / `{leg['fp_cost_mult']:.2f}` (tüm kalan stok) | `{leg['optimal_threshold']:.2f}` | {leg['val_default_050_total_loss_tl']:,.2f} TL | {leg['val_optimal_total_loss_tl']:,.2f} TL | %{leg['val_loss_reduction_vs_050_pct']:.2f} |",
                f"| **Ekonomik (`action_window` satır)** | `DTE <= {th['action_window_max_dte']}` | `{ec_row['fn_cost_mult']:.2f}` / `{ec_row['fp_cost_mult']:.2f}` (pencere satışı) | **`{ec_row['optimal_threshold']:.2f}`** | {ec_row['val_default_050_total_loss_tl']:,.2f} TL | {ec_row['val_optimal_total_loss_tl']:,.2f} TL | **%{ec_row['val_loss_reduction_vs_050_pct']:.2f}** |",
                f"| Ekonomik (`action_window` parti) | `DTE <= {th['action_window_max_dte']}` (benzersiz parti) | `{ec_row['fn_cost_mult']:.2f}` / `{ec_row['fp_cost_mult']:.2f}` (ilk alarm) | `{ec_bat['optimal_threshold']:.2f}` | {ec_bat['val_default_050_total_loss_tl']:,.2f} TL | {ec_bat['val_optimal_total_loss_tl']:,.2f} TL | %{ec_bat['val_loss_reduction_vs_050_pct']:.2f} |",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def config_fp_pct(fp_mult: float) -> str:
        return f"{fp_mult * 100.0:.0f}"
