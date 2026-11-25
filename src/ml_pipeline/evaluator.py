r"""Faz 1 / Adım 5 — Maliyet Duyarlı Eşik Optimizasyonu ve Değerlendirme (ModelEvaluator).

Girdi:
    - Eğitilmiş `EquoraHurdleModel`, Validation (`val_df`) ve Test (`test_df`)
      Polars DataFrame'leri.

Çıktı:
    - `models/hurdle_v1/threshold.json`: Asimetrik maliyet matrisiyle Validation
      seti üzerinde bulunan optimal karar eşiği (theta*).
    - `reports/phase1_eval/evaluation_report.json`: Tüm teknik metrikler,
      iş/jüri metrikleri ve Top-15 özellik önemleri (JSON).
    - `reports/phase1_eval/summary.md`: Jüri/yönetici sunumuna hazır Türkçe
      değerlendirme raporu (Markdown).

Rol:
    - Validation seti üzerinde theta in [0.10, 0.90] aralığını 0.01 adımlarla
      tarayarak toplam finansal hatayı minimize eden theta* değerini bulur.
    - Test setinde standart `0.50` ile optimal `theta*` karşılaştırmalı teknik
      metriklerini, regresyon doğruluğunu, 48 saat+ erken yakalama iş metriklerini
      (parti bazında çift saymadan) ve açıklanabilirlik tablolarını üretir.
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


@dataclass
class EvaluationOutput:
    """ModelEvaluator tam çıktı paketi."""

    optimal_threshold: float
    report: dict[str, Any]
    threshold_path: str
    report_json_path: str
    summary_md_path: str


class ModelEvaluator:
    """Maliyet duyarlı eşik optimizasyonu, teknik ve iş/jüri metrik değerlendiricisi."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.co2_calculator = CO2ImpactCalculator()

    # -------------------------------------------------------------------------
    # Yardımcı: Asimetrik Finansal Hata Maliyeti (TL)
    # -------------------------------------------------------------------------
    def _compute_row_financial_error(
        self,
        y_true: np.ndarray,
        p_hat: np.ndarray,
        final_waste_qty: np.ndarray,
        remaining_qty: np.ndarray,
        unit_price: np.ndarray,
        threshold: float,
    ) -> tuple[float, float, float]:
        """Verilen eşik için satır bazlı asimetrik finansal hata maliyetini (TL) hesaplar.

        Maliyet Matrisi:
            - False Negative (y=1, y_hat=0): Fire olacak ürün kaçırıldı ->
              `FN_COST_MULT (1.00) * final_waste_qty * unit_price` (tam ürün kaybı).
            - False Positive (y=0, y_hat=1): Satılacak ürüne gereksiz indirim verildi ->
              `FP_COST_MULT (0.25) * remaining_qty * unit_price` (marj kaybı).
        """
        y_pred = (p_hat >= threshold).astype(np.int32)

        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)

        fn_cost_tl = float(
            np.sum(
                self.config.FN_COST_MULT
                * final_waste_qty[fn_mask]
                * unit_price[fn_mask]
            )
        )
        fp_cost_tl = float(
            np.sum(
                self.config.FP_COST_MULT
                * remaining_qty[fp_mask]
                * unit_price[fp_mask]
            )
        )
        return fn_cost_tl + fp_cost_tl, fn_cost_tl, fp_cost_tl

    # -------------------------------------------------------------------------
    # 1) Validation Seti Üzerinde Maliyet Duyarlı Eşik Optimizasyonu
    # -------------------------------------------------------------------------
    def optimize_threshold(
        self,
        model: EquoraHurdleModel,
        val_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Validation setinde [0.10, 0.90] aralığını 0.01 adımla tarayıp optimal theta*'ı bulur."""
        p_val = model.predict_proba(val_df)
        y_val = val_df["will_expire_unsold"].to_numpy().astype(np.int32)
        w_val = val_df["final_waste_qty"].to_numpy().astype(np.float64)
        r_val = val_df["remaining_qty"].to_numpy().astype(np.float64)
        u_val = val_df["unit_price"].to_numpy().astype(np.float64)

        thresholds = np.round(
            np.arange(
                self.config.THRESHOLD_MIN,
                self.config.THRESHOLD_MAX + 1e-9,
                self.config.THRESHOLD_STEP,
            ),
            2,
        )

        best_theta = 0.50
        best_total_loss = float("inf")
        best_fn_loss = 0.0
        best_fp_loss = 0.0
        scan_points: list[dict[str, float]] = []

        for theta in thresholds:
            total_loss, fn_loss, fp_loss = self._compute_row_financial_error(
                y_true=y_val,
                p_hat=p_val,
                final_waste_qty=w_val,
                remaining_qty=r_val,
                unit_price=u_val,
                threshold=float(theta),
            )
            scan_points.append(
                {
                    "threshold": float(theta),
                    "total_loss_tl": round(total_loss, 2),
                    "fn_loss_tl": round(fn_loss, 2),
                    "fp_loss_tl": round(fp_loss, 2),
                }
            )
            if total_loss < best_total_loss:
                best_total_loss = total_loss
                best_fn_loss = fn_loss
                best_fp_loss = fp_loss
                best_theta = float(theta)

        default_total, default_fn, default_fp = self._compute_row_financial_error(
            y_true=y_val,
            p_hat=p_val,
            final_waste_qty=w_val,
            remaining_qty=r_val,
            unit_price=u_val,
            threshold=0.50,
        )

        threshold_payload: dict[str, Any] = {
            "optimal_threshold": best_theta,
            "default_threshold": 0.50,
            "fn_cost_mult": self.config.FN_COST_MULT,
            "fp_cost_mult": self.config.FP_COST_MULT,
            "search_range": [
                self.config.THRESHOLD_MIN,
                self.config.THRESHOLD_MAX,
                self.config.THRESHOLD_STEP,
            ],
            "val_optimal_total_loss_tl": round(best_total_loss, 2),
            "val_optimal_fn_loss_tl": round(best_fn_loss, 2),
            "val_optimal_fp_loss_tl": round(best_fp_loss, 2),
            "val_default_050_total_loss_tl": round(default_total, 2),
            "val_default_050_fn_loss_tl": round(default_fn, 2),
            "val_default_050_fp_loss_tl": round(default_fp, 2),
            "val_loss_reduction_vs_050_pct": round(
                100.0 * (default_total - best_total_loss) / max(default_total, 1e-6),
                2,
            ),
            # Özet eğri (her 0.05 adım + optimal nokta)
            "scan_summary": [
                pt
                for pt in scan_points
                if abs((pt["threshold"] * 100) % 5) < 1e-6
                or pt["threshold"] == best_theta
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
        """Test setindeki benzersiz partiler üzerinden 48 saat+ erken yakalama ve TL etkisini ölçer.

        Çift Sayım Koruması:
            - Her parti birden fazla günlük kesitte (snapshot_date) yer aldığından,
              finansal TL ve ürün adedi hesapları **benzersiz `batch_id`** düzeyinde
              yapılır; böylece aynı partinin zararı asla gün sayısı kadar katlanmaz.
        """
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
        # Literatür tabanlı (Poore & Nemecek / FAO) parti karbon ayak izi (kg CO2e)
        batch_truth_df = self.co2_calculator.add_co2_columns(
            batch_truth_df,
            qty_col="final_waste_qty",
            output_col="waste_co2_kg",
        )

        # SKT'ye en az 2 gün kala (days_to_expiry >= 2) üretilen ilk/en erken uyarılar
        early_alerts_df = (
            scored_df.filter((pl.col("days_to_expiry") >= 2) & (pl.col("is_alert") == 1))
            .sort(["batch_id", "snapshot_date"])
            .group_by("batch_id")
            .agg(
                pl.lit(1, dtype=pl.Int8).alias("caught_early_48h"),
                pl.col("days_to_expiry").first().alias("first_alert_dte"),
                pl.col("remaining_qty").first().cast(pl.Float64).alias("first_alert_rem_qty"),
            )
        )

        # Tüm test partilerini erken uyarı durumuyla birleştir
        eval_batches_df = (
            batch_truth_df.join(early_alerts_df, on="batch_id", how="left")
            .with_columns(
                pl.col("caught_early_48h").fill_null(0),
                pl.col("first_alert_rem_qty").fill_null(0.0),
            )
            .with_columns(
                # Gerçek fire TL değeri (müdahale edilmezse %100 kayıp)
                (pl.col("final_waste_qty") * pl.col("unit_price")).alias("waste_value_tl")
            )
        )

        # Baseline (hiç müdahale yok) toplam fire istatistikleri
        waste_batches = eval_batches_df.filter(pl.col("will_expire_unsold") == 1)
        non_waste_batches = eval_batches_df.filter(pl.col("will_expire_unsold") == 0)

        total_test_batches = eval_batches_df.height
        total_waste_batches = waste_batches.height
        total_waste_units = float(waste_batches["final_waste_qty"].sum() or 0.0)
        baseline_loss_tl = float(waste_batches["waste_value_tl"].sum() or 0.0)

        # 48 saat+ önceden yakalanan gerçek fire partileri (True Positive @ DTE >= 2)
        caught_waste = waste_batches.filter(pl.col("caught_early_48h") == 1)
        missed_waste = waste_batches.filter(pl.col("caught_early_48h") == 0)
        false_alarm_batches = non_waste_batches.filter(pl.col("caught_early_48h") == 1)

        caught_waste_batch_count = caught_waste.height
        rescueable_units = float(caught_waste["final_waste_qty"].sum() or 0.0)
        rescueable_gross_tl = float(caught_waste["waste_value_tl"].sum() or 0.0)

        early_catch_batch_pct = (
            100.0 * caught_waste_batch_count / max(total_waste_batches, 1)
        )
        early_catch_unit_pct = (
            100.0 * rescueable_units / max(total_waste_units, 1e-6)
        )
        early_catch_tl_pct = (
            100.0 * rescueable_gross_tl / max(baseline_loss_tl, 1e-6)
        )

        # EQUORA finansal maliyet bileşenleri (Baseline vs EQUORA karşılaştırması):
        # 1) Kaçırılan fireler (FN): birim fiyatın %100'ü kayıp
        missed_waste_loss_tl = float(missed_waste["waste_value_tl"].sum() or 0.0) * self.config.FN_COST_MULT
        # 2) Erken yakalanıp dinamik indirimle kurtarılan firelerin indirim marj maliyeti (%25)
        tp_markdown_cost_tl = rescueable_gross_tl * self.config.FP_COST_MULT
        # 3) Yanlış alarm (FP) verilen satılacak partilere uygulanan gereksiz indirim marj kaybı (%25)
        fp_markdown_cost_tl = float(
            (
                false_alarm_batches["first_alert_rem_qty"]
                * false_alarm_batches["unit_price"]
            ).sum()
            or 0.0
        ) * self.config.FP_COST_MULT

        equora_total_loss_tl = (
            missed_waste_loss_tl + tp_markdown_cost_tl + fp_markdown_cost_tl
        )
        net_savings_tl = baseline_loss_tl - equora_total_loss_tl
        net_loss_reduction_pct = (
            100.0 * net_savings_tl / max(baseline_loss_tl, 1e-6)
        )

        # Ek olarak tam DTE == 2 (48 saat kala tek gün kesiti) snapshot bazlı yakalama oranı
        dte2_df = scored_df.filter(pl.col("days_to_expiry") == 2)
        dte2_waste_df = dte2_df.filter(pl.col("will_expire_unsold") == 1)
        dte2_catch_pct = (
            100.0 * float(dte2_waste_df["is_alert"].mean() or 0.0)
            if dte2_waste_df.height > 0
            else 0.0
        )

        # ESG / CO2e Etki Hesabı (benzersiz parti bazlı — sıfır çift sayım)
        baseline_waste_co2_kg = float(waste_batches["waste_co2_kg"].sum() or 0.0)
        prevented_co2_48h_kg = float(caught_waste["waste_co2_kg"].sum() or 0.0)
        prevented_co2_pct = (
            100.0 * prevented_co2_48h_kg / max(baseline_waste_co2_kg, 1e-6)
        )
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
    # 4) Ana Değerlendirme ve Raporlama Akışı
    # -------------------------------------------------------------------------
    def evaluate(
        self,
        model: EquoraHurdleModel,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> EvaluationOutput:
        """Validation üzerinde eşiği optimize eder, Test setini değerlendirir ve raporları yazar."""
        # 1) Maliyet duyarlı eşik optimizasyonu (Validation)
        threshold_info = self.optimize_threshold(model=model, val_df=val_df)
        theta_opt = float(threshold_info["optimal_threshold"])

        # 2) Test seti tahminleri
        test_preds = model.predict_hurdle(test_df, threshold=theta_opt)
        y_cls_test = test_df["will_expire_unsold"].to_numpy().astype(np.int32)
        y_qty_test = test_df["final_waste_qty"].to_numpy().astype(np.float64)
        rem_test = test_df["remaining_qty"].to_numpy().astype(np.float64)
        price_test = test_df["unit_price"].to_numpy().astype(np.float64)
        dte_test = test_df["days_to_expiry"].to_numpy().astype(np.int32)

        # Sınıflandırma metrikleri: standart 0.50 vs optimal theta* (tüm test seti)
        cls_default = self._classification_metrics(y_cls_test, test_preds.waste_probability, 0.50)
        cls_optimal = self._classification_metrics(y_cls_test, test_preds.waste_probability, theta_opt)

        # Şeffaflık: SKT >= 2 gün (48s+ erken tahmin ufku) alt kümesi sınıflandırma metrikleri
        dte2_mask = dte_test >= 2
        cls_dte2_default = self._classification_metrics(
            y_cls_test[dte2_mask],
            test_preds.waste_probability[dte2_mask],
            0.50,
        )
        cls_dte2_optimal = self._classification_metrics(
            y_cls_test[dte2_mask],
            test_preds.waste_probability[dte2_mask],
            theta_opt,
        )

        # Test seti satır bazlı asimetrik maliyet (0.50 vs theta*)
        test_cost_050, test_fn_050, test_fp_050 = self._compute_row_financial_error(
            y_cls_test, test_preds.waste_probability, y_qty_test, rem_test, price_test, 0.50
        )
        test_cost_opt, test_fn_opt, test_fp_opt = self._compute_row_financial_error(
            y_cls_test, test_preds.waste_probability, y_qty_test, rem_test, price_test, theta_opt
        )

        # Regresyon metrikleri:
        #   a) Koşullu (yalnızca gerçek fire > 0 satırları üzerinde q_hat)
        #   b) Genel beklenen fire (tüm test satırları üzerinde W_hat = p_hat * q_hat)
        pos_mask = y_qty_test > 0
        reg_conditional = self._regression_metrics(
            y_qty_test[pos_mask],
            test_preds.predicted_waste_qty[pos_mask],
        )
        reg_expected_all = self._regression_metrics(
            y_qty_test,
            test_preds.expected_waste_qty,
        )

        # 3) Test seti iş / jüri metrikleri (benzersiz parti bazlı)
        business_metrics = self._compute_business_metrics(
            test_df=test_df,
            p_hat=test_preds.waste_probability,
            threshold=theta_opt,
        )

        # 4) Açıklanabilirlik — Top 15 Feature Importance
        feature_importances = model.get_feature_importances(top_n=15)

        report: dict[str, Any] = {
            "threshold_optimization": threshold_info,
            "test_classification_all": {
                "default_050": cls_default,
                "optimal_theta": cls_optimal,
                "row_financial_error_050_tl": round(test_cost_050, 2),
                "row_financial_error_opt_tl": round(test_cost_opt, 2),
                "row_fn_error_opt_tl": round(test_fn_opt, 2),
                "row_fp_error_opt_tl": round(test_fp_opt, 2),
            },
            "test_classification_dte_ge_2": {
                "rows": int(np.sum(dte2_mask)),
                "default_050": cls_dte2_default,
                "optimal_theta": cls_dte2_optimal,
            },
            "test_regression": {
                "conditional_positive_waste_rows": int(np.sum(pos_mask)),
                "conditional_q_hat": reg_conditional,
                "overall_all_rows": int(test_df.height),
                "overall_expected_w_hat": reg_expected_all,
            },
            "business_metrics_test": business_metrics,
            "feature_importances_top15": feature_importances,
        }

        # JSON ve Markdown raporlarını kaydet
        self.config.eval_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_md = self._render_markdown_summary(report)
        self.config.eval_summary_path.write_text(summary_md, encoding="utf-8")

        return EvaluationOutput(
            optimal_threshold=theta_opt,
            report=report,
            threshold_path=str(self.config.threshold_path),
            report_json_path=str(self.config.eval_report_path),
            summary_md_path=str(self.config.eval_summary_path),
        )

    @staticmethod
    def _render_markdown_summary(report: dict[str, Any]) -> str:
        """Jüri ve mühendislik incelemesi için okunabilir Türkçe Markdown raporu üretir."""
        th = report["threshold_optimization"]
        c_all = report["test_classification_all"]
        c_dte2 = report["test_classification_dte_ge_2"]
        reg = report["test_regression"]
        biz = report["business_metrics_test"]
        fi = report["feature_importances_top15"]

        opt_th = th["optimal_threshold"]
        d050 = c_all["default_050"]
        dopt = c_all["optimal_theta"]
        d2_050 = c_dte2["default_050"]
        d2_opt = c_dte2["optimal_theta"]

        lines: list[str] = [
            "# EQUORA Faz 1 — Elde Kalacak Ürün Tahmini (Hurdle ML) Değerlendirme Raporu",
            "",
            "## 1. Maliyet Duyarlı Eşik Optimizasyonu (Validation Seti)",
            f"- **Optimal Karar Eşiği ($\\theta^*$):** `{opt_th:.2f}` (Standart eşik: `0.50`)",
            f"- **Asimetrik Maliyet Katsayıları:** FN = `{th['fn_cost_mult']:.2f}x` (tam ürün kaybı), FP = `{th['fp_cost_mult']:.2f}x` (gereksiz indirim marj kaybı)",
            f"- **Validation Hata Maliyeti (`0.50`):** `{th['val_default_050_total_loss_tl']:,.2f} TL`",
            f"- **Validation Hata Maliyeti (`{opt_th:.2f}`):** `{th['val_optimal_total_loss_tl']:,.2f} TL` (Azalma: `%{th['val_loss_reduction_vs_050_pct']:.2f}`)",
            "",
            "## 2. Test Seti Sınıflandırma Metrikleri (Tüm Test Seti vs. 48 Saat+ Ufku)",
            "",
            "### Tüm Test Seti (`days_to_expiry >= 0`)",
            "| Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| Standart (`0.50`) | {d050['roc_auc']:.4f} | {d050['pr_auc']:.4f} | {d050['precision']:.4f} | {d050['recall']:.4f} | {d050['f1_score']:.4f} |",
            f"| **Optimal (`{opt_th:.2f}`)** | **{dopt['roc_auc']:.4f}** | **{dopt['pr_auc']:.4f}** | **{dopt['precision']:.4f}** | **{dopt['recall']:.4f}** | **{dopt['f1_score']:.4f}** |",
            "",
            f"### Aksiyon Alınabilir Erken Ufuk (`days_to_expiry >= 2`, n={c_dte2['rows']:,})",
            "| Eşik | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| Standart (`0.50`) | {d2_050['roc_auc']:.4f} | {d2_050['pr_auc']:.4f} | {d2_050['precision']:.4f} | {d2_050['recall']:.4f} | {d2_050['f1_score']:.4f} |",
            f"| **Optimal (`{opt_th:.2f}`)** | **{d2_opt['roc_auc']:.4f}** | **{d2_opt['pr_auc']:.4f}** | **{d2_opt['precision']:.4f}** | **{d2_opt['recall']:.4f}** | **{d2_opt['f1_score']:.4f}** |",
            "",
            "## 3. Test Seti Regresyon Metrikleri (Fire Miktarı Tahmini)",
            "| Kapsam | Satır Sayısı | MAE (Adet) | RMSE (Adet) | WAPE (%) |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| Koşullu Regresör (`final_waste_qty > 0`, $\\hat{{q}}$) | {reg['conditional_positive_waste_rows']:,} | {reg['conditional_q_hat']['mae']:.4f} | {reg['conditional_q_hat']['rmse']:.4f} | %{reg['conditional_q_hat']['wape_pct']:.2f} |",
            f"| Birleşik Beklenen Fire (Tümü, $\\hat{{W}} = \\hat{{p}} \\times \\hat{{q}}$) | {reg['overall_all_rows']:,} | {reg['overall_expected_w_hat']['mae']:.4f} | {reg['overall_expected_w_hat']['rmse']:.4f} | %{reg['overall_expected_w_hat']['wape_pct']:.2f} |",
            "",
            "## 4. Test Seti İş ve Jüri Metrikleri (Benzersiz Parti Bazlı)",
            f"- **Toplam Benzersiz Test Partisi:** `{biz['total_unique_test_batches']:,}` (Fire oluşan: `{biz['total_waste_batches']:,}`)",
            f"- **SKT'ye En Az 2 Gün Kala (48s+) Yakalanan Fire Oranı:**",
            f"  - Parti bazlı: **%{biz['early_catch_rate_by_batch_pct']:.2f}** (`{biz['caught_early_48h_waste_batches']:,}` / `{biz['total_waste_batches']:,}` parti)",
            f"  - Ürün adedi bazlı: **%{biz['early_catch_rate_by_units_pct']:.2f}** (`{biz['rescueable_units_48h']:,}` / `{biz['total_test_waste_units']:,}` adet)",
            f"  - TL değeri bazlı: **%{biz['early_catch_rate_by_tl_pct']:.2f}** (`{biz['rescueable_gross_value_tl']:,.2f} TL` brüt kurtarma potansiyeli)",
            f"- **Baseline (Müdahale Yok) Finansal Kayıp:** `{biz['baseline_no_action_loss_tl']:,.2f} TL`",
            f"- **EQUORA Tahmini Toplam Finansal Kayıp:** `{biz['equora_estimated_total_loss_tl']:,.2f} TL`",
            f"  - Kaçırılan Fire (FN, %100): `{biz['equora_missed_waste_fn_loss_tl']:,.2f} TL`",
            f"  - Erken Kurtarma İndirim Marj Maliyeti (TP, %25): `{biz['equora_tp_discount_cost_tl']:,.2f} TL`",
            f"  - Yanlış Alarm İndirim Marj Maliyeti (FP, %25): `{biz['equora_fp_discount_cost_tl']:,.2f} TL`",
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
            lines.append(
                f"| {idx} | `{row['feature']}` | %{row['gain_pct']:.2f} | {row['split']:,} |"
            )

        lines.extend(
            [
                "",
                "### Aşama 2: Koşullu Fire Regresörü (Top 15)",
                "| Sıra | Özellik | Gain (%) | Split |",
                "| :---: | :--- | :---: | :---: |",
            ]
        )
        for idx, row in enumerate(fi["regressor"], start=1):
            lines.append(
                f"| {idx} | `{row['feature']}` | %{row['gain_pct']:.2f} | {row['split']:,} |"
            )

        lines.append("")
        return "\n".join(lines)
