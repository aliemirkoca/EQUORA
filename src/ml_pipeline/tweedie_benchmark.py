r"""Faz 1 / Ablation Kıyaslaması — Tekil Tweedie Benchmark Modeli (TweedieWasteBenchmark).

Girdi:
    - `GroupTemporalSplitter` çıktısı olan `train_df`, `val_df`, `test_df`
      Polars DataFrame'leri ve kesin özellik şeması.

Çıktı:
    - `models/hurdle_v1/tweedie_benchmark.pkl`: En düşük Validation WAPE veren
      varyans üssüne (p*) sahip tekil LightGBM Tweedie regresörü.
    - `TweedieBenchmarkResult`: Tüm p in {1.2, 1.4, 1.6, 1.8} adaylarının
      Validation özeti ve kazanan p* modelinin Test seti metrikleri (tüm satırlar,
      sıfır-fire yanlış pozitif miktarı, pozitif-fire MAE/RMSE/WAPE, risk sıralaması
      ROC-AUC ve PR-AUC).

Rol:
    - `EQUORA_mimari.md` içinde tanımlanan tekil Tweedie (1 < p < 2) dağılım
      modelini iki aşamalı Hurdle mimarisine karşı objektif ablation benchmark'ı
      olarak eğitir ve değerlendirir. Pandas kullanmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

from src.ml_pipeline.config import MLConfig


@dataclass
class TweedieBenchmarkResult:
    """Tweedie ablation çalışmasının yapılandırılmış değerlendirme çıktısı."""

    best_variance_power: float
    best_iteration: int
    model_path: str
    val_grid_results: list[dict[str, Any]]
    test_all_metrics: dict[str, float]
    test_zero_waste_metrics: dict[str, float]
    test_positive_waste_metrics: dict[str, float]
    test_ranking_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_variance_power": self.best_variance_power,
            "best_iteration": self.best_iteration,
            "model_path": self.model_path,
            "val_grid_results": self.val_grid_results,
            "test_all_metrics": self.test_all_metrics,
            "test_zero_waste_metrics": self.test_zero_waste_metrics,
            "test_positive_waste_metrics": self.test_positive_waste_metrics,
            "test_ranking_metrics": self.test_ranking_metrics,
        }


class TweedieWasteBenchmark:
    """Sıfır-şişkin fire için tekil LightGBM Tweedie (1 < p < 2) kıyaslama sınıfı."""

    def __init__(
        self,
        config: MLConfig,
        feature_columns: list[str],
        categorical_features: list[str],
    ) -> None:
        self.config = config
        self.feature_columns = list(feature_columns)
        self.categorical_features = list(categorical_features)
        self.categorical_feature_indices = [
            self.feature_columns.index(c)
            for c in self.categorical_features
            if c in self.feature_columns
        ]
        self.model: LGBMRegressor | None = None
        self.best_variance_power: float = 1.4

    def _to_matrix(self, df: pl.DataFrame) -> np.ndarray:
        """Polars -> NumPy dönüşümü (sıfır Pandas)."""
        return (
            df.select([pl.col(c).cast(pl.Float64) for c in self.feature_columns])
            .to_numpy()
        )

    @staticmethod
    def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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

    def fit_and_evaluate(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> TweedieBenchmarkResult:
        """p in TWEEDIE_VARIANCE_POWERS adaylarını tarar, en iyi p* modelini kaydedip test eder."""
        self.config.ensure_directories()

        X_train = self._to_matrix(train_df)
        y_train = train_df["final_waste_qty"].to_numpy().astype(np.float64)

        X_val = self._to_matrix(val_df)
        y_val = val_df["final_waste_qty"].to_numpy().astype(np.float64)
        rem_val = val_df["remaining_qty"].to_numpy().astype(np.float64)

        best_model: LGBMRegressor | None = None
        best_p = float(self.config.TWEEDIE_VARIANCE_POWERS[0])
        best_val_wape = float("inf")
        val_grid: list[dict[str, Any]] = []

        for p_power in self.config.TWEEDIE_VARIANCE_POWERS:
            params = dict(self.config.LGBM_REGRESSOR_PARAMS)
            params["objective"] = "tweedie"
            params["tweedie_variance_power"] = float(p_power)
            params["metric"] = ["tweedie", "mae"]

            candidate = LGBMRegressor(**params)
            candidate.fit(
                X_train,
                y_train,
                eval_X=X_val,
                eval_y=y_val,
                categorical_feature=self.categorical_feature_indices,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                        verbose=False,
                    )
                ],
            )
            val_pred = np.clip(candidate.predict(X_val).astype(np.float64), 0.0, rem_val)
            m_val = self._reg_metrics(y_val, val_pred)
            val_grid.append(
                {
                    "variance_power": float(p_power),
                    "best_iteration": int(candidate.best_iteration_ or candidate.n_estimators),
                    "val_mae": m_val["mae"],
                    "val_rmse": m_val["rmse"],
                    "val_wape": m_val["wape"],
                    "val_wape_pct": m_val["wape_pct"],
                }
            )
            if m_val["wape"] < best_val_wape:
                best_val_wape = m_val["wape"]
                best_p = float(p_power)
                best_model = candidate

        assert best_model is not None
        for row in val_grid:
            row["selected"] = row["variance_power"] == best_p

        self.model = best_model
        self.best_variance_power = best_p

        # Kazanan Tweedie modelini kaydet
        save_path: Path = self.config.tweedie_model_path
        joblib.dump(
            {
                "model": self.model,
                "variance_power": self.best_variance_power,
                "feature_columns": self.feature_columns,
            },
            save_path,
        )

        # Out-of-Time Test değerlendirmesi
        X_test = self._to_matrix(test_df)
        y_test = test_df["final_waste_qty"].to_numpy().astype(np.float64)
        y_cls_test = test_df["will_expire_unsold"].to_numpy().astype(np.int32)
        rem_test = test_df["remaining_qty"].to_numpy().astype(np.float64)

        w_pred_test = np.clip(
            self.model.predict(X_test).astype(np.float64),
            0.0,
            rem_test,
        )

        all_metrics = self._reg_metrics(y_test, w_pred_test)

        # Sıfır-fire (W = 0) satırlarındaki yanlış pozitif adet tahmini (Hurdle avantajını görmek için)
        zero_mask = y_test <= 0.0
        pos_mask = y_test > 0.0
        zero_preds = w_pred_test[zero_mask]
        zero_metrics = {
            "zero_waste_rows": int(np.sum(zero_mask)),
            "mean_false_positive_qty": round(float(np.mean(zero_preds)) if len(zero_preds) else 0.0, 4),
            "median_false_positive_qty": round(float(np.median(zero_preds)) if len(zero_preds) else 0.0, 4),
            "pct_pred_ge_1_unit": round(
                100.0 * float(np.mean(zero_preds >= 1.0)) if len(zero_preds) else 0.0,
                2,
            ),
        }

        pos_metrics = self._reg_metrics(y_test[pos_mask], w_pred_test[pos_mask])
        pos_metrics["positive_waste_rows"] = int(np.sum(pos_mask))

        # Tweedie çıktısının risk sıralama (ROC-AUC / PR-AUC) gücü
        ranking_metrics = {
            "roc_auc": round(float(roc_auc_score(y_cls_test, w_pred_test)), 4),
            "pr_auc": round(float(average_precision_score(y_cls_test, w_pred_test)), 4),
        }

        return TweedieBenchmarkResult(
            best_variance_power=best_p,
            best_iteration=int(best_model.best_iteration_ or best_model.n_estimators),
            model_path=str(save_path),
            val_grid_results=val_grid,
            test_all_metrics=all_metrics,
            test_zero_waste_metrics=zero_metrics,
            test_positive_waste_metrics=pos_metrics,
            test_ranking_metrics=ranking_metrics,
        )
