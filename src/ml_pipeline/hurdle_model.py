r"""Faz 1 / Adım 4 — İki Aşamalı Hurdle Tahmin Modeli (EquoraHurdleModel).

Girdi:
    - `GroupTemporalSplitter` çıktısı olan `train_df` ve `val_df` Polars
      DataFrame'leri ve `models/hurdle_v1/feature_columns.json`.

Çıktı (Model Artifact'leri):
    - `models/hurdle_v1/classifier.pkl`: Aşama 1 LightGBM risk sınıflandırıcısı.
    - `models/hurdle_v1/calibrator.pkl`: Aşama 1 olasılık kalibratörü (aktifse).
    - `models/hurdle_v1/regressor.pkl`: Aşama 2 LightGBM koşullu fire regresörü.
    - `models/hurdle_v1/model_metadata.json`: Özellik listesi, kategorik sütun
      indeksleri, kalibrasyon özeti, regresör ablation sonuçları ve eğitim bilgisi.

Rol:
    - Sıfır-şişkin (zero-inflated) fire dağılımını iki aşamalı Hurdle mimarisiyle
      modeller:
        1) Risk Sınıflandırıcı (`LGBMClassifier`) + Sızıntısız Kalibrasyon:
           Tüm eğitim satırlarında `will_expire_unsold` (0/1) olasılığını (\hat{p})
           tahmin eder; Validation ilk yarısında fit edilen Isotonic/Sigmoid
           kalibratörü ikinci yarıda Brier Score iyileştirirse aktif eder.
        2) Koşullu Fire Regresörü (`LGBMRegressor`) + Kayıp/Hedef Ablation:
           Yalnızca `final_waste_qty > 0` satırlarında `raw_l2`, `log1p_l2`
           (Duan smearing düzeltmeli), `raw_l1` ve `poisson` adaylarını yarıştırıp
           en düşük Validation WAPE veren modeli seçer ve `clip(0, remaining_qty)`
           uygular.
        3) Birleşik Beklenen Fire: \hat{W} = \hat{p} * \hat{q}.
        4) Faz 2A RL ortamı için sıfır-overhead hızlı tekil tahmin (`predict_single_fast`).
    - Pandas kullanmaz; Polars -> NumPy (`.to_numpy()`) yalnızca model sınırında
      gerçekleşir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, root_mean_squared_error

from src.ml_pipeline.config import MLConfig


@dataclass
class HurdlePredictions:
    """EquoraHurdleModel birleşik tahmin çıktıları (NumPy vektörleri)."""

    waste_probability: np.ndarray        # Aşama 1: kalibre (veya ham) p_hat in [0, 1]
    raw_waste_probability: np.ndarray    # Aşama 1: kalibrasyon öncesi ham p_hat
    predicted_waste_qty: np.ndarray      # Aşama 2: q_hat clipped to [0, remaining_qty]
    expected_waste_qty: np.ndarray       # Birleşik: W_hat = p_hat * q_hat
    action_waste_qty: np.ndarray         # Eşik üstü aksiyon miktarı (p_hat >= theta -> q_hat)


class EquoraHurdleModel:
    """LightGBM tabanlı iki aşamalı (Calibrated Classifier + Improved Regressor) Hurdle modeli."""

    REGRESSOR_CANDIDATES: tuple[str, ...] = ("raw_l2", "log1p_l2", "raw_l1", "poisson")

    def __init__(
        self,
        config: MLConfig,
        feature_columns: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.config = config
        self.config.ensure_directories()

        self.feature_columns: list[str] = feature_columns or []
        self.categorical_features: list[str] = categorical_features or []
        self.categorical_feature_indices: list[int] = []

        self.classifier: LGBMClassifier | None = None
        self.regressor: LGBMRegressor | None = None
        self.calibrator: IsotonicRegression | LogisticRegression | None = None
        self.calibrator_active: bool = False
        self.calibration_method: str = config.CALIBRATION_METHOD
        self.calibration_val_metrics: dict[str, Any] = {}

        self.regressor_target_mode: str = config.REGRESSOR_TARGET_MODE
        self.duan_smearing_factor: float = 1.0
        self.regressor_ablation_results: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

        if not self.feature_columns and self.config.feature_columns_path.exists():
            self._load_feature_schema()
        elif self.feature_columns:
            self.categorical_feature_indices = [
                self.feature_columns.index(c)
                for c in self.categorical_features
                if c in self.feature_columns
            ]

    def _load_feature_schema(self) -> None:
        """`feature_columns.json` dosyasından kesin özellik ve kategorik indeksleri yükler."""
        payload = json.loads(
            self.config.feature_columns_path.read_text(encoding="utf-8")
        )
        self.feature_columns = list(payload["feature_columns"])
        self.categorical_features = list(payload.get("categorical_features", []))
        self.categorical_feature_indices = [
            self.feature_columns.index(c)
            for c in self.categorical_features
            if c in self.feature_columns
        ]

    def _to_feature_matrix(self, df: pl.DataFrame) -> np.ndarray:
        """Polars DataFrame'i model sınırında Float64 NumPy matrisine çevirir (Pandas yok)."""
        if not self.feature_columns:
            self._load_feature_schema()
        return (
            df.select([pl.col(c).cast(pl.Float64) for c in self.feature_columns])
            .to_numpy()
        )

    # -------------------------------------------------------------------------
    # Kalibrasyon Metrikleri (Brier Score & 10-Bin ECE)
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_ece(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error (ECE, 10 eşit aralıklı olasılık kutusu)."""
        y_t = y_true.astype(np.float64)
        y_p = np.clip(y_prob.astype(np.float64), 0.0, 1.0)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        n = len(y_t)
        if n == 0:
            return 0.0

        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_p >= lo) & (y_p <= hi)
            else:
                mask = (y_p >= lo) & (y_p < hi)
            count = int(np.sum(mask))
            if count > 0:
                avg_conf = float(np.mean(y_p[mask]))
                avg_acc = float(np.mean(y_t[mask]))
                ece += (count / n) * abs(avg_acc - avg_conf)
        return float(ece)

    @classmethod
    def compute_calibration_metrics(
        cls,
        y_true: np.ndarray,
        p_raw: np.ndarray,
        p_cal: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, float]:
        """Ham vs. kalibre olasılıklar için Brier Score ve ECE kıyaslaması üretir."""
        brier_raw = float(brier_score_loss(y_true, np.clip(p_raw, 0.0, 1.0)))
        brier_cal = float(brier_score_loss(y_true, np.clip(p_cal, 0.0, 1.0)))
        ece_raw = cls.compute_ece(y_true, p_raw, n_bins=n_bins)
        ece_cal = cls.compute_ece(y_true, p_cal, n_bins=n_bins)
        return {
            "brier_raw": round(brier_raw, 5),
            "brier_calibrated": round(brier_cal, 5),
            "brier_improvement": round(brier_raw - brier_cal, 5),
            "ece_10bin_raw": round(ece_raw, 5),
            "ece_10bin_calibrated": round(ece_cal, 5),
            "ece_improvement": round(ece_raw - ece_cal, 5),
        }

    # -------------------------------------------------------------------------
    # Yardımcı: Validation'ı Kronolojik Parti Bazlı İkiye Böl (Kalibrasyon Sızıntı Koruması)
    # -------------------------------------------------------------------------
    @staticmethod
    def split_val_chronologically(val_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Validation setini `batch_id` bazında kronolojik iki yarıya (`val_cal`, `val_check`) böler.

        Sızıntı Koruması:
            - İlk yarı (`val_cal`): Olasılık kalibratörünün fit edilmesi için kullanılır.
            - İkinci yarı (`val_check`): Kalibratörün Brier/ECE kontrolü ve eşik
              optimizasyonu için kalibratörden tamamen izole tutulur.
        """
        batches = (
            val_df.select(["batch_id", "first_in_date"])
            .unique()
            .sort(["first_in_date", "batch_id"])
        )
        mid = max(1, batches.height // 2)
        cal_ids = batches.head(mid).select("batch_id")
        chk_ids = batches.tail(batches.height - mid).select("batch_id")

        val_cal = val_df.join(cal_ids, on="batch_id", how="semi")
        val_chk = val_df.join(chk_ids, on="batch_id", how="semi")
        assert len(set(val_cal["batch_id"].to_list()) & set(val_chk["batch_id"].to_list())) == 0
        return val_cal, val_chk

    def _apply_calibrator(self, p_raw: np.ndarray) -> np.ndarray:
        """Aktif kalibratör varsa ham olasılıkları dönüştürür; yoksa ham olasılığı döndürür."""
        if not self.calibrator_active or self.calibrator is None:
            return np.clip(p_raw, 0.0, 1.0)
        if isinstance(self.calibrator, IsotonicRegression):
            return np.clip(
                self.calibrator.predict(p_raw.astype(np.float64)),
                0.0,
                1.0,
            ).astype(np.float64)
        # Platt Sigmoid (LogisticRegression)
        eps = 1e-6
        clipped = np.clip(p_raw.astype(np.float64), eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.clip(
            self.calibrator.predict_proba(logits)[:, 1],
            0.0,
            1.0,
        ).astype(np.float64)

    def _fit_calibrator(self, val_df: pl.DataFrame) -> dict[str, Any]:
        """Validation ilk yarısında kalibratör eğitir, ikinci yarıda Brier iyileşirse aktif eder."""
        method = self.config.CALIBRATION_METHOD.lower().strip()
        if not self.config.ENABLE_PROBABILITY_CALIBRATION or method == "none":
            self.calibrator = None
            self.calibrator_active = False
            self.calibration_method = "none"
            return {"active": False, "method": "none", "reason": "disabled_by_config"}

        val_cal, val_chk = self.split_val_chronologically(val_df)
        p_cal_raw = self.predict_proba_raw(val_cal)
        y_cal = val_cal["will_expire_unsold"].to_numpy().astype(np.int32)

        p_chk_raw = self.predict_proba_raw(val_chk)
        y_chk = val_chk["will_expire_unsold"].to_numpy().astype(np.int32)

        # Isotonic ve Sigmoid adaylarından isteneni eğit
        if method == "isotonic":
            cal: IsotonicRegression | LogisticRegression = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            cal.fit(p_cal_raw, y_cal)
            p_chk_cal = np.clip(cal.predict(p_chk_raw), 0.0, 1.0)
        elif method == "sigmoid":
            eps = 1e-6
            c_raw = np.clip(p_cal_raw, eps, 1.0 - eps)
            logits_cal = np.log(c_raw / (1.0 - c_raw)).reshape(-1, 1)
            cal = LogisticRegression(C=1.0, solver="lbfgs")
            cal.fit(logits_cal, y_cal)
            c_chk = np.clip(p_chk_raw, eps, 1.0 - eps)
            logits_chk = np.log(c_chk / (1.0 - c_chk)).reshape(-1, 1)
            p_chk_cal = np.clip(cal.predict_proba(logits_chk)[:, 1], 0.0, 1.0)
        else:
            raise ValueError(f"Desteklenmeyen CALIBRATION_METHOD: {method!r}")

        metrics = self.compute_calibration_metrics(y_chk, p_chk_raw, p_chk_cal, n_bins=10)
        improved = metrics["brier_calibrated"] < metrics["brier_raw"]

        self.calibration_method = method
        if improved:
            self.calibrator = cal
            self.calibrator_active = True
        else:
            self.calibrator = None
            self.calibrator_active = False

        return {
            "active": self.calibrator_active,
            "method": method,
            "val_cal_rows": int(val_cal.height),
            "val_check_rows": int(val_chk.height),
            **metrics,
        }

    # -------------------------------------------------------------------------
    # Yardımcı: Tek Bir Koşullu Regresör Stratejisini Eğit & Değerlendir
    # -------------------------------------------------------------------------
    def _fit_single_regressor_mode(
        self,
        mode: str,
        X_train_pos: np.ndarray,
        y_train_pos: np.ndarray,
        X_val_pos: np.ndarray,
        y_val_pos: np.ndarray,
        rem_val_pos: np.ndarray,
    ) -> tuple[LGBMRegressor, float, dict[str, Any]]:
        """Belirtilen hedef/kayıp stratejisiyle Aşama 2 koşullu regresörünü eğitir."""
        params = dict(self.config.LGBM_REGRESSOR_PARAMS)
        if mode == "raw_l2":
            params["objective"] = "regression"
            params["metric"] = ["rmse", "mae"]
            y_tr_fit, y_va_fit = y_train_pos, y_val_pos
        elif mode == "log1p_l2":
            params["objective"] = "regression"
            params["metric"] = ["rmse", "mae"]
            # Sağa çarpık fire adetlerini stabilize etmek için log1p dönüşümü
            y_tr_fit = np.log1p(y_train_pos)
            y_va_fit = np.log1p(y_val_pos)
        elif mode == "raw_l1":
            params["objective"] = "regression_l1"
            params["metric"] = ["mae", "rmse"]
            y_tr_fit, y_va_fit = y_train_pos, y_val_pos
        elif mode == "poisson":
            params["objective"] = "poisson"
            params["metric"] = ["poisson", "mae"]
            y_tr_fit, y_va_fit = y_train_pos, y_val_pos
        else:
            raise ValueError(f"Desteklenmeyen REGRESSOR_TARGET_MODE: {mode!r}")

        reg = LGBMRegressor(**params)
        reg.fit(
            X_train_pos,
            y_tr_fit,
            eval_X=X_val_pos,
            eval_y=y_va_fit,
            categorical_feature=self.categorical_feature_indices,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            ],
        )

        # log1p ters dönüşümünde Jensen eşitsizliği yanlılığını gidermek için
        # Duan Smearing faktörü: phi = mean(exp(eps_train))
        # Z = log(1 + Y) => E[Y | X] = exp(Z_hat) * phi - 1
        smearing = 1.0
        if mode == "log1p_l2":
            tr_pred_log = reg.predict(X_train_pos).astype(np.float64)
            residuals = y_tr_fit - tr_pred_log
            smearing = float(np.mean(np.exp(np.clip(residuals, -5.0, 5.0))))
            val_pred_log = reg.predict(X_val_pos).astype(np.float64)
            val_pred_raw = np.exp(val_pred_log) * smearing - 1.0
        else:
            val_pred_raw = reg.predict(X_val_pos).astype(np.float64)

        val_pred = np.clip(val_pred_raw, 0.0, rem_val_pos)
        mae = float(mean_absolute_error(y_val_pos, val_pred))
        rmse = float(root_mean_squared_error(y_val_pos, val_pred))
        wape = float(np.sum(np.abs(y_val_pos - val_pred)) / max(float(np.sum(np.abs(y_val_pos))), 1e-9))

        summary = {
            "mode": mode,
            "val_mae": round(mae, 4),
            "val_rmse": round(rmse, 4),
            "val_wape": round(wape, 4),
            "val_wape_pct": round(wape * 100.0, 2),
            "duan_smearing": round(smearing, 5),
            "best_iteration": int(reg.best_iteration_ or reg.n_estimators),
        }
        return reg, smearing, summary

    def fit_regressor_ablation(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
    ) -> list[dict[str, Any]]:
        """4 koşullu regresör stratejisini (`raw_l2`, `log1p_l2`, `raw_l1`, `poisson`) yarıştırır.

        En düşük Validation WAPE veren aday otomatik seçilerek `self.regressor` olarak atanır.
        """
        train_pos_df = train_df.filter(pl.col("final_waste_qty") > 0)
        val_pos_df = val_df.filter(pl.col("final_waste_qty") > 0)

        X_train_pos = self._to_feature_matrix(train_pos_df)
        y_train_pos = train_pos_df["final_waste_qty"].to_numpy().astype(np.float64)

        X_val_pos = self._to_feature_matrix(val_pos_df)
        y_val_pos = val_pos_df["final_waste_qty"].to_numpy().astype(np.float64)
        rem_val_pos = val_pos_df["remaining_qty"].to_numpy().astype(np.float64)

        candidates = (
            self.REGRESSOR_CANDIDATES
            if self.config.RUN_REGRESSOR_ABLATION
            else (self.config.REGRESSOR_TARGET_MODE,)
        )

        best_wape = float("inf")
        best_reg: LGBMRegressor | None = None
        best_mode = self.config.REGRESSOR_TARGET_MODE
        best_smearing = 1.0
        results: list[dict[str, Any]] = []

        for mode in candidates:
            reg, smearing, summary = self._fit_single_regressor_mode(
                mode=mode,
                X_train_pos=X_train_pos,
                y_train_pos=y_train_pos,
                X_val_pos=X_val_pos,
                y_val_pos=y_val_pos,
                rem_val_pos=rem_val_pos,
            )
            results.append(summary)
            if summary["val_wape"] < best_wape:
                best_wape = summary["val_wape"]
                best_reg = reg
                best_mode = mode
                best_smearing = smearing

        for row in results:
            row["selected"] = row["mode"] == best_mode

        self.regressor = best_reg
        self.regressor_target_mode = best_mode
        self.duan_smearing_factor = best_smearing
        self.regressor_ablation_results = results
        return results

    # -------------------------------------------------------------------------
    # Eğitim: Aşama 1 (Sınıflandırıcı + Kalibrasyon) + Aşama 2 (Regresör Ablation)
    # -------------------------------------------------------------------------
    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Hurdle modelinin her iki aşamasını erken durdurma ve kalibrasyonla eğitir."""
        if not self.feature_columns:
            self._load_feature_schema()

        # Model sınırında Polars -> NumPy dönüşümü
        X_train = self._to_feature_matrix(train_df)
        y_cls_train = train_df["will_expire_unsold"].to_numpy().astype(np.int32)

        X_val = self._to_feature_matrix(val_df)
        y_cls_val = val_df["will_expire_unsold"].to_numpy().astype(np.int32)

        # ---------------------------------------------------------------------
        # AŞAMA 1 — Risk Sınıflandırıcı (Tüm eğitim satırları) + Kalibrasyon
        # ---------------------------------------------------------------------
        self.classifier = LGBMClassifier(**self.config.LGBM_CLASSIFIER_PARAMS)
        self.classifier.fit(
            X_train,
            y_cls_train,
            eval_X=X_val,
            eval_y=y_cls_val,
            categorical_feature=self.categorical_feature_indices,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            ],
        )

        # Validation üzerinde sızıntısız olasılık kalibrasyonu (Brier iyileşirse aktif)
        self.calibration_val_metrics = self._fit_calibrator(val_df)

        # ---------------------------------------------------------------------
        # AŞAMA 2 — Koşullu Fire Regresörü (Yalnızca final_waste_qty > 0 satırlar)
        # ---------------------------------------------------------------------
        ablation_results = self.fit_regressor_ablation(train_df=train_df, val_df=val_df)
        assert self.regressor is not None

        train_pos_count = train_df.filter(pl.col("final_waste_qty") > 0).height
        val_pos_count = val_df.filter(pl.col("final_waste_qty") > 0).height

        self.metadata = {
            "model_version": "hurdle_v1_5",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": self.config.RANDOM_STATE,
            "feature_columns": self.feature_columns,
            "categorical_features": self.categorical_features,
            "categorical_feature_indices": self.categorical_feature_indices,
            "train_rows_all": int(train_df.height),
            "val_rows_all": int(val_df.height),
            "train_rows_positive_waste": int(train_pos_count),
            "val_rows_positive_waste": int(val_pos_count),
            "classifier_best_iteration": int(
                self.classifier.best_iteration_ or self.classifier.n_estimators
            ),
            "regressor_best_iteration": int(
                self.regressor.best_iteration_ or self.regressor.n_estimators
            ),
            "calibrator_active": self.calibrator_active,
            "calibration_method": self.calibration_method,
            "calibration_val_metrics": self.calibration_val_metrics,
            "regressor_target_mode": self.regressor_target_mode,
            "duan_smearing_factor": round(self.duan_smearing_factor, 6),
            "regressor_ablation_results": ablation_results,
            "classifier_params": self.config.LGBM_CLASSIFIER_PARAMS,
            "regressor_params": self.config.LGBM_REGRESSOR_PARAMS,
        }
        return self.metadata

    # -------------------------------------------------------------------------
    # Tahmin Metodları
    # -------------------------------------------------------------------------
    def predict_proba_raw(self, df: pl.DataFrame) -> np.ndarray:
        """Aşama 1: Kalibrasyon öncesi ham fire olasılığını döndürür."""
        if self.classifier is None:
            raise RuntimeError("Sınıflandırıcı eğitilmemiş veya yüklenmemiş.")
        X = self._to_feature_matrix(df)
        return self.classifier.predict_proba(X)[:, 1].astype(np.float64)

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Aşama 1: Aktifse kalibre, değilse ham elde kalma olasılığını (p_hat) döndürür."""
        p_raw = self.predict_proba_raw(df)
        return self._apply_calibrator(p_raw)

    def _inverse_transform_reg(self, raw_pred: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
        """Regresör tahminini hedef uzayından adet uzayına çevirip [0, remaining_qty] kırpar."""
        if self.regressor_target_mode == "log1p_l2":
            qty = np.exp(raw_pred) * self.duan_smearing_factor - 1.0
        else:
            qty = raw_pred
        return np.clip(qty, 0.0, upper_bound)

    def predict_conditional_qty(self, df: pl.DataFrame) -> np.ndarray:
        """Aşama 2: Koşullu fire adedini (q_hat) tahmin eder ve [0, remaining_qty] aralığına kırpar."""
        if self.regressor is None:
            raise RuntimeError("Regresör eğitilmemiş veya yüklenmemiş.")
        X = self._to_feature_matrix(df)
        raw_pred = self.regressor.predict(X).astype(np.float64)
        upper_bound = df["remaining_qty"].to_numpy().astype(np.float64)
        return self._inverse_transform_reg(raw_pred, upper_bound)

    def predict_hurdle(
        self,
        df: pl.DataFrame,
        threshold: float = 0.50,
    ) -> HurdlePredictions:
        """Birleşik Hurdle tahminlerini üretir: p_hat, q_hat, W_hat = p_hat * q_hat ve aksiyon adedi."""
        p_raw = self.predict_proba_raw(df)
        p_hat = self._apply_calibrator(p_raw)
        q_hat = self.predict_conditional_qty(df)
        upper_bound = df["remaining_qty"].to_numpy().astype(np.float64)

        # Birleşik Beklenen Fire: E[W | X] = P(W > 0 | X) * E[W | W > 0, X]
        expected_w = np.clip(p_hat * q_hat, 0.0, upper_bound)
        # Karar eşiği theta* üstündeki riskli satırlar için önerilen müdahale miktarı
        action_qty = np.where(p_hat >= threshold, q_hat, 0.0)

        return HurdlePredictions(
            waste_probability=p_hat,
            raw_waste_probability=p_raw,
            predicted_waste_qty=q_hat,
            expected_waste_qty=expected_w,
            action_waste_qty=action_qty,
        )

    def predict_single_fast(
        self,
        x_1d: np.ndarray,
        remaining_qty: float,
    ) -> tuple[float, float, float]:
        """Faz 2A RL ortamı için sıfır-overhead hızlı tek satırlık (p_hat, q_hat, w_hat) tahmini.

        Sklearn/Polars sarmalayıcısı olmadan doğrudan LightGBM C++ booster'ını çağırır;
        `predict_hurdle` ile sayısal olarak birebir özdeştir.
        """
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("Model eğitilmemiş veya yüklenmemiş.")

        x_2d = np.ascontiguousarray(x_1d.reshape(1, -1), dtype=np.float64)
        p_raw_arr = np.asarray(
            self.classifier.booster_.predict(x_2d, num_threads=1),
            dtype=np.float64,
        )
        p_hat = float(self._apply_calibrator(p_raw_arr)[0])

        q_raw_arr = np.asarray(
            self.regressor.booster_.predict(x_2d, num_threads=1),
            dtype=np.float64,
        )
        ub = np.array([max(0.0, float(remaining_qty))], dtype=np.float64)
        q_hat = float(self._inverse_transform_reg(q_raw_arr, ub)[0])
        w_hat = float(np.clip(p_hat * q_hat, 0.0, ub[0]))
        return p_hat, q_hat, w_hat

    # -------------------------------------------------------------------------
    # Açıklanabilirlik (Feature Importance)
    # -------------------------------------------------------------------------
    def get_feature_importances(
        self,
        top_n: int = 15,
    ) -> dict[str, list[dict[str, Any]]]:
        """Sınıflandırıcı ve regresör için Gain & Split bazlı Top-N özellik önemini döndürür."""
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("Özellik önemi için modeller eğitilmiş veya yüklenmiş olmalıdır.")

        def _extract(model: LGBMClassifier | LGBMRegressor) -> list[dict[str, Any]]:
            booster = model.booster_
            gains = booster.feature_importance(importance_type="gain").astype(np.float64)
            splits = booster.feature_importance(importance_type="split").astype(np.int64)
            total_gain = float(gains.sum()) or 1.0

            rows = [
                {
                    "feature": feat,
                    "gain": round(float(g), 2),
                    "gain_pct": round(100.0 * float(g) / total_gain, 2),
                    "split": int(s),
                }
                for feat, g, s in zip(self.feature_columns, gains, splits)
            ]
            rows.sort(key=lambda r: r["gain"], reverse=True)
            return rows[:top_n]

        return {
            "classifier": _extract(self.classifier),
            "regressor": _extract(self.regressor),
        }

    # -------------------------------------------------------------------------
    # Kaydetme ve Yükleme
    # -------------------------------------------------------------------------
    def save(self, model_dir: Path | None = None) -> dict[str, Path]:
        """Eğitilmiş modelleri, kalibratörü ve metadata dosyasını `models/hurdle_v1/` altına kaydeder."""
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("Kaydedilecek eğitilmiş model bulunamadı.")

        target_dir = Path(model_dir) if model_dir is not None else self.config.MODEL_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        clf_path = target_dir / "classifier.pkl"
        reg_path = target_dir / "regressor.pkl"
        cal_path = target_dir / "calibrator.pkl"
        meta_path = target_dir / "model_metadata.json"

        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.regressor, reg_path)
        if self.calibrator_active and self.calibrator is not None:
            joblib.dump(self.calibrator, cal_path)
        elif cal_path.exists():
            cal_path.unlink()

        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths = {
            "classifier": clf_path,
            "regressor": reg_path,
            "metadata": meta_path,
        }
        if self.calibrator_active and self.calibrator is not None:
            paths["calibrator"] = cal_path
        return paths

    @classmethod
    def load(cls, config: MLConfig, model_dir: Path | None = None) -> EquoraHurdleModel:
        """Diskten kaydedilmiş Hurdle modelini, kalibratörünü ve şema metadata'sını yükler."""
        target_dir = Path(model_dir) if model_dir is not None else config.MODEL_DIR
        clf_path = target_dir / "classifier.pkl"
        reg_path = target_dir / "regressor.pkl"
        cal_path = target_dir / "calibrator.pkl"
        meta_path = target_dir / "model_metadata.json"

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        instance = cls(
            config=config,
            feature_columns=list(metadata["feature_columns"]),
            categorical_features=list(metadata.get("categorical_features", [])),
        )
        instance.classifier = joblib.load(clf_path)
        instance.regressor = joblib.load(reg_path)
        instance.metadata = metadata
        instance.calibrator_active = bool(metadata.get("calibrator_active", False))
        instance.calibration_method = str(metadata.get("calibration_method", "none"))
        instance.calibration_val_metrics = dict(metadata.get("calibration_val_metrics", {}))
        instance.regressor_target_mode = str(metadata.get("regressor_target_mode", "raw_l2"))
        instance.duan_smearing_factor = float(metadata.get("duan_smearing_factor", 1.0))
        instance.regressor_ablation_results = list(metadata.get("regressor_ablation_results", []))

        if instance.calibrator_active and cal_path.exists():
            instance.calibrator = joblib.load(cal_path)
        else:
            instance.calibrator = None
            instance.calibrator_active = False
        return instance
