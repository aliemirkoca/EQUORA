r"""Faz 1 / Adım 4 — İki Aşamalı Hurdle Tahmin Modeli (EquoraHurdleModel).

Girdi:
    - `GroupTemporalSplitter` çıktısı olan `train_df` ve `val_df` Polars
      DataFrame'leri ve `models/hurdle_v1/feature_columns.json`.

Çıktı (Model Artifact'leri):
    - `models/hurdle_v1/classifier.pkl`: Aşama 1 LightGBM risk sınıflandırıcısı.
    - `models/hurdle_v1/regressor.pkl`: Aşama 2 LightGBM koşullu fire regresörü.
    - `models/hurdle_v1/model_metadata.json`: Özellik listesi, kategorik sütun
      indeksleri, en iyi iterasyonlar ve eğitim istatistikleri.

Rol:
    - Sıfır-şişkin (zero-inflated) fire dağılımını iki aşamalı Hurdle mimarisiyle
      modeller:
        1) Risk Sınıflandırıcı (`LGBMClassifier`): Tüm eğitim satırlarında
           `will_expire_unsold` (0/1) olasılığını (\hat{p}) tahmin eder.
        2) Koşullu Fire Regresörü (`LGBMRegressor`): Yalnızca `final_waste_qty > 0`
           olan pozitif fire satırlarında fire adedini (\hat{q}) tahmin eder ve
           mantıksal üst sınır olarak `clip(0, remaining_qty)` uygular.
        3) Birleşik Beklenen Fire: \hat{W} = \hat{p} * \hat{q}.
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

from src.ml_pipeline.config import MLConfig


@dataclass
class HurdlePredictions:
    """EquoraHurdleModel birleşik tahmin çıktıları (NumPy vektörleri)."""

    waste_probability: np.ndarray        # Aşama 1: p_hat in [0, 1]
    predicted_waste_qty: np.ndarray      # Aşama 2: q_hat clipped to [0, remaining_qty]
    expected_waste_qty: np.ndarray       # Birleşik: W_hat = p_hat * q_hat
    action_waste_qty: np.ndarray         # Eşik üstü aksiyon miktarı (p_hat >= theta -> q_hat)


class EquoraHurdleModel:
    """LightGBM tabanlı iki aşamalı (Classifier + Conditional Regressor) Hurdle modeli."""

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
    # Eğitim: Aşama 1 (Sınıflandırıcı) + Aşama 2 (Koşullu Regresör)
    # -------------------------------------------------------------------------
    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Hurdle modelinin her iki aşamasını erken durdurma (early stopping) ile eğitir."""
        if not self.feature_columns:
            self._load_feature_schema()

        # Model sınırında Polars -> NumPy dönüşümü
        X_train = self._to_feature_matrix(train_df)
        y_cls_train = train_df["will_expire_unsold"].to_numpy().astype(np.int32)

        X_val = self._to_feature_matrix(val_df)
        y_cls_val = val_df["will_expire_unsold"].to_numpy().astype(np.int32)

        # ---------------------------------------------------------------------
        # AŞAMA 1 — Risk Sınıflandırıcı (Tüm eğitim satırları)
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

        # ---------------------------------------------------------------------
        # AŞAMA 2 — Koşullu Fire Regresörü (Yalnızca final_waste_qty > 0 satırlar)
        #           Sıfır fireli satırlar elenerek regresörün yalnızca koşullu
        #           beklenen fire miktarını E[Q | Q > 0, X] öğrenmesi sağlanır.
        # ---------------------------------------------------------------------
        train_pos_df = train_df.filter(pl.col("final_waste_qty") > 0)
        val_pos_df = val_df.filter(pl.col("final_waste_qty") > 0)

        X_train_pos = self._to_feature_matrix(train_pos_df)
        y_reg_train_pos = train_pos_df["final_waste_qty"].to_numpy().astype(np.float64)

        X_val_pos = self._to_feature_matrix(val_pos_df)
        y_reg_val_pos = val_pos_df["final_waste_qty"].to_numpy().astype(np.float64)

        self.regressor = LGBMRegressor(**self.config.LGBM_REGRESSOR_PARAMS)
        self.regressor.fit(
            X_train_pos,
            y_reg_train_pos,
            eval_X=X_val_pos,
            eval_y=y_reg_val_pos,
            categorical_feature=self.categorical_feature_indices,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            ],
        )

        self.metadata = {
            "model_version": "hurdle_v1",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": self.config.RANDOM_STATE,
            "feature_columns": self.feature_columns,
            "categorical_features": self.categorical_features,
            "categorical_feature_indices": self.categorical_feature_indices,
            "train_rows_all": int(train_df.height),
            "val_rows_all": int(val_df.height),
            "train_rows_positive_waste": int(train_pos_df.height),
            "val_rows_positive_waste": int(val_pos_df.height),
            "classifier_best_iteration": int(
                self.classifier.best_iteration_ or self.classifier.n_estimators
            ),
            "regressor_best_iteration": int(
                self.regressor.best_iteration_ or self.regressor.n_estimators
            ),
            "classifier_params": self.config.LGBM_CLASSIFIER_PARAMS,
            "regressor_params": self.config.LGBM_REGRESSOR_PARAMS,
        }
        return self.metadata

    # -------------------------------------------------------------------------
    # Tahmin Metodları
    # -------------------------------------------------------------------------
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Aşama 1: Her satır için elde kalma (fire) olasılığını (p_hat) döndürür."""
        if self.classifier is None:
            raise RuntimeError("Sınıflandırıcı eğitilmemiş veya yüklenmemiş.")
        X = self._to_feature_matrix(df)
        return self.classifier.predict_proba(X)[:, 1].astype(np.float64)

    def predict_conditional_qty(self, df: pl.DataFrame) -> np.ndarray:
        """Aşama 2: Koşullu fire adedini (q_hat) tahmin eder ve [0, remaining_qty] aralığına kırpar."""
        if self.regressor is None:
            raise RuntimeError("Regresör eğitilmemiş veya yüklenmemiş.")
        X = self._to_feature_matrix(df)
        raw_qty = self.regressor.predict(X).astype(np.float64)
        # Fiziksel tutarlılık sınırı: Bir partiden kalan stoktan fazla fire çıkamaz!
        upper_bound = df["remaining_qty"].to_numpy().astype(np.float64)
        return np.clip(raw_qty, 0.0, upper_bound)

    def predict_hurdle(
        self,
        df: pl.DataFrame,
        threshold: float = 0.50,
    ) -> HurdlePredictions:
        """Birleşik Hurdle tahminlerini üretir: p_hat, q_hat, W_hat = p_hat * q_hat ve aksiyon adedi."""
        p_hat = self.predict_proba(df)
        q_hat = self.predict_conditional_qty(df)
        upper_bound = df["remaining_qty"].to_numpy().astype(np.float64)

        # Birleşik Beklenen Fire: E[W | X] = P(W > 0 | X) * E[W | W > 0, X]
        expected_w = np.clip(p_hat * q_hat, 0.0, upper_bound)
        # Karar eşiği theta* üstündeki riskli satırlar için önerilen müdahale miktarı
        action_qty = np.where(p_hat >= threshold, q_hat, 0.0)

        return HurdlePredictions(
            waste_probability=p_hat,
            predicted_waste_qty=q_hat,
            expected_waste_qty=expected_w,
            action_waste_qty=action_qty,
        )

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
        """Eğitilmiş modelleri ve metadata dosyasını `models/hurdle_v1/` altına kaydeder."""
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("Kaydedilecek eğitilmiş model bulunamadı.")

        target_dir = Path(model_dir) if model_dir is not None else self.config.MODEL_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        clf_path = target_dir / "classifier.pkl"
        reg_path = target_dir / "regressor.pkl"
        meta_path = target_dir / "model_metadata.json"

        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.regressor, reg_path)
        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "classifier": clf_path,
            "regressor": reg_path,
            "metadata": meta_path,
        }

    @classmethod
    def load(cls, config: MLConfig, model_dir: Path | None = None) -> EquoraHurdleModel:
        """Diskten kaydedilmiş Hurdle modelini ve şema metadata'sını yükler."""
        target_dir = Path(model_dir) if model_dir is not None else config.MODEL_DIR
        clf_path = target_dir / "classifier.pkl"
        reg_path = target_dir / "regressor.pkl"
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
        return instance
