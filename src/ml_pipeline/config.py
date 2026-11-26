"""Faz 1 — Merkezi ML Yapılandırma Modülü (MLConfig).

Girdi:
    - Proje kök dizin yapısı ve `.env` dosyası (veritabanı bağlantısı için).

Çıktı:
    - Tüm Faz 1 ML modüllerinin (Snapshot, Feature Engineering, Splitter,
      Hurdle Model, Evaluator, Live Inference) ortak kullandığı `MLConfig`
      dataclass yapılandırma nesnesi.

Rol:
    - Dosya/klasör yollarını, veri kaynağı bayrağını (`USE_POSTGRES`),
      zaman bazlı bölme oranlarını, asimetrik maliyet katsayılarını,
      LightGBM hiperparametrelerini ve canlı aksiyon yönlendirme eşiklerini
      tek merkezde yönetir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_lgbm_classifier_params() -> dict[str, Any]:
    """Aşama 1 (Risk Sınıflandırıcı) varsayılan LightGBM hiperparametreleri."""
    return {
        "objective": "binary",
        "metric": ["binary_logloss", "average_precision"],
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def _default_lgbm_regressor_params() -> dict[str, Any]:
    """Aşama 2 (Koşullu Fire Regresörü) varsayılan LightGBM hiperparametreleri."""
    return {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


@dataclass
class MLConfig:
    """EQUORA Faz 1 Hurdle ML Pipeline merkezi yapılandırma sınıfı."""

    # -------------------------------------------------------------------------
    # 1) Proje ve Ara Çıktı (Checkpoint) Yolları
    # -------------------------------------------------------------------------
    PROJECT_ROOT: Path = PROJECT_ROOT
    ENV_PATH: Path = PROJECT_ROOT / ".env"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    ML_DATA_DIR: Path = PROJECT_ROOT / "data" / "ml"
    MODEL_DIR: Path = PROJECT_ROOT / "models" / "hurdle_v1"
    REPORT_DIR: Path = PROJECT_ROOT / "reports" / "phase1_eval"

    # -------------------------------------------------------------------------
    # 2) Veri Kaynağı ve Zaman Dilimi Ayarları
    # -------------------------------------------------------------------------
    # True -> Yerel PostgreSQL `equora_db` üzerinden okur (.env)
    # False -> `data/processed/*.parquet` dosyalarından okur
    USE_POSTGRES: bool = True
    # Simülatör (fefo_simulator.py) Türkiye saat dilimi (UTC+3) kullandığı için
    # gün sınırlarının (özellikle 00:01 WASTE kayıtlarının) şaşmaması adına TR TZ kullanılır.
    TIMEZONE: str = "Europe/Istanbul"

    # -------------------------------------------------------------------------
    # 3) Grup-Zaman Bölme Oranları (Parti Bazlı Kronolojik Split)
    # -------------------------------------------------------------------------
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # -------------------------------------------------------------------------
    # 4) Asimetrik Maliyet Katsayıları ve Eşik Arama Izgarası
    # -------------------------------------------------------------------------
    # Legacy (Mod 1) katsayıları — geriye dönük kıyaslama (before/after) için korunur
    # FN: Fire olacak ürünü kaçırma maliyeti = birim fiyatın %100'ü kayıp
    FN_COST_MULT: float = 1.00
    # FP: Satılacak ürüne gereksiz indirim yapma maliyeti = %25 marj kaybı
    FP_COST_MULT: float = 0.25
    THRESHOLD_MIN: float = 0.10
    THRESHOLD_MAX: float = 0.90
    THRESHOLD_STEP: float = 0.01

    # --- Faz 1.5 Rötuş Parametreleri ---
    # Aşama 1 olasılık kalibrasyonu (Brier Score & ECE iyileştirmesi)
    ENABLE_PROBABILITY_CALIBRATION: bool = True
    CALIBRATION_METHOD: str = "isotonic"  # "isotonic" | "sigmoid" | "none"

    # Aşama 2 koşullu regresör hedef/kayıp stratejisi ve ablation bayrağı
    # Adaylar: "raw_l2" | "log1p_l2" | "raw_l1" | "poisson"
    REGRESSOR_TARGET_MODE: str = "log1p_l2"
    RUN_REGRESSOR_ABLATION: bool = True

    # Tekil Tweedie Benchmark (1 < p < 2) ablation kıyaslaması
    RUN_TWEEDIE_BENCHMARK: bool = True
    TWEEDIE_VARIANCE_POWERS: tuple[float, ...] = (1.2, 1.4, 1.6, 1.8)

    # Ekonomik-Tutarlı Maliyet Matrisi (Mod 2 — Faz 2A RL ile ortak birim ekonomisi)
    UNIT_COST_RATIO: float = 0.65          # COGS / unit_price
    DISPOSAL_COST_RATIO: float = 0.05      # İmha & operasyon / unit_price
    FP_DISCOUNT_MARGIN_LOSS: float = 0.20  # Gereksiz indirim ortalama marj kaybı
    THRESHOLD_OPT_MAX_DTE: int = 4         # Eşik optimizasyonunda aksiyon penceresi üst sınırı

    # -------------------------------------------------------------------------
    # 5) Model Hiperparametreleri ve Tekrarlanabilirlik
    # -------------------------------------------------------------------------
    RANDOM_STATE: int = 42
    EARLY_STOPPING_ROUNDS: int = 50
    LGBM_CLASSIFIER_PARAMS: dict[str, Any] = field(
        default_factory=_default_lgbm_classifier_params
    )
    LGBM_REGRESSOR_PARAMS: dict[str, Any] = field(
        default_factory=_default_lgbm_regressor_params
    )

    # -------------------------------------------------------------------------
    # 6) Canlı Aksiyon Yönlendirme Eşikleri (DTE = days_to_expiry)
    # -------------------------------------------------------------------------
    # DYNAMIC_DISCOUNT öncelikli pencere: DTE 2-4 (genel kural: DTE >= 2 & p >= theta*)
    DYNAMIC_DISCOUNT_MIN_DTE: int = 2
    DYNAMIC_DISCOUNT_MAX_DTE: int = 4
    SOCIAL_AID_DTE: int = 1
    URGENT_BULK_MAX_DTE: int = 0

    def __post_init__(self) -> None:
        """Oranların toplamını ve sabit random_state değerini doğrular."""
        total_ratio = round(self.TRAIN_RATIO + self.VAL_RATIO + self.TEST_RATIO, 6)
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO = 1.0 olmalıdır, bulunan: {total_ratio}"
            )
        # Hiperparametre sözlüklerinde random_state'in her zaman tutarlı olmasını garanti et
        self.LGBM_CLASSIFIER_PARAMS["random_state"] = self.RANDOM_STATE
        self.LGBM_REGRESSOR_PARAMS["random_state"] = self.RANDOM_STATE

    def ensure_directories(self) -> None:
        """Tüm çıktı ve checkpoint klasörlerini oluşturur."""
        for directory in (self.ML_DATA_DIR, self.MODEL_DIR, self.REPORT_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Standart Dosya Yolları (Property'ler)
    # -------------------------------------------------------------------------
    @property
    def products_parquet_path(self) -> Path:
        return self.PROCESSED_DIR / "products.parquet"

    @property
    def movements_parquet_path(self) -> Path:
        return self.PROCESSED_DIR / "stock_movements.parquet"

    @property
    def snapshots_path(self) -> Path:
        return self.ML_DATA_DIR / "snapshots.parquet"

    @property
    def active_snapshots_path(self) -> Path:
        return self.ML_DATA_DIR / "active_snapshots.parquet"

    @property
    def daily_sales_path(self) -> Path:
        """Satış hızı ve gecikmeli indirim hesabı için tam takvimli günlük satış özeti."""
        return self.ML_DATA_DIR / "daily_store_product_sales.parquet"

    @property
    def features_path(self) -> Path:
        return self.ML_DATA_DIR / "features.parquet"

    @property
    def active_features_path(self) -> Path:
        return self.ML_DATA_DIR / "active_features.parquet"

    @property
    def live_predictions_path(self) -> Path:
        return self.ML_DATA_DIR / "live_predictions.parquet"

    @property
    def feature_columns_path(self) -> Path:
        return self.MODEL_DIR / "feature_columns.json"

    @property
    def threshold_path(self) -> Path:
        return self.MODEL_DIR / "threshold.json"

    @property
    def classifier_path(self) -> Path:
        return self.MODEL_DIR / "classifier.pkl"

    @property
    def regressor_path(self) -> Path:
        return self.MODEL_DIR / "regressor.pkl"

    @property
    def calibrator_path(self) -> Path:
        return self.MODEL_DIR / "calibrator.pkl"

    @property
    def tweedie_model_path(self) -> Path:
        return self.MODEL_DIR / "tweedie_benchmark.pkl"

    @property
    def model_metadata_path(self) -> Path:
        return self.MODEL_DIR / "model_metadata.json"

    @property
    def eval_report_path(self) -> Path:
        return self.REPORT_DIR / "evaluation_report.json"

    @property
    def eval_summary_path(self) -> Path:
        return self.REPORT_DIR / "summary.md"
