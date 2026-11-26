"""Faz 2A — RL Dinamik İndirim Merkezi Yapılandırma Modülü (RLPricingConfig).

Girdi:
    - Proje kök dizini ve Faz 1 artifact yolları (`models/hurdle_v1/`, `data/ml/`).

Çıktı:
    - Faz 2A ortamı (`EquoraDiscountEnv`), karşı-olgusal talep modeli
      (`CounterfactualDemandModel`), eğitim (`RLPricingTrainer`) ve canlı öneri
      servisi (`LiveDiscountRecommender`) için merkezi `RLPricingConfig` nesnesi.

Rol:
    - İndirim aksiyon uzayını, pencere sınırlarını (`DISCOUNT_START_MAX_DTE`,
      `DISCOUNT_STOP_DTE`), monotonluk/kapı kısıtlarını, birim ekonomi + ESG
      ödül ağırlıklarını, talep elastikiyeti parametrelerini ve eğitim yollarını
      tek noktadan yönetir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_category_elasticity_mult() -> dict[str, float]:
    """Kategori bazlı fiyat esnekliği çarpanları (taze/hızlı tüketim daha esnek)."""
    return {
        "Fırın & Unlu Mamul": 1.15,
        "Meyve & Sebze": 1.10,
        "Hazır Gıda": 1.05,
        "Süt & Süt Ürünleri": 1.00,
        "Şarküteri": 0.90,
        "Et & Tavuk": 0.95,
    }


@dataclass
class RLPricingConfig:
    """EQUORA Faz 2A RL Dinamik İndirim merkezi yapılandırma sınıfı."""

    # -------------------------------------------------------------------------
    # 1) Aksiyon Uzayı (6 Kademeli Ayrık İndirim Oranı)
    # -------------------------------------------------------------------------
    DISCOUNT_GRID: tuple[float, ...] = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)

    # -------------------------------------------------------------------------
    # 2) İndirim Penceresi Sınırları (DTE = days_to_expiry)
    # -------------------------------------------------------------------------
    # SKT'ye en fazla 4 gün kala indirim başlar
    DISCOUNT_START_MAX_DTE: int = 4
    # DTE=2 gün sonuna kadar indirim aktif (DTE=1 artık stoğu Faz 2B sosyal yardıma devredilir)
    # Not: Stres testi / ablation için `0` (SKT gün sonuna kadar indirim) modu desteklenir.
    DISCOUNT_STOP_DTE: int = 2

    # -------------------------------------------------------------------------
    # 3) Operasyonel Kısıtlar (Ablation için açılıp kapatılabilir)
    # -------------------------------------------------------------------------
    ENFORCE_MONOTONIC_DISCOUNT: bool = True   # d_t >= d_{t-1} (indirim geri alınmaz)
    GATE_WITH_PHASE1_THRESHOLD: bool = True   # p_hat < theta* ve prev_d=0 ise 0% maskelenir
    PRICE_CHURN_PENALTY: float = 0.015        # Gereksiz günlük etiket değişim cezası (normalize)

    # -------------------------------------------------------------------------
    # 4) Birim Ekonomi & ESG Ödül Ağırlıkları
    # -------------------------------------------------------------------------
    UNIT_COST_RATIO: float = 0.65             # COGS / unit_price (referans metrikler)
    EQUORA_COMMISSION_RATE: float = 0.08      # İndirimli satışlardan %8 EQUORA komisyonu
    RESIDUAL_WASTE_PENALTY_MULT: float = 0.35 # Pencere sonu artık stoğun imha/devir ceza çarpanı
    CO2_SHADOW_PRICE_TL_PER_KG: float = 2.50  # Önlenen/oluşan CO2e'nin TL gölge fiyatı

    # -------------------------------------------------------------------------
    # 5) Karşı-olgusal Talep Elastikiyeti & Stres Testi Parametreleri
    # -------------------------------------------------------------------------
    ELASTICITY_DIMINISHING_EXP: float = 0.85  # d^0.85 azalan marjinal verim
    CANNIBALIZATION_RATE: float = 0.00        # Temel eğitimde 0.0; stres testinde 0.15
    CATEGORY_ELASTICITY_MULT: dict[str, float] = field(
        default_factory=_default_category_elasticity_mult
    )

    # -------------------------------------------------------------------------
    # 6) RL Eğitim & Epizot Bölme Ayarları
    # -------------------------------------------------------------------------
    TOTAL_TIMESTEPS: int = 200_000
    DQN_TIMESTEPS: int = 120_000
    SEED: int = 42
    TRAIN_EPISODE_RATIO: float = 0.75
    TEST_EPISODE_RATIO: float = 0.25

    # MaskablePPO Hiperparametreleri
    PPO_LEARNING_RATE: float = 3e-4
    PPO_GAMMA: float = 0.99
    PPO_ENT_COEF: float = 0.01
    PPO_N_STEPS: int = 2048
    PPO_BATCH_SIZE: int = 128
    PPO_NET_ARCH: tuple[int, ...] = (128, 128)

    # DQN Kıyaslama Hiperparametreleri
    DQN_LEARNING_RATE: float = 3e-4
    DQN_BUFFER_SIZE: int = 50_000
    DQN_BATCH_SIZE: int = 128
    DQN_EXPLORATION_FRACTION: float = 0.25
    DQN_NET_ARCH: tuple[int, ...] = (128, 128)

    # -------------------------------------------------------------------------
    # 7) Dosya & Klasör Yolları
    # -------------------------------------------------------------------------
    PROJECT_ROOT: Path = PROJECT_ROOT
    ENV_PATH: Path = PROJECT_ROOT / ".env"
    RL_MODEL_DIR: Path = PROJECT_ROOT / "models" / "rl_pricing_v1"
    RL_DATA_DIR: Path = PROJECT_ROOT / "data" / "rl"
    RL_REPORT_DIR: Path = PROJECT_ROOT / "reports" / "phase2a_rl_eval"
    USE_POSTGRES: bool = True

    def __post_init__(self) -> None:
        total = round(self.TRAIN_EPISODE_RATIO + self.TEST_EPISODE_RATIO, 6)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"TRAIN_EPISODE_RATIO + TEST_EPISODE_RATIO = 1.0 olmalıdır, bulunan: {total}"
            )
        if self.DISCOUNT_STOP_DTE < 0 or self.DISCOUNT_STOP_DTE > self.DISCOUNT_START_MAX_DTE:
            raise ValueError(
                f"Geçersiz pencere: DISCOUNT_STOP_DTE={self.DISCOUNT_STOP_DTE}, "
                f"DISCOUNT_START_MAX_DTE={self.DISCOUNT_START_MAX_DTE}"
            )

    def ensure_directories(self) -> None:
        """RL model, veri ve rapor klasörlerini oluşturur."""
        for directory in (self.RL_MODEL_DIR, self.RL_DATA_DIR, self.RL_REPORT_DIR):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def ppo_model_path(self) -> Path:
        return self.RL_MODEL_DIR / "maskable_ppo.zip"

    @property
    def dqn_model_path(self) -> Path:
        return self.RL_MODEL_DIR / "dqn_benchmark.zip"

    @property
    def metadata_path(self) -> Path:
        return self.RL_MODEL_DIR / "rl_metadata.json"

    @property
    def eval_report_path(self) -> Path:
        return self.RL_REPORT_DIR / "rl_evaluation_report.json"

    @property
    def eval_summary_path(self) -> Path:
        return self.RL_REPORT_DIR / "summary.md"

    @property
    def live_recommendations_path(self) -> Path:
        return self.RL_DATA_DIR / "live_discount_recommendations.parquet"
