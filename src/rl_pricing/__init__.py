"""EQUORA Faz 2A — RL Tabanlı Dinamik İndirim Motoru (`src/rl_pricing/`).

Bu paket, dondurulmuş Faz 1 `EquoraHurdleModel` risk tahminlerini ve mevcut
`fefo_simulator.py` talep/FEFO fiziğini kullanarak Gymnasium uyumlu kapalı döngü
dinamik fiyatlama ortamı, MaskablePPO / Double DQN ajanları, 5+1 politika
kıyaslaması, stres testleri ve canlı indirim öneri servisini sunar.
"""

from src.rl_pricing.baselines import (
    BasePricingPolicy,
    MyopicExpectedValuePolicy,
    NoDiscountPolicy,
    Phase1MLFixedDiscountPolicy,
    StaticRulePolicy,
)
from src.rl_pricing.config import RLPricingConfig
from src.rl_pricing.demand_model import CounterfactualDemandModel
from src.rl_pricing.discount_env import EquoraDiscountEnv
from src.rl_pricing.episode_builder import BatchEpisode, DiscountEpisodeBuilder
from src.rl_pricing.live_recommender import LiveDiscountRecommender
from src.rl_pricing.train_rl import RLPricingTrainer

__all__ = [
    "RLPricingConfig",
    "CounterfactualDemandModel",
    "BatchEpisode",
    "DiscountEpisodeBuilder",
    "EquoraDiscountEnv",
    "BasePricingPolicy",
    "NoDiscountPolicy",
    "StaticRulePolicy",
    "Phase1MLFixedDiscountPolicy",
    "MyopicExpectedValuePolicy",
    "RLPricingTrainer",
    "LiveDiscountRecommender",
]
