r"""Faz 2A — Karşı-olgusal Talep Tepki ve FEFO Satış Fiziği Modeli (CounterfactualDemandModel).

Girdi:
    - `src/data_pipeline/fefo_simulator.py` içindeki `DemandProfileExtractor`
      (`fresh_retail`, `retail_markdown`, `grocery_inventory` istatistiksel profili),
      `CATEGORY_CALIBRATION` ve `STORE_MULTIPLIERS`.
    - `RLPricingConfig` elastikiyet, azalan marjinal verim ve yamyamlaşma ayarları.

Çıktı:
    - Verilen parti durumu ve seçilen indirim oranı (`d_t`) için:
      * Analitik beklenen talep ve talep artışı (`expected_demand_lift`),
      * Overdispersed Gamma-Poisson (Negative Binomial) günlük talep örneklemesi,
      * FEFO ön kuyruğu (`upstream_fefo_qty`) sonrası partiden gerçekleşen satış
        (`batch_sold`) ve güncellenmiş kuyruk/stok değerleri.

Rol:
    - RL ortamının (`EquoraDiscountEnv`), miyopik analitik kıyaslama politikasının
      (`MyopicExpectedValuePolicy`) ve canlı öneri servisinin (`LiveDiscountRecommender`)
      ortak fizik motorudur. Veri sızıntısız ve tam deterministik seed desteklidir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data_pipeline.fefo_simulator import (
    DemandProfileExtractor,
    FefoInventorySimulator,
    StatisticalDemandProfile,
)
from src.rl_pricing.config import RLPricingConfig


@dataclass(frozen=True)
class DailyMarketOutcome:
    """Tek günlük FEFO satış simülasyonunun fiziksel çıktısı."""

    expected_demand: float
    demand_lift: float
    realized_total_demand: int
    upstream_sold: int
    batch_sold: int
    new_upstream_qty: int
    new_remaining_qty: int
    weather_shock_occurred: bool


class CounterfactualDemandModel:
    """Gamma-Poisson talep tepki ve FEFO ön-kuyruk erime fiziği motoru."""

    def __init__(
        self,
        config: RLPricingConfig,
        profile: StatisticalDemandProfile | None = None,
    ) -> None:
        self.config = config
        self.profile = profile or DemandProfileExtractor().extract()
        self.category_calibration = FefoInventorySimulator.CATEGORY_CALIBRATION
        self.store_multipliers = FefoInventorySimulator.STORE_MULTIPLIERS

    def get_category_elasticity(self, category: str) -> float:
        """Kategoriye özgü indirim esnekliği katsayısı: eta_c = eta_profile * mult[c]."""
        mult = self.config.CATEGORY_ELASTICITY_MULT.get(category, 1.0)
        return float(self.profile.markdown_elasticity * mult)

    def get_dispersion(self, category: str) -> float:
        """Kategoriye özgü Gamma-Poisson oynaklık (dispersion) katsayısı."""
        calib = self.category_calibration.get(category)
        return float(calib.demand_dispersion) if calib is not None else 0.52

    def get_store_multiplier(self, store_id: int) -> float:
        """Şube bazlı trafik/talep çarpanı."""
        return float(self.store_multipliers.get(store_id, 1.0))

    @staticmethod
    def freshness_factor(days_to_expiry: int) -> float:
        """SKT yaklaştıkça tüketicinin tazelik algısı çarpanı phi(DTE).

        DTE >= 2 -> 1.00 | DTE == 1 -> 0.92 | DTE <= 0 -> 0.82
        """
        if days_to_expiry >= 2:
            return 1.00
        if days_to_expiry == 1:
            return 0.92
        return 0.82

    def calendar_multiplier(self, is_weekend: bool, is_payday_window: bool) -> float:
        """Hafta içi / hafta sonu ve maaş dönemi takvim talep çarpanı."""
        dow_mult = (
            self.profile.weekend_multiplier
            if is_weekend
            else self.profile.weekday_multiplier
        )
        payday_mult = 1.05 if is_payday_window else 1.00
        return float(dow_mult * payday_mult)

    def expected_weather_multiplier(self) -> float:
        """Beklenen ortalama hava şoku çarpanı (analitik hesaplar için)."""
        p_shock = float(self.profile.weather_shock_prob)
        m_shock = float(self.profile.weather_shock_multiplier)
        return (1.0 - p_shock) * 1.0 + p_shock * m_shock

    def estimate_base_daily_demand(
        self,
        category: str,
        store_id: int,
        vel_14d: float,
        vel_7d: float,
    ) -> float:
        """Ürün-şube baz günlük temel talebi (mu_base) gözlenen hızla kalibreli tahmin eder."""
        calib = self.category_calibration.get(category)
        prior_mu = (
            calib.base_daily_demand * self.get_store_multiplier(store_id)
            if calib is not None
            else 8.0
        )
        obs_vel = 0.65 * max(0.0, vel_14d) + 0.35 * max(0.0, vel_7d)
        if obs_vel > 0.1:
            # Gözlenen ürün popülerliğini korurken gürültüyü kategori prior'u ile yumuşat
            return float(max(0.5, 0.80 * obs_vel + 0.20 * prior_mu))
        return float(max(0.5, prior_mu))

    def compute_demand_lift(
        self,
        category: str,
        discount_rate: float,
        days_to_expiry: int,
        prev_discount_rate: float = 0.0,
    ) -> float:
        """İndirimin talep üzerindeki çarpan etkisini hesaplar.

        Formül:
            lift = (1 + eta_c * (d_t ^ kappa) * phi(DTE)) * (1 - beta_cann * d_{t-1})
        """
        d = max(0.0, min(0.50, float(discount_rate)))
        d_prev = max(0.0, min(0.50, float(prev_discount_rate)))
        eta_c = self.get_category_elasticity(category)
        phi = self.freshness_factor(days_to_expiry)

        # Azalan marjinal verim (d^0.85): %10->%20 artışı %40->%50'den daha verimlidir
        markdown_boost = 1.0 + eta_c * (d ** self.config.ELASTICITY_DIMINISHING_EXP) * phi
        # Stratejik tüketici / talep öne çekme (cannibalization) düzeltmesi
        cann_factor = max(0.50, 1.0 - self.config.CANNIBALIZATION_RATE * d_prev)
        return float(markdown_boost * cann_factor)

    def expected_daily_demand(
        self,
        base_mu: float,
        category: str,
        discount_rate: float,
        days_to_expiry: int,
        is_weekend: bool,
        is_payday_window: bool,
        prev_discount_rate: float = 0.0,
        weather_multiplier: float | None = None,
    ) -> tuple[float, float]:
        """Beklenen günlük toplam talebi (mu_bt) ve indirim lift çarpanını döndürür."""
        cal_mult = self.calendar_multiplier(is_weekend, is_payday_window)
        w_mult = (
            self.expected_weather_multiplier()
            if weather_multiplier is None
            else float(weather_multiplier)
        )
        lift = self.compute_demand_lift(
            category=category,
            discount_rate=discount_rate,
            days_to_expiry=days_to_expiry,
            prev_discount_rate=prev_discount_rate,
        )
        expected_mu = max(0.0, base_mu * cal_mult * w_mult * lift)
        return float(expected_mu), float(lift)

    def expected_batch_sales(
        self,
        base_mu: float,
        category: str,
        discount_rate: float,
        days_to_expiry: int,
        is_weekend: bool,
        is_payday_window: bool,
        remaining_qty: int,
        upstream_fefo_qty: int,
        prev_discount_rate: float = 0.0,
    ) -> tuple[float, float]:
        """FEFO ön kuyruğu dikkate alınarak partiden beklenen günlük satış ve lift döner."""
        exp_mu, lift = self.expected_daily_demand(
            base_mu=base_mu,
            category=category,
            discount_rate=discount_rate,
            days_to_expiry=days_to_expiry,
            is_weekend=is_weekend,
            is_payday_window=is_payday_window,
            prev_discount_rate=prev_discount_rate,
        )
        # Zero-sale frekansı düzeltmesi (fefo_simulator.py ile uyumlu)
        eff_mu = exp_mu * (1.0 - self.profile.zero_sale_prob * 0.5)
        # Ön kuyruk erimeden bu partiden satış olmaz
        after_upstream = max(0.0, eff_mu - float(upstream_fefo_qty))
        exp_batch_sold = min(float(remaining_qty), after_upstream)
        return float(exp_batch_sold), float(lift)

    def simulate_day(
        self,
        rng: np.random.Generator,
        base_mu: float,
        category: str,
        discount_rate: float,
        days_to_expiry: int,
        is_weekend: bool,
        is_payday_window: bool,
        remaining_qty: int,
        upstream_fefo_qty: int,
        prev_discount_rate: float = 0.0,
    ) -> DailyMarketOutcome:
        """Tek günlük stokastik Gamma-Poisson talep ve FEFO satış adımını simüle eder."""
        # 1) Hava / ani talep şoku örneklemesi
        shock_occurred = bool(rng.random() < self.profile.weather_shock_prob)
        w_mult = self.profile.weather_shock_multiplier if shock_occurred else 1.0

        exp_mu, lift = self.expected_daily_demand(
            base_mu=base_mu,
            category=category,
            discount_rate=discount_rate,
            days_to_expiry=days_to_expiry,
            is_weekend=is_weekend,
            is_payday_window=is_payday_window,
            prev_discount_rate=prev_discount_rate,
            weather_multiplier=w_mult,
        )

        # 2) Overdispersed Gamma-Poisson (Negative Binomial) talep örneklemesi
        #    (src/data_pipeline/fefo_simulator.py _sample_daily_demand ile birebir aynı fizik)
        if exp_mu <= 0.05 or rng.random() < self.profile.zero_sale_prob * 0.5:
            realized = 0
        else:
            disp = self.get_dispersion(category)
            shape = 1.0 / max(disp * disp, 0.05)
            scale = exp_mu / shape
            lam = float(rng.gamma(shape=shape, scale=scale))
            realized = int(rng.poisson(max(lam, 0.0)))

        # 3) FEFO Kuyruk Fiziği:
        #    Önce daha yakın SKT'li upstream_fefo_qty tükenir, kalan talep hedef partiyi eritir.
        upstream_sold = min(int(upstream_fefo_qty), realized)
        rem_demand = realized - upstream_sold
        batch_sold = min(int(remaining_qty), rem_demand)

        return DailyMarketOutcome(
            expected_demand=exp_mu,
            demand_lift=lift,
            realized_total_demand=realized,
            upstream_sold=upstream_sold,
            batch_sold=batch_sold,
            new_upstream_qty=int(upstream_fefo_qty) - upstream_sold,
            new_remaining_qty=int(remaining_qty) - batch_sold,
            weather_shock_occurred=shock_occurred,
        )
