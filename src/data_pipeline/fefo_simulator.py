"""Adım 3 — İstatistiksel Profil Çıkarımı & FEFO Simülatörü.

- DemandProfileExtractor:
    * fresh_retail/train.parquet (Polars LazyFrame): günlük satış dağılımı,
      hafta sonu/hafta içi çarpanı, kötü hava/talep şoku frekansı ve saatlik
      satış yoğunluğu.
    * retail_markdown/SYNTHETIC Markdown Dataset.csv (Polars LazyFrame):
      indirim oranına (0.00 - 0.50) bağlı fiyat esnekliği / satış artışı katsayısı.
    * grocery_inventory/Grocery_Inventory_and_Sales_Dataset.csv (Polars LazyFrame):
      kategori bazlı yeniden sipariş (reorder_level, reorder_qty) oranları.
- FefoInventorySimulator:
    3 farklı şube (store_id = 1, 2, 3) için son 180 gün boyunca günlük FEFO
    (First Expired, First Out) stok hareket simülasyonu üretir.

Not: Yalnızca Polars (LazyFrame / scan_parquet / scan_csv) ve NumPy kullanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FRESH_PARQUET = PROJECT_ROOT / "raw_data" / "fresh_retail" / "train.parquet"
DEFAULT_MARKDOWN_CSV = (
    PROJECT_ROOT / "raw_data" / "retail_markdown" / "SYNTHETIC Markdown Dataset.csv"
)
DEFAULT_GROCERY_CSV = (
    PROJECT_ROOT / "raw_data" / "grocery_inventory" / "Grocery_Inventory_and_Sales_Dataset.csv"
)

TR_TZ = timezone(timedelta(hours=3))


@dataclass
class CategoryReorderProfile:
    """grocery_inventory verisinden çıkarılan kategori bazlı yeniden sipariş oranları."""

    reorder_level_ratio: float
    reorder_qty_ratio: float
    turnover_rate: float


@dataclass
class StatisticalDemandProfile:
    """3 dış veri setinden Polars LazyFrame ile çıkarılan birleşik istatistiksel profil."""

    daily_sale_mean: float
    daily_sale_std: float
    daily_sale_cv: float
    zero_sale_prob: float
    weekday_multiplier: float
    weekend_multiplier: float
    weather_shock_prob: float
    weather_shock_multiplier: float
    holiday_multiplier: float
    hourly_sale_probs: list[float]  # 24 saatlik normalize olasılıklar
    markdown_elasticity: float
    markdown_tier_lifts: dict[float, float] = field(default_factory=dict)
    reorder_profiles: dict[str, CategoryReorderProfile] = field(default_factory=dict)


@dataclass
class InventoryBatch:
    """Tek bir SKT partisini temsil eden FEFO stok birimi."""

    expiry_date: date
    remaining_qty: int
    in_date: date


class DemandProfileExtractor:
    """fresh_retail, retail_markdown ve grocery_inventory setlerinden profil çıkarır."""

    EQUORA_TO_GROCERY_MAP: dict[str, str] = {
        "Süt & Süt Ürünleri": "Dairy",
        "Fırın & Unlu Mamul": "Bakery",
        "Et & Tavuk": "Seafood",
        "Şarküteri": "Dairy",
        "Meyve & Sebze": "Fruits & Vegetables",
        "Hazır Gıda": "Bakery",
    }

    def __init__(
        self,
        fresh_parquet_path: Path | str = DEFAULT_FRESH_PARQUET,
        markdown_csv_path: Path | str = DEFAULT_MARKDOWN_CSV,
        grocery_csv_path: Path | str = DEFAULT_GROCERY_CSV,
    ) -> None:
        self.fresh_parquet_path = Path(fresh_parquet_path)
        self.markdown_csv_path = Path(markdown_csv_path)
        self.grocery_csv_path = Path(grocery_csv_path)

    def _extract_fresh_retail(self) -> dict[str, object]:
        """FreshRetailNet-50K Parquet verisinden talep, takvim, hava ve saatlik profil çıkarır."""
        lf = pl.scan_parquet(self.fresh_parquet_path)

        # 1) Temel günlük satış istatistikleri ve yağış eşiği
        base_stats = lf.select(
            pl.col("sale_amount").mean().alias("mean_sale"),
            pl.col("sale_amount").std().alias("std_sale"),
            (pl.col("sale_amount") <= 0.0).mean().alias("zero_prob"),
            pl.col("precpt").quantile(0.90).alias("precpt_p90"),
        ).collect()

        mean_sale = float(base_stats["mean_sale"][0] or 1.0)
        std_sale = float(base_stats["std_sale"][0] or 0.6)
        zero_prob = float(base_stats["zero_prob"][0] or 0.05)
        precpt_p90 = float(base_stats["precpt_p90"][0] or 8.95)
        cv = max(0.35, min(1.25, std_sale / max(mean_sale, 1e-6)))

        # 2) Hafta içi vs. hafta sonu çarpanları (dt -> weekday: 1=Pzt .. 7=Paz)
        dow_stats = (
            lf.select(
                (pl.col("dt").str.to_date("%Y-%m-%d", strict=False).dt.weekday() >= 6)
                .alias("is_weekend"),
                pl.col("sale_amount"),
            )
            .group_by("is_weekend")
            .agg(pl.col("sale_amount").mean().alias("avg_sale"))
            .collect()
        )
        dow_map = {bool(r["is_weekend"]): float(r["avg_sale"]) for r in dow_stats.iter_rows(named=True)}
        wd_raw = dow_map.get(False, mean_sale)
        we_raw = dow_map.get(True, mean_sale * 1.15)
        weighted_avg = (5.0 * wd_raw + 2.0 * we_raw) / 7.0
        weekday_mult = wd_raw / max(weighted_avg, 1e-6)
        weekend_mult = we_raw / max(weighted_avg, 1e-6)
        # Türkiye market hafta sonu dinamiğini koruyarak aşırı düzleşmeyi önle
        if weekend_mult < 1.10:
            weekend_mult = max(weekend_mult, 1.18)
            weekday_mult = (7.0 - 2.0 * weekend_mult) / 5.0

        # 3) Kötü hava / talep şoku frekansı ve etkisi
        weather_stats = (
            lf.select(
                (pl.col("precpt") >= precpt_p90).alias("is_shock"),
                pl.col("sale_amount"),
            )
            .group_by("is_shock")
            .agg(
                pl.len().alias("cnt"),
                pl.col("sale_amount").mean().alias("avg_sale"),
            )
            .collect()
        )
        w_map = {bool(r["is_shock"]): (int(r["cnt"]), float(r["avg_sale"])) for r in weather_stats.iter_rows(named=True)}
        shock_cnt, shock_avg = w_map.get(True, (1, mean_sale * 0.85))
        norm_cnt, norm_avg = w_map.get(False, (9, mean_sale))
        shock_prob = shock_cnt / max(shock_cnt + norm_cnt, 1)
        raw_shock_mult = shock_avg / max(norm_avg, 1e-6)
        # Kötü hava günlerinde fiziksel mağaza trafiğinde makul negatif talep şoku (0.65-0.88)
        weather_shock_mult = min(0.85, max(0.65, raw_shock_mult * 0.82))

        # 4) Tatil/kampanya çarpanı
        hol_stats = (
            lf.group_by("holiday_flag")
            .agg(pl.col("sale_amount").mean().alias("avg_sale"))
            .collect()
        )
        h_map = {int(r["holiday_flag"]): float(r["avg_sale"]) for r in hol_stats.iter_rows(named=True)}
        holiday_mult = h_map.get(1, mean_sale) / max(h_map.get(0, mean_sale), 1e-6)
        holiday_mult = max(0.95, min(1.30, holiday_mult))

        # 5) 24 saatlik satış yoğunluğu (hours_sale: List(Float64))
        hour_exprs = [
            pl.col("hours_sale").list.get(h, null_on_oob=True).mean().alias(f"h_{h}")
            for h in range(24)
        ]
        hourly_row = lf.select(hour_exprs).collect().to_dicts()[0]
        raw_hours = [max(0.0, float(hourly_row.get(f"h_{h}") or 0.0)) for h in range(24)]

        # Market çalışma saatleri: 08:00 - 21:59 (TR zincir market saatleri)
        store_hours = [raw_hours[h] if 8 <= h <= 21 else 0.0 for h in range(24)]
        total_h = sum(store_hours)
        if total_h <= 0:
            store_hours = [1.0 / 14.0 if 8 <= h <= 21 else 0.0 for h in range(24)]
        else:
            store_hours = [val / total_h for val in store_hours]

        return {
            "daily_sale_mean": mean_sale,
            "daily_sale_std": std_sale,
            "daily_sale_cv": cv,
            "zero_sale_prob": zero_prob,
            "weekday_multiplier": round(weekday_mult, 4),
            "weekend_multiplier": round(weekend_mult, 4),
            "weather_shock_prob": round(shock_prob, 4),
            "weather_shock_multiplier": round(weather_shock_mult, 4),
            "holiday_multiplier": round(holiday_mult, 4),
            "hourly_sale_probs": store_hours,
        }

    def _extract_markdown_elasticity(self) -> tuple[float, dict[float, float]]:
        """SYNTHETIC Markdown Dataset.csv'den indirim esnekliğini Polars LazyFrame ile hesaplar."""
        lf = pl.scan_csv(self.markdown_csv_path, ignore_errors=True).filter(
            (pl.col("Historical_Sales") > 0) & (pl.col("Markdown_1") > 0)
        )

        # İlk ve ikinci markdown kademelerindeki gözlenen satış artışı esnekliği
        elast_df = lf.select(
            ((pl.col("Sales_After_M1") / pl.col("Historical_Sales") - 1.0) / pl.col("Markdown_1"))
            .median()
            .alias("e1"),
            ((pl.col("Sales_After_M2") / pl.col("Sales_After_M1").clip(lower_bound=1) - 1.0) / pl.col("Markdown_2"))
            .median()
            .alias("e2"),
            ((pl.col("Sales_After_M4") / pl.col("Historical_Sales") - 1.0)
             / (pl.col("Markdown_1") + pl.col("Markdown_2") + pl.col("Markdown_3") + pl.col("Markdown_4"))
             .clip(lower_bound=0.05))
            .median()
            .alias("e_cum"),
        ).collect()

        e1 = float(elast_df["e1"][0] or 0.65)
        e2 = float(elast_df["e2"][0] or 0.65)
        e_cum = float(elast_df["e_cum"][0] or 0.85)

        # Taze gıda son tüketim indirimleri için veri temelli birleşik esneklik katsayısı
        base_elasticity = round(max(0.50, min(1.80, 0.5 * (e1 + e2) + 0.35 * e_cum)), 4)

        tier_lifts = {
            0.00: 1.00,
            0.15: round(1.0 + base_elasticity * 0.15, 4),
            0.30: round(1.0 + base_elasticity * 0.30, 4),
            0.50: round(1.0 + base_elasticity * 0.50, 4),
        }
        return base_elasticity, tier_lifts

    def _extract_reorder_profiles(self) -> dict[str, CategoryReorderProfile]:
        """Grocery_Inventory_and_Sales_Dataset.csv'den kategori bazlı reorder oranları çıkarır."""
        lf = (
            pl.scan_csv(self.grocery_csv_path, ignore_errors=True)
            .with_columns(pl.col("Catagory").str.strip_chars().alias("cat"))
            .filter(pl.col("cat").is_not_null() & (pl.col("Sales_Volume") > 0))
            .group_by("cat")
            .agg(
                (pl.col("Reorder_Level") / pl.col("Sales_Volume")).median().alias("lvl_ratio"),
                (pl.col("Reorder_Quantity") / pl.col("Sales_Volume")).median().alias("qty_ratio"),
                pl.col("Inventory_Turnover_Rate").median().alias("turnover"),
            )
        )
        df = lf.collect()
        raw_map: dict[str, CategoryReorderProfile] = {}
        for row in df.iter_rows(named=True):
            raw_map[str(row["cat"])] = CategoryReorderProfile(
                reorder_level_ratio=round(float(row["lvl_ratio"] or 0.85), 4),
                reorder_qty_ratio=round(float(row["qty_ratio"] or 0.85), 4),
                turnover_rate=round(float(row["turnover"] or 15.0), 2),
            )

        fallback = CategoryReorderProfile(reorder_level_ratio=0.85, reorder_qty_ratio=0.85, turnover_rate=15.0)
        return {
            eq_cat: raw_map.get(g_cat, fallback)
            for eq_cat, g_cat in self.EQUORA_TO_GROCERY_MAP.items()
        }

    def extract(self) -> StatisticalDemandProfile:
        """Tüm dış veri setlerinden istatistiksel profilleri tek seferde çıkarır."""
        fresh = self._extract_fresh_retail()
        elasticity, tier_lifts = self._extract_markdown_elasticity()
        reorder_profiles = self._extract_reorder_profiles()

        return StatisticalDemandProfile(
            daily_sale_mean=float(fresh["daily_sale_mean"]),
            daily_sale_std=float(fresh["daily_sale_std"]),
            daily_sale_cv=float(fresh["daily_sale_cv"]),
            zero_sale_prob=float(fresh["zero_sale_prob"]),
            weekday_multiplier=float(fresh["weekday_multiplier"]),
            weekend_multiplier=float(fresh["weekend_multiplier"]),
            weather_shock_prob=float(fresh["weather_shock_prob"]),
            weather_shock_multiplier=float(fresh["weather_shock_multiplier"]),
            holiday_multiplier=float(fresh["holiday_multiplier"]),
            hourly_sale_probs=list(fresh["hourly_sale_probs"]),  # type: ignore[arg-type]
            markdown_elasticity=elasticity,
            markdown_tier_lifts=tier_lifts,
            reorder_profiles=reorder_profiles,
        )


@dataclass(frozen=True)
class CategorySimCalibration:
    """Kategori bazlı günlük baz talep ve fire kalibrasyon parametreleri."""

    base_daily_demand: float      # Ortalama şubede ürün başı günlük baz satış adedi
    reorder_point_shelf_frac: float  # Kritik stok eşiği (shelf_life_days oranında)
    order_cover_shelf_frac: float    # Sipariş parti büyüklüğü (shelf_life_days oranında)
    case_pack: int                   # Koli/parti içi minimum paket katı
    demand_dispersion: float         # Gamma-Poisson talep oynaklığı (CV ölçeği)


class FefoInventorySimulator:
    """3 şube x 180 gün boyunca gerçekçi FEFO stok hareket simülasyonu üretir."""

    STORE_MULTIPLIERS: dict[int, float] = {
        1: 1.25,  # Merkez / yüksek trafikli şube
        2: 1.00,  # Standart mahalle şubesi
        3: 0.78,  # Daha küçük / oynak talep şubesi
    }

    # Kategori bazlı gerçekçi TR market talep ve fire (%6 - %18) kalibrasyonu
    # (selling_window = shelf_life_days + 1 baz alınarak kalibre edilmiştir)
    CATEGORY_CALIBRATION: dict[str, CategorySimCalibration] = {
        "Fırın & Unlu Mamul": CategorySimCalibration(
            base_daily_demand=14.0,
            reorder_point_shelf_frac=0.36,
            order_cover_shelf_frac=1.04,
            case_pack=4,
            demand_dispersion=0.56,
        ),
        "Et & Tavuk": CategorySimCalibration(
            base_daily_demand=7.5,
            reorder_point_shelf_frac=0.37,
            order_cover_shelf_frac=1.00,
            case_pack=3,
            demand_dispersion=0.56,
        ),
        "Meyve & Sebze": CategorySimCalibration(
            base_daily_demand=11.0,
            reorder_point_shelf_frac=0.36,
            order_cover_shelf_frac=0.98,
            case_pack=4,
            demand_dispersion=0.54,
        ),
        "Hazır Gıda": CategorySimCalibration(
            base_daily_demand=6.5,
            reorder_point_shelf_frac=0.35,
            order_cover_shelf_frac=0.95,
            case_pack=3,
            demand_dispersion=0.55,
        ),
        "Süt & Süt Ürünleri": CategorySimCalibration(
            base_daily_demand=9.0,
            reorder_point_shelf_frac=0.34,
            order_cover_shelf_frac=0.91,
            case_pack=6,
            demand_dispersion=0.50,
        ),
        "Şarküteri": CategorySimCalibration(
            base_daily_demand=5.5,
            reorder_point_shelf_frac=0.32,
            order_cover_shelf_frac=0.88,
            case_pack=4,
            demand_dispersion=0.50,
        ),
    }

    def __init__(
        self,
        catalog_df: pl.DataFrame,
        profile: StatisticalDemandProfile,
        store_ids: tuple[int, ...] = (1, 2, 3),
        num_days: int = 180,
        end_date: date | None = None,
        seed: int = 42,
    ) -> None:
        self.catalog_df = catalog_df
        self.profile = profile
        self.store_ids = store_ids
        self.num_days = num_days
        self.end_date = end_date or date.today()
        self.start_date = self.end_date - timedelta(days=num_days - 1)
        self.rng = np.random.default_rng(seed)
        self._hours = np.arange(24)
        self._hour_probs = np.array(self.profile.hourly_sale_probs, dtype=np.float64)
        self._hour_probs /= self._hour_probs.sum()

    def _choose_markdown(self, days_left: int) -> float:
        """SKT'ye 0-2 gün kalan partilerde kademeli ve kısmi indirim politikası uygular."""
        roll = float(self.rng.random())
        if days_left <= 0:
            if roll < 0.65:
                return 0.50
            if roll < 0.85:
                return 0.30
            return 0.00
        if days_left == 1:
            if roll < 0.55:
                return 0.30
            if roll < 0.80:
                return 0.15
            return 0.00
        if days_left == 2:
            if roll < 0.42:
                return 0.15
            return 0.00
        return 0.00

    def _sample_daily_demand(self, expected_mean: float, dispersion: float) -> int:
        """Overdispersed Gamma-Poisson (Negative Binomial) ile gerçekçi günlük adet üretir."""
        if expected_mean <= 0.05:
            return 0
        # fresh_retail zero-sale frekansına paralel düşük talep günleri
        if self.rng.random() < self.profile.zero_sale_prob * 0.5:
            return 0
        shape = 1.0 / max(dispersion * dispersion, 0.05)
        scale = expected_mean / shape
        lam = float(self.rng.gamma(shape=shape, scale=scale))
        return int(self.rng.poisson(max(lam, 0.0)))

    def _split_intraday_sales(
        self,
        sim_day: date,
        total_qty: int,
    ) -> list[tuple[datetime, int]]:
        """Bir partiden gün içinde yapılan satışı saatlik yoğunluk profiline göre böler."""
        if total_qty <= 0:
            return []

        if total_qty <= 2:
            n_tx = 1
        elif total_qty <= 6:
            n_tx = 2
        else:
            n_tx = 3

        # Adetleri pozitif tam parçalara böl (her hareket quantity > 0 garantili)
        if n_tx == 1:
            parts = [total_qty]
        else:
            probs = self.rng.dirichlet(np.ones(n_tx))
            raw = np.floor(probs * (total_qty - n_tx)).astype(int) + 1
            diff = total_qty - int(raw.sum())
            raw[0] += diff
            parts = [int(x) for x in raw if x > 0]

        sampled_hours = sorted(
            int(h) for h in self.rng.choice(self._hours, size=len(parts), p=self._hour_probs)
        )
        events: list[tuple[datetime, int]] = []
        used_seconds: set[int] = set()
        for hour, qty in zip(sampled_hours, parts):
            minute = int(self.rng.integers(0, 60))
            second = int(self.rng.integers(0, 60))
            key = hour * 3600 + minute * 60 + second
            while key in used_seconds:
                second = (second + 1) % 60
                key = hour * 3600 + minute * 60 + second
            used_seconds.add(key)
            ts = datetime.combine(sim_day, time(hour, minute, second), tzinfo=TR_TZ)
            events.append((ts, qty))
        return events

    def simulate(self) -> pl.DataFrame:
        """3 şube x 180 gün FEFO simülasyonunu çalıştırır ve Polars DataFrame döndürür.

        Kolonlar (PostgreSQL stock_movements şemasıyla birebir uyumlu):
            store_id, product_id, expiry_date, movement_type,
            quantity, movement_date, discount_rate
        """
        # 180 günlük ortak şube-hava-takvim takvimi üret
        sim_days = [self.start_date + timedelta(days=i) for i in range(self.num_days)]

        # Şube x Gün hava şoku ve talep çarpanı matrisi
        day_weather_shock: dict[tuple[int, int], float] = {}
        for store_id in self.store_ids:
            for d_idx, sim_day in enumerate(sim_days):
                is_weekend = sim_day.weekday() >= 5
                dow_mult = (
                    self.profile.weekend_multiplier
                    if is_weekend
                    else self.profile.weekday_multiplier
                )
                # Kötü hava / ani talep düşüşü şoku (fresh_retail frekansı)
                if self.rng.random() < self.profile.weather_shock_prob:
                    w_mult = self.profile.weather_shock_multiplier
                else:
                    w_mult = 1.0
                # Hafif mevsimsel / haftalık dalgalanma (sinüzoidal +-10%)
                seasonal = 1.0 + 0.10 * float(np.sin(2.0 * np.pi * d_idx / 45.0 + store_id))
                day_weather_shock[(store_id, d_idx)] = dow_mult * w_mult * seasonal

        # Kolon bazlı listeler (Polars DataFrame oluşturmada çok hızlı)
        out_store_id: list[int] = []
        out_product_id: list[int] = []
        out_expiry_date: list[date] = []
        out_movement_type: list[str] = []
        out_quantity: list[int] = []
        out_movement_date: list[datetime] = []
        out_discount_rate: list[float] = []

        def record(
            store_id: int,
            product_id: int,
            expiry_date: date,
            movement_type: str,
            quantity: int,
            movement_date: datetime,
            discount_rate: float,
        ) -> None:
            if quantity <= 0:
                return
            out_store_id.append(store_id)
            out_product_id.append(product_id)
            out_expiry_date.append(expiry_date)
            out_movement_type.append(movement_type)
            out_quantity.append(int(quantity))
            out_movement_date.append(movement_date)
            out_discount_rate.append(round(float(discount_rate), 4))

        products = self.catalog_df.to_dicts()

        for store_id in self.store_ids:
            store_mult = self.STORE_MULTIPLIERS.get(store_id, 1.0)

            for prod in products:
                product_id = int(prod["product_id"])
                category = str(prod["category"])
                shelf_life = int(prod["shelf_life_days"])
                pop_weight = float(prod.get("popularity_weight", 1.0))

                calib = self.CATEGORY_CALIBRATION[category]
                reorder_prof = self.profile.reorder_profiles.get(category)

                # grocery_inventory reorder oranlarından hafif kategori modülasyonu
                lvl_mod = 1.0
                qty_mod = 1.0
                if reorder_prof is not None:
                    lvl_mod = max(0.88, min(1.12, reorder_prof.reorder_level_ratio / 0.85))
                    qty_mod = max(0.90, min(1.12, reorder_prof.reorder_qty_ratio / 0.85))

                # Ürün+Şube baz günlük beklenen satış hızı ve etkin satış penceresi (IN günü dahil)
                base_mu = calib.base_daily_demand * store_mult * pop_weight
                selling_window = shelf_life + 1

                # Kritik stok seviyesi ve koli/parti sipariş adedi
                reorder_point = max(
                    calib.case_pack,
                    int(round(base_mu * selling_window * calib.reorder_point_shelf_frac * lvl_mod)),
                )
                raw_order_qty = max(
                    calib.case_pack,
                    int(round(base_mu * selling_window * calib.order_cover_shelf_frac * qty_mod)),
                )
                # Koli/paket katına yuvarla
                order_qty = max(
                    calib.case_pack,
                    int(round(raw_order_qty / calib.case_pack)) * calib.case_pack,
                )

                # Aktif partiler (her zaman expiry_date artan sırada — FEFO)
                batches: list[InventoryBatch] = []

                for d_idx, sim_day in enumerate(sim_days):
                    # ---------------------------------------------------------
                    # 1) 00:01 — Süresi dolmuş (expiry_date < sim_day) partileri WASTE kaydet
                    # ---------------------------------------------------------
                    still_fresh: list[InventoryBatch] = []
                    for batch in batches:
                        if batch.expiry_date < sim_day and batch.remaining_qty > 0:
                            waste_ts = datetime.combine(sim_day, time(0, 1, 0), tzinfo=TR_TZ)
                            record(
                                store_id=store_id,
                                product_id=product_id,
                                expiry_date=batch.expiry_date,
                                movement_type="WASTE",
                                quantity=batch.remaining_qty,
                                movement_date=waste_ts,
                                discount_rate=0.0,
                            )
                            batch.remaining_qty = 0
                        elif batch.remaining_qty > 0:
                            still_fresh.append(batch)
                    batches = still_fresh

                    # ---------------------------------------------------------
                    # 2) 06:30-08:15 — Kritik stok kontrolü ve koli/parti IN hareketi
                    # ---------------------------------------------------------
                    active_stock = sum(b.remaining_qty for b in batches)
                    if active_stock <= reorder_point:
                        # Günlük operasyonel sipariş adedinde küçük koli dalgalanması (0.9x - 1.15x)
                        jitter = float(self.rng.choice([0.9, 1.0, 1.0, 1.15]))
                        in_qty = max(
                            calib.case_pack,
                            int(round((order_qty * jitter) / calib.case_pack)) * calib.case_pack,
                        )
                        exp_date = sim_day + timedelta(days=shelf_life)
                        in_hour = int(self.rng.integers(6, 8))
                        in_min = int(self.rng.integers(15, 59))
                        in_ts = datetime.combine(sim_day, time(in_hour, in_min, 0), tzinfo=TR_TZ)

                        record(
                            store_id=store_id,
                            product_id=product_id,
                            expiry_date=exp_date,
                            movement_type="IN",
                            quantity=in_qty,
                            movement_date=in_ts,
                            discount_rate=0.0,
                        )

                        # Aynı expiry_date varsa birleştir, yoksa yeni parti ekle
                        merged = False
                        for b in batches:
                            if b.expiry_date == exp_date:
                                b.remaining_qty += in_qty
                                merged = True
                                break
                        if not merged:
                            batches.append(
                                InventoryBatch(
                                    expiry_date=exp_date,
                                    remaining_qty=in_qty,
                                    in_date=sim_day,
                                )
                            )
                        batches.sort(key=lambda b: b.expiry_date)

                    if not batches:
                        continue

                    # ---------------------------------------------------------
                    # 3) 08:00-22:00 — SKT'ye yakın partide indirim + FEFO OUT satışı
                    # ---------------------------------------------------------
                    earliest = batches[0]
                    days_left = (earliest.expiry_date - sim_day).days
                    applied_discount = self._choose_markdown(days_left)
                    markdown_lift = self.profile.markdown_tier_lifts.get(
                        applied_discount,
                        1.0 + self.profile.markdown_elasticity * applied_discount,
                    )

                    day_mult = day_weather_shock[(store_id, d_idx)]
                    expected_demand = base_mu * day_mult * markdown_lift
                    realized_demand = self._sample_daily_demand(
                        expected_mean=expected_demand,
                        dispersion=calib.demand_dispersion,
                    )

                    if realized_demand <= 0:
                        continue

                    # FEFO sırasıyla (en yakın SKT'li partiden) satış düş
                    remaining_demand = realized_demand
                    for batch in batches:
                        if remaining_demand <= 0:
                            break
                        if batch.remaining_qty <= 0:
                            continue

                        sold = min(batch.remaining_qty, remaining_demand)
                        batch.remaining_qty -= sold
                        remaining_demand -= sold

                        # Sadece SKT'ye <= 2 gün kalan partiye indirim etiketi yansıt
                        batch_days_left = (batch.expiry_date - sim_day).days
                        batch_discount = applied_discount if batch_days_left <= 2 else 0.0

                        for sale_ts, part_qty in self._split_intraday_sales(sim_day, sold):
                            record(
                                store_id=store_id,
                                product_id=product_id,
                                expiry_date=batch.expiry_date,
                                movement_type="OUT",
                                quantity=part_qty,
                                movement_date=sale_ts,
                                discount_rate=batch_discount,
                            )

                    # Tükenen partileri temizle
                    batches = [b for b in batches if b.remaining_qty > 0]

        movements_df = pl.DataFrame(
            {
                "store_id": pl.Series("store_id", out_store_id, dtype=pl.Int32),
                "product_id": pl.Series("product_id", out_product_id, dtype=pl.Int32),
                "expiry_date": pl.Series("expiry_date", out_expiry_date, dtype=pl.Date),
                "movement_type": pl.Series("movement_type", out_movement_type, dtype=pl.String),
                "quantity": pl.Series("quantity", out_quantity, dtype=pl.Int32),
                "movement_date": pl.Series(
                    "movement_date",
                    out_movement_date,
                    dtype=pl.Datetime(time_unit="us", time_zone="UTC"),
                ),
                "discount_rate": pl.Series("discount_rate", out_discount_rate, dtype=pl.Float64),
            }
        )
        return movements_df
