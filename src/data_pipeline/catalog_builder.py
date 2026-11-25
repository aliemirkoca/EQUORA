"""Adım 2 — Türkiye Ürün Kataloğu Üretici (TurkishCatalogBuilder).

raw_data/turkish_market/MarketSales.xlsx verisinden bozulabilir/kısa-orta
raf ömürlü 6 gıda kategorisinden dengeli şekilde 60-100 arası özgün ürün seçer;
raw_data/grocery_inventory/Grocery_Inventory_and_Sales_Dataset.csv raf ömrü
dağılımlarından ve Türkiye market standartlarından esinlenerek shelf_life_days,
EAN-13 (869...) barkod ve güncel TR piyasa birim fiyatı üretir.

Not: Yalnızca Polars (LazyFrame / scan_csv) kullanır, Pandas kullanmaz.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MARKET_XLSX = PROJECT_ROOT / "raw_data" / "turkish_market" / "MarketSales.xlsx"
DEFAULT_GROCERY_CSV = (
    PROJECT_ROOT / "raw_data" / "grocery_inventory" / "Grocery_Inventory_and_Sales_Dataset.csv"
)


@dataclass(frozen=True)
class CategoryRule:
    """Her bozulabilir gıda kategorisi için seçim, raf ömrü ve fiyat ölçekleme kuralları."""

    equora_category: str
    grocery_ref_category: str
    target_count: int
    shelf_life_min: int
    shelf_life_max: int
    price_scale_2017_to_tr: float
    min_tr_price: float
    max_tr_price: float


CATEGORY_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        equora_category="Süt & Süt Ürünleri",
        grocery_ref_category="Dairy",
        target_count=14,
        shelf_life_min=7,
        shelf_life_max=14,
        price_scale_2017_to_tr=22.0,
        min_tr_price=24.90,
        max_tr_price=369.90,
    ),
    CategoryRule(
        equora_category="Fırın & Unlu Mamul",
        grocery_ref_category="Bakery",
        target_count=14,
        shelf_life_min=2,
        shelf_life_max=3,
        price_scale_2017_to_tr=18.0,
        min_tr_price=12.50,
        max_tr_price=145.00,
    ),
    CategoryRule(
        equora_category="Et & Tavuk",
        grocery_ref_category="Seafood",
        target_count=14,
        shelf_life_min=3,
        shelf_life_max=5,
        price_scale_2017_to_tr=25.0,
        min_tr_price=119.90,
        max_tr_price=799.90,
    ),
    CategoryRule(
        equora_category="Şarküteri",
        grocery_ref_category="Dairy",
        target_count=14,
        shelf_life_min=10,
        shelf_life_max=21,
        price_scale_2017_to_tr=23.0,
        min_tr_price=69.90,
        max_tr_price=489.90,
    ),
    CategoryRule(
        equora_category="Meyve & Sebze",
        grocery_ref_category="Fruits & Vegetables",
        target_count=14,
        shelf_life_min=4,
        shelf_life_max=6,
        price_scale_2017_to_tr=24.0,
        min_tr_price=24.90,
        max_tr_price=189.90,
    ),
    CategoryRule(
        equora_category="Hazır Gıda",
        grocery_ref_category="Bakery",
        target_count=14,
        shelf_life_min=4,
        shelf_life_max=8,
        price_scale_2017_to_tr=24.0,
        min_tr_price=44.90,
        max_tr_price=249.90,
    ),
)


class TurkishCatalogBuilder:
    """Türkiye market şartlarına uyarlanmış bozulabilir ürün kataloğu üreticisi."""

    def __init__(
        self,
        market_xlsx_path: Path | str = DEFAULT_MARKET_XLSX,
        grocery_csv_path: Path | str = DEFAULT_GROCERY_CSV,
        rules: tuple[CategoryRule, ...] = CATEGORY_RULES,
    ) -> None:
        self.market_xlsx_path = Path(market_xlsx_path)
        self.grocery_csv_path = Path(grocery_csv_path)
        self.rules = rules

    @staticmethod
    def compute_ean13_check_digit(first_12_digits: str) -> str:
        """Standart GS1 EAN-13 kontrol hanesini (13. hane) hesaplar."""
        if len(first_12_digits) != 12 or not first_12_digits.isdigit():
            raise ValueError(f"EAN-13 için tam 12 rakam gerekli: {first_12_digits!r}")
        total = sum(
            int(ch) * (1 if idx % 2 == 0 else 3)
            for idx, ch in enumerate(first_12_digits)
        )
        check_digit = (10 - (total % 10)) % 10
        return str(check_digit)

    @classmethod
    def build_turkish_ean13(
        cls,
        raw_itemcode: str | None,
        product_name: str,
        used_barcodes: set[str],
    ) -> str:
        """869 ile başlayan geçerli ve benzersiz 13 haneli EAN-13 barkod üretir."""
        digits = "".join(ch for ch in (raw_itemcode or "") if ch.isdigit())
        if len(digits) == 13 and digits.startswith("869"):
            candidate = digits[:12] + cls.compute_ean13_check_digit(digits[:12])
            if candidate not in used_barcodes:
                used_barcodes.add(candidate)
                return candidate

        base_num = int(digits[-9:]) if digits else int(
            hashlib.sha256(product_name.encode("utf-8")).hexdigest()[:8], 16
        ) % 1_000_000_000

        offset = 0
        while True:
            body9 = f"{(base_num + offset) % 1_000_000_000:09d}"
            first12 = f"869{body9}"
            barcode = first12 + cls.compute_ean13_check_digit(first12)
            if barcode not in used_barcodes:
                used_barcodes.add(barcode)
                return barcode
            offset += 1

    @staticmethod
    def _clean_product_name(name: str) -> str:
        """Ürün adını temizler ve okunaklı hale getirir."""
        cleaned = " ".join(name.strip().split())
        return cleaned[:150]

    @staticmethod
    def _round_market_price(raw_price: float, min_price: float, max_price: float) -> float:
        """Fiyatı TR market psikolojik fiyatlandırmasına (.50 / .90 / .95) yuvarlar."""
        clamped = max(min_price, min(max_price, raw_price))
        whole = int(clamped)
        frac = clamped - whole
        if frac < 0.35:
            return round(max(min_price, whole - 0.10 if whole > 1 else clamped), 2)
        if frac < 0.70:
            return round(whole + 0.50, 2)
        return round(whole + 0.90, 2)

    def _extract_grocery_shelf_quantiles(self) -> dict[str, list[float]]:
        """grocery_inventory verisinden kategori bazlı normalize raf ömrü dağılımını çıkarır.

        Gerçek kolon adları: Catagory, Date_Received (M/D/YYYY), Expiration_Date (M/D/YYYY).
        """
        lf = (
            pl.scan_csv(self.grocery_csv_path, ignore_errors=True)
            .with_columns(
                pl.col("Catagory").str.strip_chars().alias("cat"),
                pl.col("Date_Received").str.to_date("%m/%d/%Y", strict=False).alias("d_rec"),
                pl.col("Expiration_Date").str.to_date("%m/%d/%Y", strict=False).alias("d_exp"),
            )
            .with_columns((pl.col("d_exp") - pl.col("d_rec")).dt.total_days().alias("raw_shelf"))
            .filter(pl.col("cat").is_not_null() & (pl.col("raw_shelf") > 0))
        )

        df = lf.select(["cat", "raw_shelf"]).collect()
        quantiles_by_cat: dict[str, list[float]] = {}

        for cat_name in df["cat"].unique().to_list():
            series = df.filter(pl.col("cat") == cat_name)["raw_shelf"].sort()
            p10 = float(series.quantile(0.10) or 1.0)
            p90 = float(series.quantile(0.90) or (p10 + 1.0))
            span = max(p90 - p10, 1.0)
            norm_vals = [
                max(0.0, min(1.0, (float(v) - p10) / span))
                for v in series.to_list()
            ]
            quantiles_by_cat[cat_name] = norm_vals

        return quantiles_by_cat

    def _load_market_lazy(self) -> pl.LazyFrame:
        """MarketSales.xlsx dosyasını Polars calamine motoru ile okuyup LazyFrame'e dönüştürür."""
        if not self.market_xlsx_path.exists():
            raise FileNotFoundError(f"MarketSales dosyası bulunamadı: {self.market_xlsx_path}")

        df = pl.read_excel(
            self.market_xlsx_path,
            sheet_name="Sheet1",
            engine="calamine",
            columns=[
                "ITEMCODE",
                "ITEMNAME",
                "PRICE",
                "AMOUNT",
                "CATEGORY_NAME1",
                "CATEGORY_NAME2",
                "CATEGORY_NAME3",
            ],
        )
        return df.lazy().with_columns(
            pl.col("ITEMCODE").cast(pl.String).str.strip_chars(),
            pl.col("ITEMNAME").cast(pl.String).str.strip_chars(),
            pl.col("CATEGORY_NAME1").cast(pl.String).str.strip_chars(),
            pl.col("CATEGORY_NAME2").cast(pl.String).str.strip_chars(),
            pl.col("CATEGORY_NAME3").cast(pl.String).str.strip_chars(),
        )

    @staticmethod
    def _map_perishable_category_expr() -> pl.Expr:
        """Gerçek MarketSales kategori kolonlarını 6 bozulabilir EQUORA kategorisine eşler."""
        c1 = pl.col("CATEGORY_NAME1")
        c2 = pl.col("CATEGORY_NAME2")
        c3 = pl.col("CATEGORY_NAME3").fill_null("")
        name_upper = pl.col("ITEMNAME").str.to_uppercase()

        # Fırın & Unlu Mamul: yalnızca taze/kısa raf ömürlü fırın ve yufka ürünleri
        is_bakery = (c2 == "UNLU MAMÜLLER") & (
            (c3 == "EKMEK")
            | name_upper.str.contains("YUFKA|EKMEK|PIDE|LAVAS|SIMIT|POGACA|BOREK|TORTILLA|BAZLAM|PASTA")
        ) & (~name_upper.str.contains("PUDING|NISASTA|PUDRA|KABARTMA|VANIL|MAYA|KREM SANTI|UNU 1|UNU 2|UNU 5"))

        # Süt & Süt Ürünleri
        is_dairy = (
            (c1 == "SÜT KAHVALTILIK")
            & (
                c2.is_in(["SÜT", "YOĞURT", "PEYNİR"])
                | c3.is_in(["KAYMAK"])
            )
        ) | ((c2 == "GÜNLÜK İÇECEK") & (c3 == "AYRAN"))

        # Et & Tavuk
        is_meat_poultry = (c1 == "ET TAVUK") & c2.is_in(["KIRMIZI ET", "KÜMES HAYVANLARI"])

        # Şarküteri
        is_deli = (c1 == "ET TAVUK") & (c2 == "ET ŞARKÜTERİ")

        # Meyve & Sebze
        is_produce = (c1 == "MEYVE SEBZE") & c2.is_in(["MEYVE", "SEBZE"])

        # Hazır Gıda: meze, çiğ köfte, sütlü tatlı, soğuk hazır yemek
        is_ready_food = (
            ((c2 == "HAZIR YEMEKLER") & (c3.is_in(["MEZE", "HAZIR YEMEK"])))
            | ((c2 == "SÜTLÜ TATLILAR DONDURMA") & (c3 == "SÜTLÜ TATLILAR"))
        ) & (~name_upper.str.contains("SALCA|TURSU|KETCAP|SIRKE|SOSU|EZME 190"))

        return (
            pl.when(is_bakery).then(pl.lit("Fırın & Unlu Mamul"))
            .when(is_dairy).then(pl.lit("Süt & Süt Ürünleri"))
            .when(is_meat_poultry).then(pl.lit("Et & Tavuk"))
            .when(is_deli).then(pl.lit("Şarküteri"))
            .when(is_produce).then(pl.lit("Meyve & Sebze"))
            .when(is_ready_food).then(pl.lit("Hazır Gıda"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("equora_category")
        )

    def _assign_shelf_life(
        self,
        rule: CategoryRule,
        item_name: str,
        sub_cat: str,
        rank_idx: int,
        ref_quantiles: list[float],
    ) -> int:
        """grocery_inventory dağılım pozisyonu + TR gıda kurallarıyla shelf_life_days atar."""
        upper_name = item_name.upper()
        upper_sub = (sub_cat or "").upper()

        # Kategori içi alt-ürün raf ömrü hassasiyeti (TR market gerçekliği)
        if rule.equora_category == "Süt & Süt Ürünleri":
            if "AYRAN" in upper_name or "KAYMAK" in upper_name or "PASTÖRİZE" in upper_sub:
                low, high = 7, 10
            elif "YOĞURT" in upper_name or "YOGURT" in upper_name:
                low, high = 9, 12
            else:
                low, high = 10, 14
        elif rule.equora_category == "Fırın & Unlu Mamul":
            if "YUFKA" in upper_name or "LAVAS" in upper_name:
                low, high = 3, 3
            else:
                low, high = 2, 3
        elif rule.equora_category == "Et & Tavuk":
            if "KIYMA" in upper_name or "CIGER" in upper_name or "PİLİÇ" in upper_sub or "PILIC" in upper_name:
                low, high = 3, 4
            else:
                low, high = 4, 5
        elif rule.equora_category == "Meyve & Sebze":
            if any(k in upper_name for k in ("KIVIRCIK", "MAYDANOZ", "ROKA", "MARUL", "DEREOTU", "MUZ", "CILEK")):
                low, high = 4, 4
            elif any(k in upper_name for k in ("PATATES", "SOGAN", "HAVUC", "ELMA", "PORTAKAL", "LIMON")):
                low, high = 5, 6
            else:
                low, high = 4, 5
        else:
            low, high = rule.shelf_life_min, rule.shelf_life_max

        if ref_quantiles:
            q_idx = int((rank_idx / max(rule.target_count, 1)) * len(ref_quantiles)) % len(ref_quantiles)
            norm = ref_quantiles[q_idx]
        else:
            norm = 0.5

        days = int(round(low + norm * (high - low)))
        return max(rule.shelf_life_min, min(rule.shelf_life_max, days))

    def build_catalog(self) -> pl.DataFrame:
        """60-100 arası özgün, temiz ve PostgreSQL products şemasına uyumlu katalog döndürür.

        Kolonlar:
            product_id, barcode, product_name, category, unit_price, shelf_life_days,
            popularity_weight (simülatör için dahili satış ağırlığı — DB'ye yazılırken ayrılır)
        """
        shelf_quantiles = self._extract_grocery_shelf_quantiles()
        market_lf = self._load_market_lazy()

        aggregated_lf = (
            market_lf.with_columns(self._map_perishable_category_expr())
            .filter(
                pl.col("equora_category").is_not_null()
                & pl.col("ITEMNAME").is_not_null()
                & (pl.col("ITEMNAME").str.len_chars() >= 3)
                & (pl.col("PRICE") > 0)
            )
            .group_by(["equora_category", "ITEMNAME"])
            .agg(
                pl.col("ITEMCODE").drop_nulls().first().alias("raw_itemcode"),
                pl.col("CATEGORY_NAME3").drop_nulls().first().alias("sub_cat"),
                pl.col("PRICE").median().alias("median_2017_price"),
                pl.len().alias("tx_count"),
            )
            .sort(["equora_category", "tx_count"], descending=[False, True])
        )

        candidates_df = aggregated_lf.collect()

        rows: list[dict[str, object]] = []
        used_barcodes: set[str] = set()
        used_names: set[str] = set()
        product_id = 1

        for rule in self.rules:
            cat_df = candidates_df.filter(pl.col("equora_category") == rule.equora_category)
            ref_q = shelf_quantiles.get(rule.grocery_ref_category, [])

            selected_for_cat = 0
            for record in cat_df.iter_rows(named=True):
                if selected_for_cat >= rule.target_count:
                    break

                clean_name = self._clean_product_name(str(record["ITEMNAME"]))
                norm_key = clean_name.upper()
                if norm_key in used_names:
                    continue

                used_names.add(norm_key)
                barcode = self.build_turkish_ean13(
                    raw_itemcode=record["raw_itemcode"],
                    product_name=clean_name,
                    used_barcodes=used_barcodes,
                )

                scaled_price = float(record["median_2017_price"]) * rule.price_scale_2017_to_tr
                unit_price = self._round_market_price(
                    scaled_price, rule.min_tr_price, rule.max_tr_price
                )
                shelf_life_days = self._assign_shelf_life(
                    rule=rule,
                    item_name=clean_name,
                    sub_cat=str(record["sub_cat"] or ""),
                    rank_idx=selected_for_cat,
                    ref_quantiles=ref_q,
                )

                rows.append(
                    {
                        "product_id": product_id,
                        "barcode": barcode,
                        "product_name": clean_name,
                        "category": rule.equora_category,
                        "unit_price": unit_price,
                        "shelf_life_days": shelf_life_days,
                        "tx_count": int(record["tx_count"]),
                    }
                )
                product_id += 1
                selected_for_cat += 1

        catalog_df = pl.DataFrame(
            rows,
            schema={
                "product_id": pl.Int32,
                "barcode": pl.String,
                "product_name": pl.String,
                "category": pl.String,
                "unit_price": pl.Float64,
                "shelf_life_days": pl.Int32,
                "tx_count": pl.Int64,
            },
        )

        # Kategori içi görece satış popülerliği ağırlığı (0.65 - 1.55 arası)
        catalog_df = catalog_df.with_columns(
            (
                0.65
                + 0.90
                * (
                    (pl.col("tx_count").log1p() - pl.col("tx_count").log1p().min().over("category"))
                    / (
                        (pl.col("tx_count").log1p().max().over("category")
                         - pl.col("tx_count").log1p().min().over("category"))
                        .clip(lower_bound=1e-6)
                    )
                )
            )
            .round(4)
            .alias("popularity_weight")
        ).drop("tx_count")

        total = catalog_df.height
        if not (60 <= total <= 100):
            raise RuntimeError(f"Beklenen ürün sayısı 60-100 aralığında olmalı, üretilen: {total}")

        return catalog_df


if __name__ == "__main__":
    builder = TurkishCatalogBuilder()
    df = builder.build_catalog()
    print(f"Toplam üretilen ürün sayısı: {df.height}")
    print(df.group_by("category").agg([
        pl.len().alias("count"),
        pl.col("unit_price").median().round(2).alias("med_price"),
        pl.col("shelf_life_days").min().alias("min_shelf"),
        pl.col("shelf_life_days").max().alias("max_shelf"),
    ]).sort("category"))
    print(df.head(12))
