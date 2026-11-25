"""EQUORA Uçtan Uca Sentetik Stok & FEFO Veri Hattı (run_pipeline.py).

Çalıştırma:
    python -m src.data_pipeline.run_pipeline
    python src/data_pipeline/run_pipeline.py --days 180 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Doğrudan `python src/data_pipeline/run_pipeline.py` çağrısı için kök dizini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.catalog_builder import TurkishCatalogBuilder
from src.data_pipeline.db_loader import PostgresLoader
from src.data_pipeline.fefo_simulator import DemandProfileExtractor, FefoInventorySimulator
from src.data_pipeline.inspector import DataInspector


class EquoraDataPipeline:
    """5 adımlı EQUORA veri keşif, katalog, FEFO simülasyon ve yükleme orkestratörü."""

    def __init__(
        self,
        num_days: int = 180,
        store_ids: tuple[int, ...] = (1, 2, 3),
        seed: int = 42,
        truncate: bool = True,
        run_inspection: bool = False,
    ) -> None:
        self.num_days = num_days
        self.store_ids = store_ids
        self.seed = seed
        self.truncate = truncate
        self.run_inspection = run_inspection

    def run(self) -> dict[str, object]:
        t0 = time.perf_counter()

        if self.run_inspection:
            print("[ADIM 1/5] Ham veri setleri keşfediliyor...")
            inspector = DataInspector()
            inspector.print_report()
        else:
            print("[ADIM 1/5] Ham veri keşfi atlandı (--inspect ile açılabilir).")

        print("\n[ADIM 2/5] Türkiye bozulabilir ürün kataloğu oluşturuluyor...")
        catalog_builder = TurkishCatalogBuilder()
        catalog_df = catalog_builder.build_catalog()
        print(f"  [OK] {catalog_df.height} özgün ürün 6 kategoride oluşturuldu.")

        print("\n[ADIM 3/5] İstatistiksel profiller çıkarılıyor ve FEFO simülasyonu çalıştırılıyor...")
        extractor = DemandProfileExtractor()
        profile = extractor.extract()
        print(
            f"  [OK] Profil çıkarıldı | Hafta içi: {profile.weekday_multiplier:.2f}x, "
            f"Hafta sonu: {profile.weekend_multiplier:.2f}x, "
            f"Hava şoku olasılığı: %{profile.weather_shock_prob * 100:.1f} "
            f"({profile.weather_shock_multiplier:.2f}x), "
            f"İndirim esnekliği: {profile.markdown_elasticity:.3f}"
        )

        simulator = FefoInventorySimulator(
            catalog_df=catalog_df,
            profile=profile,
            store_ids=self.store_ids,
            num_days=self.num_days,
            seed=self.seed,
        )
        movements_df = simulator.simulate()
        print(
            f"  [OK] {len(self.store_ids)} şube x {self.num_days} gün için "
            f"{movements_df.height:,} stok hareketi simüle edildi."
        )

        print("\n[ADIM 4/5] Parquet dosyaları kaydediliyor ve PostgreSQL (equora_db) yükleniyor...")
        loader = PostgresLoader()
        prod_pq, mov_pq = loader.save_parquet(catalog_df, movements_df)
        print(f"  [OK] Parquet kaydedildi: {prod_pq}")
        print(f"  [OK] Parquet kaydedildi: {mov_pq}")

        load_stats = loader.load_to_postgres(
            products_df=catalog_df,
            movements_df=movements_df,
            truncate=self.truncate,
        )
        print(
            f"  [OK] PostgreSQL yükleme tamamlandı: "
            f"{load_stats['products_loaded']:,} ürün, "
            f"{load_stats['movements_loaded']:,} stok hareketi."
        )

        print("\n[ADIM 5/5] PostgreSQL doğrulama raporu çalıştırılıyor...")
        verification = loader.run_verification_report()

        elapsed = time.perf_counter() - t0
        print(f"\n[TAMAMLANDI] Tüm veri hattı {elapsed:.2f} saniyede başarıyla tamamlandı.")
        return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EQUORA FEFO stok simülasyonu ve PostgreSQL yükleme hattı"
    )
    parser.add_argument("--days", type=int, default=180, help="Simüle edilecek gün sayısı")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Yüklemeden önce mevcut PostgreSQL tablolarını temizleme",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Adım 1 DataInspector raporunu da çalışma başında yazdır",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = EquoraDataPipeline(
        num_days=args.days,
        seed=args.seed,
        truncate=not args.no_truncate,
        run_inspection=args.inspect,
    )
    verification = pipeline.run()
    return 0 if verification.get("negative_remaining_count", 1) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
