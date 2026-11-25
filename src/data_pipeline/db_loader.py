"""Adım 4 & 5 — Parquet Kayıt, PostgreSQL Toplu Yükleme ve Doğrulama (PostgresLoader).

Üretilen products ve stock_movements DataFrame'lerini önce data/processed/
altına Parquet olarak kaydeder; ardından .env bağlantı bilgileriyle yerel
PostgreSQL (equora_db) veritabanına hızlı toplu aktarım (execute_values + COPY)
ile yükler ve v_stock_summary doğrulama raporunu üretir.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class PostgresConnectionConfig:
    """Yerel PostgreSQL bağlantı ayarları."""

    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls, env_path: Path | str = DEFAULT_ENV_PATH) -> PostgresConnectionConfig:
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=True)
        else:
            load_dotenv(override=True)

        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "equora_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )

    def as_kwargs(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
        }


class PostgresLoader:
    """Parquet çıktısı üreten, PostgreSQL'e toplu yükleyen ve doğrulayan sınıf."""

    PRODUCT_SCHEMA_COLS = [
        "barcode",
        "product_name",
        "category",
        "unit_price",
        "shelf_life_days",
    ]
    MOVEMENT_SCHEMA_COLS = [
        "store_id",
        "product_id",
        "expiry_date",
        "movement_type",
        "quantity",
        "movement_date",
        "discount_rate",
    ]

    def __init__(
        self,
        env_path: Path | str = DEFAULT_ENV_PATH,
        processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        self.config = PostgresConnectionConfig.from_env(env_path)
        self.processed_dir = Path(processed_dir)

    def save_parquet(
        self,
        products_df: pl.DataFrame,
        movements_df: pl.DataFrame,
    ) -> tuple[Path, Path]:
        """Üretilen tabloları data/processed/*.parquet olarak kaydeder."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        products_path = self.processed_dir / "products.parquet"
        movements_path = self.processed_dir / "stock_movements.parquet"

        clean_products = products_df.select(["product_id", *self.PRODUCT_SCHEMA_COLS])
        clean_movements = movements_df.select(self.MOVEMENT_SCHEMA_COLS)

        clean_products.write_parquet(products_path, compression="zstd")
        clean_movements.write_parquet(movements_path, compression="zstd")
        return products_path, movements_path

    def load_to_postgres(
        self,
        products_df: pl.DataFrame,
        movements_df: pl.DataFrame,
        truncate: bool = True,
    ) -> dict[str, int]:
        """products (execute_values) ve stock_movements (COPY) tablolarını toplu yükler."""
        conn = psycopg2.connect(**self.config.as_kwargs())
        try:
            with conn:
                with conn.cursor() as cur:
                    if truncate:
                        cur.execute(
                            "TRUNCATE TABLE stock_movements, products RESTART IDENTITY CASCADE;"
                        )

                    # 1) products toplu yükleme ve gerçek DB product_id eşlemesi
                    prod_rows = [
                        (
                            str(r["barcode"]),
                            str(r["product_name"]),
                            str(r["category"]),
                            round(float(r["unit_price"]), 2),
                            int(r["shelf_life_days"]),
                        )
                        for r in products_df.select(self.PRODUCT_SCHEMA_COLS).iter_rows(named=True)
                    ]

                    inserted_pairs = execute_values(
                        cur,
                        """
                        INSERT INTO products
                            (barcode, product_name, category, unit_price, shelf_life_days)
                        VALUES %s
                        ON CONFLICT (barcode) DO UPDATE SET
                            product_name = EXCLUDED.product_name,
                            category = EXCLUDED.category,
                            unit_price = EXCLUDED.unit_price,
                            shelf_life_days = EXCLUDED.shelf_life_days
                        RETURNING product_id, barcode;
                        """,
                        prod_rows,
                        fetch=True,
                    )
                    barcode_to_db_id = {str(bc): int(pid) for pid, bc in inserted_pairs}

                    # Simülatörün yerel product_id -> PostgreSQL product_id haritası
                    id_map_df = pl.DataFrame(
                        {
                            "sim_product_id": products_df["product_id"].cast(pl.Int32),
                            "db_product_id": [
                                barcode_to_db_id[str(bc)]
                                for bc in products_df["barcode"].to_list()
                            ],
                        },
                        schema={"sim_product_id": pl.Int32, "db_product_id": pl.Int32},
                    )

                    mapped_movements = (
                        movements_df.join(
                            id_map_df,
                            left_on="product_id",
                            right_on="sim_product_id",
                            how="inner",
                        )
                        .with_columns(pl.col("db_product_id").alias("product_id"))
                        .select(self.MOVEMENT_SCHEMA_COLS)
                        .with_columns(
                            pl.col("discount_rate").round(4),
                        )
                    )

                    # 2) stock_movements için yüksek hızlı PostgreSQL COPY FROM STDIN
                    csv_buffer = io.StringIO()
                    mapped_movements.write_csv(
                        csv_buffer,
                        include_header=True,
                        datetime_format="%Y-%m-%dT%H:%M:%S%z",
                        date_format="%Y-%m-%d",
                        float_precision=4,
                    )
                    csv_buffer.seek(0)

                    cur.copy_expert(
                        """
                        COPY stock_movements (
                            store_id,
                            product_id,
                            expiry_date,
                            movement_type,
                            quantity,
                            movement_date,
                            discount_rate
                        )
                        FROM STDIN WITH (FORMAT CSV, HEADER TRUE);
                        """,
                        csv_buffer,
                    )

            return {
                "products_loaded": len(prod_rows),
                "movements_loaded": mapped_movements.height,
            }
        finally:
            conn.close()

    def run_verification_report(self) -> dict[str, object]:
        """PostgreSQL üzerinde Adım 5 doğrulama sorgularını çalıştırır ve yazdırır."""
        conn = psycopg2.connect(**self.config.as_kwargs())
        try:
            with conn.cursor() as cur:
                # 1) Toplam satır sayıları ve movement_type kırılımı
                cur.execute("SELECT COUNT(*) FROM products;")
                products_count = int(cur.fetchone()[0])

                cur.execute(
                    """
                    SELECT movement_type, COUNT(*), COALESCE(SUM(quantity), 0)
                    FROM stock_movements
                    GROUP BY movement_type
                    ORDER BY movement_type;
                    """
                )
                movement_breakdown = cur.fetchall()
                total_movements = sum(int(r[1]) for r in movement_breakdown)

                # 2) v_stock_summary'de remaining_qty < 0 kontrolü
                cur.execute(
                    "SELECT COUNT(*) FROM v_stock_summary WHERE remaining_qty < 0;"
                )
                negative_remaining_count = int(cur.fetchone()[0])

                # 3) Kategori bazlı giren, satılan, fire ve fire yüzdesi (%)
                cur.execute(
                    """
                    SELECT
                        category,
                        SUM(total_in) AS total_in,
                        SUM(total_out) AS total_out,
                        SUM(total_waste) AS total_waste,
                        SUM(remaining_qty) AS active_remaining,
                        ROUND(
                            100.0 * SUM(total_waste) / NULLIF(SUM(total_in), 0),
                            2
                        ) AS waste_pct
                    FROM v_stock_summary
                    GROUP BY category
                    ORDER BY waste_pct DESC;
                    """
                )
                category_stats = cur.fetchall()

                # 4) v_stock_summary örnek 10 satır (farklı kategorilerden dengeli kesit)
                cur.execute(
                    """
                    SELECT
                        store_id,
                        product_id,
                        barcode,
                        product_name,
                        category,
                        expiry_date,
                        total_in,
                        total_out,
                        total_waste,
                        remaining_qty
                    FROM v_stock_summary
                    ORDER BY expiry_date DESC, store_id, product_id
                    LIMIT 10;
                    """
                )
                sample_rows = cur.fetchall()
        finally:
            conn.close()

        print("\n" + "=" * 88)
        print("EQUORA POSTGRESQL DOĞRULAMA RAPORU (equora_db)")
        print("=" * 88)

        print("\n1) Tablo Satır Sayıları ve Hareket Kırılımı:")
        print(f"   - products toplam satır        : {products_count:,}")
        print(f"   - stock_movements toplam satır : {total_movements:,}")
        for m_type, cnt, qty_sum in movement_breakdown:
            print(f"       * {m_type:<6}: {int(cnt):>8,} hareket | toplam {int(qty_sum):>9,} adet")

        print("\n2) FEFO Tutarlılık Kontrolü (v_stock_summary -> remaining_qty < 0):")
        status_tag = "BAŞARILI (0 TUTARSIZLIK)" if negative_remaining_count == 0 else "HATA"
        print(f"   - remaining_qty < 0 satır sayısı: {negative_remaining_count} [{status_tag}]")

        print("\n3) Kategori Bazlı Stok & Fire (%) Özeti:")
        header = (
            f"   {'Kategori':<22} | {'Giren':>9} | {'Satılan':>9} | "
            f"{'Fire':>8} | {'Kalan':>7} | {'Fire %':>7}"
        )
        print(header)
        print("   " + "-" * (len(header) - 3))
        grand_in = grand_out = grand_waste = grand_rem = 0
        for cat, t_in, t_out, t_waste, t_rem, w_pct in category_stats:
            grand_in += int(t_in)
            grand_out += int(t_out)
            grand_waste += int(t_waste)
            grand_rem += int(t_rem)
            print(
                f"   {str(cat):<22} | {int(t_in):>9,} | {int(t_out):>9,} | "
                f"{int(t_waste):>8,} | {int(t_rem):>7,} | {float(w_pct):>6.2f}%"
            )
        overall_waste_pct = (100.0 * grand_waste / grand_in) if grand_in else 0.0
        print("   " + "-" * (len(header) - 3))
        print(
            f"   {'GENEL TOPLAM':<22} | {grand_in:>9,} | {grand_out:>9,} | "
            f"{grand_waste:>8,} | {grand_rem:>7,} | {overall_waste_pct:>6.2f}%"
        )

        print("\n4) v_stock_summary Örnek 10 Satır:")
        s_header = (
            f"   {'Şube':>4} | {'ID':>3} | {'Barkod':<13} | {'Ürün Adı':<28} | "
            f"{'Kategori':<18} | {'SKT':<10} | {'IN':>4} | {'OUT':>4} | {'WST':>4} | {'KAL':>4}"
        )
        print(s_header)
        print("   " + "-" * (len(s_header) - 3))
        for row in sample_rows:
            st_id, pid, bc, pname, cat, exp_d, t_in, t_out, t_wst, rem = row
            short_name = str(pname)[:28]
            short_cat = str(cat)[:18]
            print(
                f"   {int(st_id):>4} | {int(pid):>3} | {str(bc):<13} | {short_name:<28} | "
                f"{short_cat:<18} | {str(exp_d):<10} | {int(t_in):>4} | {int(t_out):>4} | "
                f"{int(t_wst):>4} | {int(rem):>4}"
            )

        print("\n" + "=" * 88)

        return {
            "products_count": products_count,
            "total_movements": total_movements,
            "negative_remaining_count": negative_remaining_count,
            "category_stats": category_stats,
            "overall_waste_pct": round(overall_waste_pct, 2),
        }
