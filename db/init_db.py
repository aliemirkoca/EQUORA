"""EQUORA PostgreSQL veritabanı ve şema kurulum scripti.

.env dosyasından bağlantı bilgilerini okur, hedef veritabanı yoksa oluşturur,
db/schema.sql dosyasını uygular ve tablo/view yapısını doğrular.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

EXPECTED_TABLES = ("products", "stock_movements")
EXPECTED_VIEWS = ("v_stock_summary",)
PLACEHOLDER_PASSWORD = "BURAYA_SIFRE_GELECEK"


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    def connection_params(self, dbname: str | None = None) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": dbname or self.dbname,
            "user": self.user,
            "password": self.password,
        }


def load_db_config(env_path: Path = ENV_PATH) -> DatabaseConfig:
    """Proje kökündeki .env dosyasından PostgreSQL bağlantı bilgilerini yükler."""
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)

    host = os.getenv("DB_HOST", "localhost")
    port_raw = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "equora_db")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")

    if password == PLACEHOLDER_PASSWORD:
        raise ValueError(
            f"Lütfen '{env_path}' dosyasındaki DB_PASSWORD alanına yerel PostgreSQL "
            "şifrenizi girin."
        )

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"Geçersiz DB_PORT değeri: {port_raw!r}") from exc

    return DatabaseConfig(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def ensure_database_exists(config: DatabaseConfig) -> bool:
    """Hedef veritabanı yoksa 'postgres' veritabanı üzerinden oluşturur.

    Returns:
        Veritabanı yeni oluşturulduysa True, zaten mevcutsa False.
    """
    conn = psycopg2.connect(**config.connection_params(dbname="postgres"))
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;",
                (config.dbname,),
            )
            exists = cur.fetchone() is not None
            if exists:
                print(f"[BİLGİ] '{config.dbname}' veritabanı zaten mevcut.")
                return False

            cur.execute(
                sql.SQL("CREATE DATABASE {};").format(sql.Identifier(config.dbname))
            )
            print(f"[OK] '{config.dbname}' veritabanı oluşturuldu.")
            return True
    finally:
        conn.close()


def apply_schema(config: DatabaseConfig, schema_path: Path = SCHEMA_PATH) -> None:
    """Verilen SQL şema dosyasını hedef veritabanında uygular."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Şema dosyası bulunamadı: {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")

    conn = psycopg2.connect(**config.connection_params())
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        print(f"[OK] Şema başarıyla uygulandı: {schema_path.name}")
    finally:
        conn.close()


def verify_schema(config: DatabaseConfig) -> bool:
    """Beklenen tablo ve view nesnelerinin oluştuğunu doğrular."""
    conn = psycopg2.connect(**config.connection_params())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                  AND table_name = ANY(%s);
                """,
                (list(EXPECTED_TABLES),),
            )
            found_tables = {row[0] for row in cur.fetchall()}

            cur.execute(
                """
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'public'
                  AND table_name = ANY(%s);
                """,
                (list(EXPECTED_VIEWS),),
            )
            found_views = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    print("\n--- Doğrulama Sonuçları ---")
    all_ok = True

    for table in EXPECTED_TABLES:
        if table in found_tables:
            print(f"  [OK] Tablo mevcut : {table}")
        else:
            print(f"  [HATA] Tablo eksik: {table}")
            all_ok = False

    for view in EXPECTED_VIEWS:
        if view in found_views:
            print(f"  [OK] View mevcut  : {view}")
        else:
            print(f"  [HATA] View eksik : {view}")
            all_ok = False

    return all_ok


def main() -> int:
    try:
        config = load_db_config()
        print(
            f"[BİLGİ] PostgreSQL bağlantısı: "
            f"{config.user}@{config.host}:{config.port}/{config.dbname}"
        )
        ensure_database_exists(config)
        apply_schema(config)
        if verify_schema(config):
            print("\n[BAŞARILI] EQUORA veritabanı altyapısı hazır.")
            return 0

        print("\n[HATA] Bazı veritabanı nesneleri doğrulanamadı.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[YAPILANDIRMA HATASI] {exc}", file=sys.stderr)
        return 1
    except psycopg2.Error as exc:
        print(f"[VERİTABANI HATASI] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[BEKLENMEYEN HATA] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
