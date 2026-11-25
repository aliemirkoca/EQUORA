"""Adım 1 — Ham Veri Keşfi (DataInspector).

raw_data/ altındaki tüm CSV, Parquet ve XLSX dosyalarını otomatik tarar;
gerçek dosya adlarını, ayırıcıları, karakter kodlamasını, kolon adlarını,
satır sayılarını ve örnek satırları Polars (LazyFrame) ile raporlar.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fastexcel
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "raw_data"
SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx"}


@dataclass
class DatasetInspection:
    """Tek bir ham veri dosyasına ait keşif özeti."""

    dataset_group: str
    file_path: Path
    file_format: str
    file_size_mb: float
    encoding: str | None = None
    delimiter: str | None = None
    sheet_names: list[str] = field(default_factory=list)
    row_count: int = 0
    columns: list[tuple[str, str]] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)


class DataInspector:
    """raw_data/ altındaki ham veri setlerini keşfeden ve raporlayan OOP sınıfı."""

    CANDIDATE_ENCODINGS = ("utf-8-sig", "utf-8", "cp1254", "latin1")
    CANDIDATE_DELIMITERS = (",", ";", "\t", "|")

    def __init__(self, raw_data_dir: Path | str = DEFAULT_RAW_DIR, sample_size: int = 3) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.sample_size = sample_size

    def discover_files(self) -> list[Path]:
        """raw_data/ altındaki desteklenen tüm veri dosyalarını sıralı döndürür."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Ham veri dizini bulunamadı: {self.raw_data_dir}")

        files = [
            path
            for path in sorted(self.raw_data_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return files

    @classmethod
    def detect_encoding(cls, file_path: Path, sample_bytes: int = 65536) -> str:
        """CSV dosyası için karakter kodlamasını tespit eder."""
        raw = file_path.read_bytes()[:sample_bytes]
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        for enc in cls.CANDIDATE_ENCODINGS[1:]:
            try:
                raw.decode(enc)
                return enc
            except UnicodeDecodeError:
                continue
        return "latin1"

    @classmethod
    def detect_delimiter(cls, file_path: Path, encoding: str) -> str:
        """CSV dosyası için sütun ayırıcısını tespit eder."""
        with file_path.open("r", encoding=encoding, errors="replace") as fh:
            sample_lines = "".join(fh.readline() for _ in range(10))

        try:
            dialect = csv.Sniffer().sniff(sample_lines, delimiters="".join(cls.CANDIDATE_DELIMITERS))
            if dialect.delimiter in cls.CANDIDATE_DELIMITERS:
                return dialect.delimiter
        except csv.Error:
            pass

        first_line = sample_lines.splitlines()[0] if sample_lines else ""
        counts = {sep: first_line.count(sep) for sep in cls.CANDIDATE_DELIMITERS}
        best_sep = max(counts, key=counts.get)
        return best_sep if counts[best_sep] > 0 else ","

    def _inspect_csv(self, file_path: Path, group: str, size_mb: float) -> DatasetInspection:
        encoding = self.detect_encoding(file_path)
        delimiter = self.detect_delimiter(file_path, encoding)
        pl_encoding = "utf8" if encoding in ("utf-8", "utf-8-sig") else "utf8-lossy"

        lf = pl.scan_csv(
            file_path,
            separator=delimiter,
            encoding=pl_encoding,
            infer_schema_length=10000,
            ignore_errors=True,
        )
        schema = lf.collect_schema()
        row_count = int(lf.select(pl.len()).collect().item())
        sample_df = lf.head(self.sample_size).collect()

        return DatasetInspection(
            dataset_group=group,
            file_path=file_path,
            file_format="CSV",
            file_size_mb=size_mb,
            encoding=encoding,
            delimiter=delimiter,
            row_count=row_count,
            columns=[(name, str(dtype)) for name, dtype in schema.items()],
            sample_rows=sample_df.to_dicts(),
        )

    def _inspect_parquet(self, file_path: Path, group: str, size_mb: float) -> DatasetInspection:
        lf = pl.scan_parquet(file_path)
        schema = lf.collect_schema()
        row_count = int(lf.select(pl.len()).collect().item())
        sample_df = lf.head(self.sample_size).collect()

        return DatasetInspection(
            dataset_group=group,
            file_path=file_path,
            file_format="PARQUET",
            file_size_mb=size_mb,
            row_count=row_count,
            columns=[(name, str(dtype)) for name, dtype in schema.items()],
            sample_rows=sample_df.to_dicts(),
        )

    def _inspect_xlsx(self, file_path: Path, group: str, size_mb: float) -> DatasetInspection:
        excel_reader = fastexcel.read_excel(str(file_path))
        sheet_names = list(excel_reader.sheet_names)
        first_sheet = sheet_names[0]

        df = pl.read_excel(file_path, sheet_name=first_sheet, engine="calamine")
        lf = df.lazy()
        schema = lf.collect_schema()
        row_count = int(lf.select(pl.len()).collect().item())
        sample_df = lf.head(self.sample_size).collect()

        return DatasetInspection(
            dataset_group=group,
            file_path=file_path,
            file_format="XLSX",
            file_size_mb=size_mb,
            encoding="xlsx (utf-8)",
            sheet_names=sheet_names,
            row_count=row_count,
            columns=[(name, str(dtype)) for name, dtype in schema.items()],
            sample_rows=sample_df.to_dicts(),
        )

    def inspect_all(self) -> list[DatasetInspection]:
        """Keşfedilen tüm dosyaları analiz edip DatasetInspection listesi döndürür."""
        results: list[DatasetInspection] = []
        for file_path in self.discover_files():
            rel_parts = file_path.relative_to(self.raw_data_dir).parts
            group = rel_parts[0] if len(rel_parts) > 1 else "root"
            size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            ext = file_path.suffix.lower()

            if ext == ".csv":
                results.append(self._inspect_csv(file_path, group, size_mb))
            elif ext == ".parquet":
                results.append(self._inspect_parquet(file_path, group, size_mb))
            elif ext == ".xlsx":
                results.append(self._inspect_xlsx(file_path, group, size_mb))
        return results

    def print_report(self, inspections: list[DatasetInspection] | None = None) -> list[DatasetInspection]:
        """Keşif sonuçlarını okunaklı biçimde konsola raporlar."""
        if inspections is None:
            inspections = self.inspect_all()

        print("=" * 80)
        print(f"EQUORA HAM VERİ KEŞİF RAPORU ({len(inspections)} dosya bulundu)")
        print("=" * 80)

        for idx, info in enumerate(inspections, start=1):
            print(f"\n[{idx}] Grup        : {info.dataset_group}")
            print(f"    Dosya       : {info.file_path.name} ({info.file_size_mb:.2f} MB)")
            print(f"    Tam Yol     : {info.file_path}")
            print(f"    Format      : {info.file_format}")
            if info.encoding:
                print(f"    Kodlama     : {info.encoding}")
            if info.delimiter:
                print(f"    Ayırıcı     : {info.delimiter!r}")
            if info.sheet_names:
                print(f"    Sayfalar    : {info.sheet_names}")
            print(f"    Satır Sayısı: {info.row_count:,}")
            print(f"    Kolonlar ({len(info.columns)}):")
            for col_name, col_type in info.columns:
                print(f"      - {col_name}: {col_type}")
            print(f"    Örnek Satırlar (ilk {len(info.sample_rows)}):")
            for row_idx, row in enumerate(info.sample_rows, start=1):
                compact = {
                    k: (str(v)[:60] + "..." if len(str(v)) > 60 else v)
                    for k, v in row.items()
                }
                print(f"      #{row_idx}: {compact}")

        print("\n" + "=" * 80)
        return inspections


def main() -> None:
    inspector = DataInspector()
    inspector.print_report()


if __name__ == "__main__":
    main()
