from pathlib import Path

import polars as pl

from ediechain.core.storage import TypeHandler, register_handler


class PolarsDataFrameHandler(TypeHandler):
    extension: str = "parquet"

    def save(self, value: pl.DataFrame, path: Path) -> None:
        value.write_parquet(path)

    def load(self, path: Path) -> pl.DataFrame:
        return pl.read_parquet(path)


class PolarsLazyFrameHandler(TypeHandler):
    extension: str = "parquet"

    def save(self, value: pl.LazyFrame, path: Path) -> None:
        value.sink_parquet(path)

    def load(self, path: Path) -> pl.LazyFrame:
        return pl.scan_parquet(path)


class PolarsSeriesHandler(TypeHandler):
    extension: str = "parquet"

    def save(self, value: pl.Series, path: Path) -> None:
        value.to_frame().write_parquet(path)

    def load(self, path: Path) -> pl.Series:
        return pl.read_parquet(path).to_series()


register_handler(pl.DataFrame, PolarsDataFrameHandler())
register_handler(pl.LazyFrame, PolarsLazyFrameHandler())
register_handler(pl.Series, PolarsSeriesHandler())
