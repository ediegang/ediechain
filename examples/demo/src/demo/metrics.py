from typing import cast

import polars as pl


def accuracy(scored: pl.DataFrame) -> float:
    return float(cast("float", (scored["label"] == scored["predicted"]).mean()))
