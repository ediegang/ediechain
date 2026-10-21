import polars as pl

from ediechain.core.registry import register

register(pl.DataFrame)(lambda df: {"schema": [(n, str(t)) for n, t in df.schema.items()], "rows": df.rows()})
register(pl.Series)(lambda s: {"name": s.name, "dtype": str(s.dtype), "values": s.to_list()})
