import random

import polars as pl


def load_data(rows: int = 500, seed: int = 0) -> pl.DataFrame:
    generator = random.Random(seed)
    return pl.DataFrame(
        {
            "x1": [generator.gauss(0, 1) for _ in range(rows)],
            "x2": [generator.gauss(0, 1) for _ in range(rows)],
            "label": [generator.randrange(2) for _ in range(rows)],
        }
    )
