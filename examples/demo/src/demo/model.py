import random

import polars as pl
from ediechain.core.converters import by_init_params
from ediechain.core.registry import converted_by


@converted_by(by_init_params())
class RandomModel:
    """No storage handler covers this, which is why its task is `IN_MEMORY`."""

    def __init__(self, weights: list[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def score(self, frame: pl.DataFrame) -> pl.Series:
        total = pl.lit(self.bias)
        for column, weight in zip(frame.columns, self.weights, strict=False):
            total = total + pl.col(column) * weight
        return frame.select(total.alias("score")).to_series()


def random_model(seed: int = 0, dimensions: int = 3) -> RandomModel:
    generator = random.Random(seed)
    return RandomModel([generator.gauss(0, 1) for _ in range(dimensions)], generator.gauss(0, 1))
