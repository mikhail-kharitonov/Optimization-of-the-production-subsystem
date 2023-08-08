import pydantic_numpy.typing as pnd
from pydantic_numpy.model import NumpyModel


class InputModel(NumpyModel):
    number_production_factors: int
    structure_complexity: int
    resource: pnd.Np1DArrayFp32
    technological_coefficients: pnd.Np2DArrayFp32


class InputJsonModel(NumpyModel):
    flows_value: pnd.Np1DArrayFp32
    objective_value: float

