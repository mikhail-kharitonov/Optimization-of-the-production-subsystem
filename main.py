import numpy as np
from src.api_models.input_model import InputModel
from src.solver.solver_wrapper import SolverWrapper
from src.solver.data_manager import DataManager
from src.converters.complicated import Complicated
from src.converters.dedicated import Dedicated
from src.api_models.converter_type import ConverterType

m = 3
n = 3
resource = np.array([1.0, 10, 10]) #np.random.uniform(1, 5, size=m),
technological_coefficients = np.random.uniform(10, 40, size=(n+1, m)) #np.ones((n+1, m)) # np.random.uniform(10, 40, size=(n+1, m))

input_data = {"number_production_factors": m,
              "structure_complexity": n,
              "resource": resource,
              "technological_coefficients": technological_coefficients
              }

im = InputModel(**input_data)

print(im.technological_coefficients)


dm = DataManager(Complicated(im),
                 Dedicated(im),
                 ConverterType.complicated)

sw = SolverWrapper(dm)
sw.solve()

print(sw.solve())


