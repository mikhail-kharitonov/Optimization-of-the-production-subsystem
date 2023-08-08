import json
import numpy as np


from src.api_models.converter_type import ConverterType
from src.api_models.input_model import InputModel
from src.converters.complicated import Complicated
from src.converters.dedicated import Dedicated
from src.solver.data_manager import DataManager
from src.solver.solver_wrapper import SolverWrapper

cases = (["case_complicated_3x7.json",
          "case_complicated_5x10.json"])


def test_solver_results():

    for case in cases:
        print(f'\n{case}')
        with open("data/solver/" + case, "r") as jf_in:
            req = json.load(jf_in)

        input_data = {"number_production_factors": req["problem"]["numberProductionFactors"],
                      "structure_complexity": req["problem"]["structureComplexity"],
                      "resource": np.array(req["problem"]["resource"]),
                      "technological_coefficients": np.array(req["problem"]["technologicalCoefficients"])
                      }

        im = InputModel(**input_data)

        dm = DataManager(Complicated(im),
                         Dedicated(im),
                         ConverterType.complicated)

        sw = SolverWrapper(dm)
        calc_flows_value, calc_objective_value = sw.solve()

        optimal_flows_value = np.array(req["correctResult"]["optimalFlowsValue"])

        for row in range(len(optimal_flows_value)):
            assert np.abs(optimal_flows_value[row] - calc_flows_value[row]) <= 0.000001

        optimal_objective_value = np.array(req["correctResult"]["optimalObjectiveValue"])
        assert np.abs(optimal_objective_value - calc_objective_value) <= 0.000001
