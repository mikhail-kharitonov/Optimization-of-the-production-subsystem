import json
import numpy as np


from src.api_models.converter_type import ConverterType
from src.api_models.input_models import InputModel
from src.converters.complicated import Complicated
from src.converters.dedicated import Dedicated
from src.solver.data_manager import DataManager
from src.solver.solver_wrapper import SolverWrapper
from src.api_models.input_models import InputJsonModel

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

        input_data_from_json = {
            "flows_value": req["correctResult"]["optimalFlowsValue"],
            "objective_value": req["correctResult"]["optimalObjectiveValue"]
        }
        ijm = InputJsonModel(**input_data_from_json)

        dm = DataManager(Complicated(im),
                         Dedicated(im),
                         ConverterType.complicated)

        sw = SolverWrapper(dm)
        calc_flows_value, calc_objective_value = sw.solve()

        for row in range(len(calc_flows_value)):
            assert np.abs(ijm.flows_value[row] - calc_flows_value[row]) <= 0.00001

        assert np.abs(ijm.objective_value - calc_objective_value) <= 0.00001
