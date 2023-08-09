import cvxpy as cp
from src.solver.data_manager import DataManager


class SolverWrapper:

    def __init__(self,
                 data_manager: DataManager):
        self.data_manager = data_manager

    def solve(self):
        variables = self.data_manager.variables
        objective = self.data_manager.objective
        constraints = self.data_manager.constraints

        return self.__solve(variables, constraints, objective)

    def __solve(self, variables, constraints, objective):
        self.prob = cp.Problem(
                objective=cp.Maximize(objective),
                constraints=constraints
            )
        self.prob.solve(solver=cp.SCIPY)
        if self.prob.solution.status == 'optimal':
            return variables.value, objective.value
        else:
            print('The solver could not find an optimal solution')

