import cvxpy as cp

from src.converters.complicated import Complicated
from src.converters.dedicated import Dedicated
from src.api_models.converter_type import ConverterType
from src.api_models.component_problem_type import ComponentProblemType


class DataManager:
    def __init__(self,
                 complicated: Complicated,
                 dedicated: Dedicated,
                 converter_type: ConverterType):

        self.converter_type = converter_type
        self.complicated = complicated
        self.dedicated = dedicated
        self.component_problem = {
            component_type: []
            for component_type in ComponentProblemType
        }

        self.variables = cp.Variable(
            shape=len(self.__build_component_problem[ComponentProblemType.objective])
        )
        self.objective = self.__build_objective_function()
        self.constraints = self.__build_constraints()

    @property
    def __build_component_problem(self):
        match self.converter_type:

            case self.converter_type.complicated:
                objective_function_coefficients = (
                    self.complicated.objective_function_coefficients)
                constraint_matrix = (
                    self.complicated.constraint_matrix)
                free_members = self.complicated.free_members

                self.component_problem[ComponentProblemType.objective] = (
                    objective_function_coefficients)
                self.component_problem[ComponentProblemType.constraint_matrix] = (
                    constraint_matrix)
                self.component_problem[ComponentProblemType.free_members] = (
                    free_members)
                return self.component_problem

            case self.converter_type.dedicated:
                objective_function_coefficients = (
                    self.dedicated.objective_function_coefficients)
                constraint_matrix = (
                    self.dedicated.constraint_matrix)
                free_members = (
                    self.dedicated.free_members)

                self.component_problem[ComponentProblemType.objective] = (
                    objective_function_coefficients)
                self.component_problem[ComponentProblemType.constraint_matrix] = (
                    constraint_matrix)
                self.component_problem[ComponentProblemType.free_members] = (
                    free_members)
                return self.component_problem

    def __build_objective_function(self):
        objective = (
            self.__build_component_problem)[ComponentProblemType.objective]
        return cp.sum(
                    cp.multiply(objective,
                                self.variables)
                )

    def __build_constraints(self):
        constraint_matrix = (
            self.__build_component_problem)[ComponentProblemType.constraint_matrix]
        free_members = (
            self.__build_component_problem)[ComponentProblemType.free_members]
        variables = self.variables
        return [constraint_matrix @ variables <= free_members,
                variables >= 0]

