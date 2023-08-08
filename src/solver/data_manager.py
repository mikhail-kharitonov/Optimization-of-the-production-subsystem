import numpy as np
import cvxpy as cp

from src.converters.complicated import Complicated
from src.converters.dedicated import Dedicated
from src.api_models.converter_type import ConverterType


class DataManager:
    def __init__(self,
                 complicated: Complicated,
                 dedicated: Dedicated,
                 converter_type: ConverterType):
        self.converter_type = converter_type
        self.complicated = complicated
        self.dedicated = dedicated
        self.__objective_function_coefficients = self.__build_component_problem["objective_coefficients"]
        self.__constraint_matrix = self.__build_component_problem["constraint_matrix"]
        self.__free_members = self.__build_component_problem["free_members"]

        self.variables = cp.Variable(
            shape=len(self.__objective_function_coefficients)
        )
        self.objective = self.__build_objective_function()
        self.constraints = self.__build_constraints()

    @property
    def __build_component_problem(self):
        match self.converter_type:

            case self.converter_type.complicated:
                objective_function_coefficients = self.complicated.objective_function_coefficients
                constraint_matrix = self.complicated.constraint_matrix
                free_members = self.complicated.free_members

                component_problem = {
                    "objective_coefficients": objective_function_coefficients,
                    "constraint_matrix": constraint_matrix,
                    "free_members": free_members
                }
                return component_problem

            case self.converter_type.dedicated:
                objective_function_coefficients = self.dedicated.objective_function_coefficients
                constraint_matrix = self.dedicated.constraint_matrix
                free_members = self.dedicated.free_members


                component_problem = {
                    "objective_coefficients": objective_function_coefficients,
                    "constraint_matrix": constraint_matrix,
                    "free_members": free_members
                }
                return component_problem

    def __build_objective_function(self):
        return cp.sum(
                    cp.multiply(self.__objective_function_coefficients,
                                self.variables)
                )

    def __build_constraints(self):
        return [
            self.__constraint_matrix @ self.variables <= self.__free_members,
            self.variables >= 0]

