from enum import Enum

class ComponentProblemType(str, Enum):
    objective = "objective"
    constraint_matrix = "constraint_matrix"
    free_members = "free_members"

