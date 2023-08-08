import numpy as np
from src.api_models.input_models import InputModel


class Complicated:
    """
    Структура сложного преобразователя
    """
    def __init__(self, input_model: InputModel):
        self.__m = input_model.number_production_factors
        self.__n = input_model.structure_complexity
        self.__a = input_model.technological_coefficients
        self.__resource = input_model.resource
        self.__cm = np.zeros((self.__u, self.__p), dtype=float)
        self.__ofc = np.zeros(self.__p, dtype=float)
        self.__fm = np.zeros(self.__u, dtype=float)
        self.objective_function_coefficients = self.__build_objective_function_coefficients()
        self.constraint_matrix = self.__build_constraint_matrix()
        self.free_members = self.__build_free_members()

    @property
    def __u(self):
        return int(self.__m * (self.__n + 2))

    @property
    def __p(self):
        return int(((self.__n + 1) * (self.__n + 2 * self.__m) + 2) / 2)

    def __calc_n_i_j(self, i, j):
        return int((self.__n - 1) * i - ((i - 1) * i) / 2 + j - 1)

    def __calc_r_l_i(self, l, i):
        return int(self.__calc_n_i_j(self.__n - 1, self.__n) + (l - 1) * self.__n + i)

    def __calc_n_f(self, i):
        return int(self.__p - 1 - (self.__m - 1) - (self.__n + 1) + i)

    def __calc_r_f(self, i):
        return int(self.__p - 1 - (self.__m - 1) + i - 1)

    def __build_objective_function_coefficients(self):
        for i in np.arange(0, self.__p):
            if i == self.__p - 1:
                self.__ofc[i] = 1.
        return self.__ofc

    def __build_free_members(self):
        for i in np.arange(0,self.__u):
            if i < self.__m:
                self.__fm[i] = self.__resource[i]
        return self.__fm

    def __build_constraint_matrix(self):
        for i in np.arange(1, self.__n + 1):
            self.__cm[0][self.__calc_n_i_j(0, i)] = 1.0
            self.__cm[0][self.__calc_n_f(0)] = 1.0
            for l in np.arange(1, self.__m):
                self.__cm[l][self.__calc_r_l_i(l, i)] = self.__cm[l][self.__calc_r_f(l)] = 1.0

        for i in np.arange(1, self.__n):
            for l in np.arange(0, i):
                self.__cm[self.__m + self.__m * (i - 1)][self.__calc_n_i_j(l, i)] = \
                    - float(self.__a.sum(axis=1)[i] / self.__a[i][0])

            for j in np.arange(i + 1, self.__n + 1):
                self.__cm[self.__m + self.__m * (i - 1)][self.__calc_n_i_j(i, j)] = 1.0
            self.__cm[self.__m + self.__m * (i - 1)][self.__calc_n_f(i)] = 1.0  # 1 заменить на e_i

            for k in np.arange(1, self.__m):
                self.__cm[self.__m + self.__m * (i - 1) + k][self.__calc_r_l_i(k, i)] = \
                    - float(self.__a.sum(axis=1)[i] / self.__a[i][k])

                for j in np.arange(i + 1, self.__n + 1):
                    self.__cm[self.__m + self.__m * (i - 1) + k][self.__calc_n_i_j(i, j)] = 1.0
                self.__cm[self.__m + self.__m * (i - 1) + k][self.__calc_n_f(i)] = 1.0

        for l in np.arange(0, self.__n):
            self.__cm[self.__u - 2 * self.__m][self.__calc_n_i_j(l, self.__n)] = \
                - float(self.__a.sum(axis=1)[self.__n] / self.__a[self.__n][0])

        self.__cm[self.__u - 2 * self.__m][self.__calc_n_f(self.__n)] = 1.0
        for k in np.arange(1, self.__m):
            self.__cm[self.__u - 2 * self.__m + k][self.__calc_r_l_i(k, self.__n)] = \
                - float(self.__a.sum(axis=1)[self.__n] / self.__a[self.__n][k])
            self.__cm[self.__u - 2 * self.__m + k][self.__calc_n_f(self.__n)] = 1.0

        for i in np.arange(0, self.__n + 1):
            self.__cm[self.__u - self.__m][self.__calc_n_f(i)] = \
                - float(self.__a.sum(axis=1)[0] / self.__a[0][0])
            self.__cm[self.__u - self.__m][self.__p - 1] = 1.0

        for k in np.arange(1, self.__m):
            self.__cm[self.__u - self.__m + k][self.__calc_r_f(k)] = \
                - float(self.__a.sum(axis=1)[0] / self.__a[0][k])
            self.__cm[self.__u - self.__m + k][self.__p - 1] = 1.0

        return self.__cm

