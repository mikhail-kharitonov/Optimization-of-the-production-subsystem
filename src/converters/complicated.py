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
        """
        Количество ограничений в ЗЛП
        :return: int
        """
        return int(self.__m * (self.__n + 2))

    @property
    def __p(self):
        """
        Количество переменных в ЗЛП
        :return: int
        """
        return int(((self.__n + 1) * (self.__n + 2 * self.__m) + 2) / 2)

    def __calc_location_flow_type_n_from_node_i_to_j(self, i, j):
        """
        Вычисляется номер компонеты потока N^i_j в векторе переменных
        :return: int
        """
        return int((self.__n - 1) * i - ((i - 1) * i) / 2 + j - 1)

    def __calc_location_flow_type_r_from_node_l_to_j(self, l, i):
        """
        Вычисляется номер компонеты потока R^l_i в векторе переменных
        :return: int
        """
        n_ij = self.__calc_location_flow_type_n_from_node_i_to_j
        return int(n_ij(self.__n - 1,self.__n) + (l - 1) * self.__n + i)

    def __calc_location_flow_type_n_from_node_i_to_f(self, i):
        """
        Вычисляется номер компонеты потока N^i_F в векторе переменных
        :return: int
        """
        return int(self.__p - 1 - (self.__m - 1) - (self.__n + 1) + i)

    def __calc_location_flow_type_r_from_node_i_to_f(self, i):
        """
        Вычисляется номер компонеты потока R^i_F в векторе переменных
        :return: int
        """
        return int(self.__p - 1 - (self.__m - 1) + i - 1)

    def __build_objective_function_coefficients(self):
        """
        Построение вектора коэффициентов целевой функции ЗЛП
        :return: List[float]
        """
        self.__ofc[self.__p - 1] = 1.
        return self.__ofc

    def __build_free_members(self):
        """
        Построение вектора правых частей ЗЛП
        :return: List[float]
        """

        for i in np.arange(0,self.__m):
            self.__fm[i] = self.__resource[i]
        return self.__fm

    def __build_constraint_matrix(self):
        """
        Построение матрицы ЗЛП
        :return: List[float,float]
        """

        n = self.__n
        m = self.__m
        a = self.__a
        c = self.__cm
        u = self.__u
        p = self.__p
        n_ij = self.__calc_location_flow_type_n_from_node_i_to_j
        n_f = self.__calc_location_flow_type_n_from_node_i_to_f
        r_li = self.__calc_location_flow_type_r_from_node_l_to_j
        r_f = self.__calc_location_flow_type_r_from_node_i_to_f

        for i in np.arange(1, n + 1):
            c[0][n_ij(0, i)] = 1.0
            c[0][n_f(0)] = 1.0
            for l in np.arange(1, m):
                c[l][r_li(l, i)] = 1.0
                c[l][r_f(l)] = 1.0

        for i in np.arange(1, n):
            for l in np.arange(0, i):
                c[m + m * (i - 1)][n_ij(l, i)] = - float(a.sum(axis=1)[i] / a[i][0])

            for j in np.arange(i + 1, n + 1):
                c[m + m * (i - 1)][n_ij(i, j)] = 1.0

            c[m + m * (i - 1)][n_f(i)] = 1.0  # 1 заменить на e_i

            for k in np.arange(1, m):
                c[m + m * (i - 1) + k][r_li(k, i)] = - float(a.sum(axis=1)[i] / a[i][k])

                for j in np.arange(i + 1, n + 1):
                    c[m + m * (i - 1) + k][n_ij(i, j)] = 1.0
                c[m + m * (i - 1) + k][n_f(i)] = 1.0

        for l in np.arange(0, n):
            c[u - 2 * m][n_ij(l, n)] = - float(a.sum(axis=1)[n] / a[n][0])

        c[u - 2 * m][n_f(n)] = 1.0
        for k in np.arange(1, m):
            c[u - 2 * m + k][r_li(k, n)] = - float(a.sum(axis=1)[n] / a[n][k])
            c[u - 2 * m + k][n_f(n)] = 1.0

        for i in np.arange(0, n + 1):
            c[u - m][n_f(i)] = - float(a.sum(axis=1)[0] / a[0][0])
            c[u - m][p - 1] = 1.0

        for k in np.arange(1, m):
            c[u - m + k][r_f(k)] = - float(a.sum(axis=1)[0] / a[0][k])
            c[u - m + k][p - 1] = 1.0

        return c

