from typing import Callable
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, identity
import scipy.sparse.linalg as splu 

from solverpy import Solver, BoundaryCondition

class EDOSolver(Solver):
    def __init__(
        self,
        x_min:float,
        x_max:float,
        number_partitions: int,
        bc_left:float,
        bc_right:float,
        alpha:float,
        source:Callable[[float], float],
        left_condition: BoundaryCondition,
        right_condition: BoundaryCondition
    ) -> None:
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.number_partitions = int(number_partitions)
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.alpha = float(alpha)
        self.source = source
        self.left_condition = left_condition
        self.right_condition = right_condition

        self.grid_width = None
        self.nodes = None
        self.solver_matrix = None
        self.right_hand_term = None
        self.solution = None

    def solve(self)->None:
        self._get_mesh_setting()
        self._get_matrix()
        LU = splu(self.solver_matrix)
        self.solution = LU.solve(self.right_hand_term)
        return self.solution

    def _get_mesh_setting(self) -> None:
        self.grid_width = (self.x_max - self.x_min)/self.number_partitions
        self.nodes = np.linspace(self.x_min, self.x_max, self.number_partitions + 1)
        
    def _get_matrix(self) -> None:
        idty = identity(self.number_partitions + 1, dtype="float64", format="csc")
        laplacian = lil_matrix((self.number_partitions + 1, self.number_partitions + 1), dtype="float64")
        laplacian.setdiag(-2.0, 0)
        laplacian.setdiag(1.0, 1)
        laplacian.setdiag(1.0, -1)
        self._boundary_condition_setting(laplacian)
        laplacian = laplacian.tocsc()

        self.solver_matrix = idty - self.alpha/(self.grid_width*self.grid_width) * laplacian

    def _boundary_condition_setting(
        self,
        laplacian: lil_matrix,
    ) -> None:
        self.right_hand_term = self.source(self.nodes)

        if self.left_condition.is_neumann:  
            laplacian[0, 1] = 2 * laplacian[0, 1]
            self.right_hand_term[0] -= self.bc_left * 2 * self.alpha / self.grid_width
        else:
            laplacian[0, 0] = 0.0
            laplacian[0, 1] = 0.0
            self.right_hand_term[0] = self.bc_left

        if self.right_condition.is_neumann:  # Derecha es Neumann.
            laplacian[self.number_partitions, self.number_partitions - 1] = 2 * laplacian[self.number_partitions, self.number_partitions - 1]
            self.right_hand_term[self.number_partitions] += self.bc_right * 2 * self.alpha / self.grid_width

        else:  
            laplacian[self.number_partitions, self.number_partitions] = 0.0
            laplacian[self.number_partitions, self.number_partitions - 1] = 0.0
            self.right_hand_term[self.number_partitions] = self.bc_right
