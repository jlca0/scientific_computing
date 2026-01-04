from abc import ABC, abstractmethod


class BoundaryCondition:

    def __init__(
        self,
        condition:str
    ) -> None:
        """
        Object that indicates whether a boundary condition is Neumann or, by default, Dirichlet.

        Atributes:
            condition: (str) A string, either 'Dirichlet' or 'Neumann'
        """
        if condition in ['Neumann', 'Dirichlet']:
            self.condition = condition
        else:
            raise ValueError('Boundary conditions are either Dirichlet or Neumann'
            )

    @property
    def is_neumann(self)->bool:
        """
        Checks whether a condition is Neumann.

        Returns:
             : (bool) True if and only if Neumann.
        """
        return self.condition == 'Neumann'

class Solver(ABC):

    @abstractmethod
    def solve(self)->None:
        """
        Abstract method for solving a given PDE.
        """
        pass

    @abstractmethod
    def visualize(self)->None:
        """
        Abstract method for visualizing the solution of a given PDE.
        """
        pass

    @abstractmethod
    def _get_mesh_setting(self)->None:
        """
        Abstract private method that sets the mesh of the solver.
        """
        pass
    
    @abstractmethod
    def _get_matrix(self)->None:
        """
        Abstract private method that sets the discretization matrices of
        the solver.
        """
        pass