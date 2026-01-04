import time
from typing import Callable, Optional, Tuple

from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, csc_matrix, identity
from matplotlib.pyplot import *

from solver.base import Solver

class eliptico_2d_dirichlet(Solver):
    def __init__(
    self, 
    xi: float, 
    xf: float, 
    yi: float, 
    yf: float, 
    nu: float, 
    u0: Callable, 
    u1: Callable, 
    u2: Callable, 
    u3: Callable, 
    fuente: Callable,
    exact: Optional[Callable] = None
    ) -> None:
        """Initialize solver for the equation u - nu * (u_xx + u_yy) = f.
        
        Args:
            x0: Lower bound of the X interval.
            xf: Upper bound of the X interval.
            y0: Lower bound of the Y interval.
            yf: Upper bound of the Y interval.
            alpha: Diffusion coefficient (must be positive).
            u0: Lower boundary condition value.
            u1: Right boundary condition value.
            u2: Upper boundary condition value.
            u3: Left boundary condition value.
            fun: Function defining the PDE (callable).
            exact: Optional exact solution for error computation and plotting.
        """
        super().__init__(xi, xf, nu, 0,0, fuente, False, False, exact)
        self._validate_input()
        self.xi = xi
        self.xf = xf
        self.Nx = Nx
        self.yi = yi
        self.yf = yf
        self.Ny = Ny
        self.nu = nu
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3

    def solver(
        self, 
        Internal_Call: bool = False, 
        Nx: Optional[int] = None,
        Ny: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve using a second order finite difference scheme.
        
        Args:
            Internal_Call: If True, suppresses output messages.
            Nx: Number of partitions of the X axis mesh.
            Ny: Number of partitions of the Y axis mesh.
            **kwargs: Additional parameters (unused).
            
        Returns:
            Tuple of (X, Y, solution) arrays.
            
        Raises:
            ValueError: If required parameters are not provided.
        """
        if Nx is None:
            raise ValueError("Nx (number of X axis partitions) must be provided.")
        if Ny is None:
            raise ValueError("Ny (number of X axis partitions) must be provided.")
        t1 = time.time()
        xi=float(self.xi)
        xf=float(self.xf)
        yi=float(self.yi)
        yf=float(self.yf)
        x, y, X, Y, N, A, Mx, My, Id = self._mesh_and_sys_setting(Nx, Ny)
        Mx[0,0]=0.0
        Mx[0,1]=0.0
        Mx[Nx,Nx]=0.0
        Mx[Nx,Nx-1]=0.0
        My[0,0]=0.0
        My[Nx,Nx]=0.0
        
        for i in range(1,Ny): 
            A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx   
            A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My 
            A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
        A=Id+self.nu*A
        A=A.tocsc()

        b=zeros((Ny+1,Nx+1))
        b=self.fuente(X,Y)
        b[0,:]=self.u0(x)
        b[Ny,:]=self.u2(x)
        b[:,0]=self.u3(y)
        b[:,Nx]=self.u1(y)
        b=b.reshape(N)
        LU=splu(A)
        usol=LU.solve(b)

        if not Internal_Call:
            tf = time.time()
            print(f"Running time: {format(tf - t1)}")
        return X, Y, usol
    
    def _mesh_and_sys_setting(
        self, 
        Nx: int, 
        Ny: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, lil_matrix, lil_matrix, lil_matrix, csc_matrix]:
        """Calculate mesh and construct system block matrix.
        
        Args:
            Nx: Number of partitions of the X axis mesh.
            Ny: Number of partitions of the Y axis mesh.
            
        Returns:
            Tuple containing:
                - x: X axis grid points (1D)
                - y: Y axis grid points (1D)
                - X: X axis grid points (matrix)
                - Y: Y axis grid points (matrix)
                - N: Number of points of the 2D grid
                - A: System matrix (LIL format)
                - Mx: Diagonal block submatrix of A (LIL format)
                - My: Infra/Supradiagonal block submatrix of A (LIL format)
                - Id: Identity matrix (CSC format)
        """
        x=linspace(self.xi,self.xf,Nx+1)
        y=linspace(self.yi,self.yf,Ny+1)
        X,Y=meshgrid(x,y)
        dx=(self.xf-self.xi)/float(Nx)
        dy=(self.yf-self.yi)/float(Ny)
        N=(Nx+1)*(Ny+1)
        A = lil_matrix((N,N), dtype='float64');
        Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
        My=lil_matrix((Nx+1,Nx+1),dtype='float64')
        Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
        Mx.setdiag(-1.0/(dx**2)*ones(Nx),1)
        Mx.setdiag(-1.0/(dx**2)*ones(Nx),-1)
        My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)
        Id=identity(N,dtype='float64',format='csc')
        return x, y, X, Y, N, A, Mx, My, Id

    def _validate_input(self) -> None:
        """Validate input parameters.
        
        Raises:
            ValueError: If interval is degenerate or alpha is non-positive.
        """
        if self.xi >= self.xf:
            raise ValueError("Degenerate interval.")
        elif self.yi >= self.yf:
            raise ValueError("Degenerate interval.")
        elif self.nu <= 0:
            raise ValueError("Nu must be positive.")

    def solution_plot(self) -> None:
        """Plot numerical and exact solutions sliced view.
        
        Args:
            **kwargs: Parameters passed to solver method.
            
        Raises:
            ValueError: If no exact solution was provided.
        """
        if self.exact is not None:
            X, Y, usol = self.solver(Internal_Call=True)
            usol = usol.reshape((self.Nx+1,self.Ny+1))
            cu=contourf(X,Y,usol,20)
            colorbar(cu)
            cl=contour(X,Y,usol,20,colors='k')
            clabel(cl,inline=1,fontsize=8)
            show()
        else:
            raise ValueError("No exact solution was provided.")
    


def f0(x,y):
    z=sin(x*y)*(1+x**2+y**2)
    return z
def u0(x):
    z=0*x 
    return z
def u1(y):
    z=sin(2*pi*y) 
    return z
def u2(x):
    z=sin(2*pi*x) 
    return z
def u3(y):
    z=0*y 
    return z
def exacta(x,y):
    z=sin(x*y)
    return z

solver  = eliptico_2d_dirichlet(0.0,2*pi,0.0,2*pi,1.0,u0,u1,u2,u3,f0,exacta)
solver.solution_plot(200,200)