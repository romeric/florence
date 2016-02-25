import numpy as np

class FEMSolver():
    """Solver for linear and non-linear finite elements.
        This is different from the LinearSolver, as linear solver
        specifically deals with solution of matrices, whereas FEM
        solver is essentially responsible for linear, linearised
        and nonlinear finite element formulations
    """

    def __init__(self):

        self.is_geometrically_linearised = True
        self.requires_geometry_update = True
        self.requires_line_search = False
        self.requires_arc_length = False
        self.has_moving_boundary = False

        self.number_of_load_increments = 1
        self.newton_raphson_tolerance = 2.0e-04
        self.maximum_iteration_for_newton_raphson = 50


    @property
    def WhichFEMSolver():
        pass

    @property
    def WhichFEMSolvers():
        pass
