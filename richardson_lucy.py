# Modifed by Craig Russell to work with sparse matrices


# __BEGIN_LICENSE__
#
# Copyright (C) 2010-2013 Stanford University.
# All rights reserved.
#
# __END_LICENSE__

import numpy as np
import time
from numpy import matrix  # scipy.matrix was removed in SciPy 1.12
from scipy.sparse.linalg import LinearOperator

# from lflib.lightfield import LightField
# from lflib.imageio import save_image
# from lflib.linear_operators import LightFieldOperator, RegularizedNormalEquationLightFieldOperator

# ----------------------------------------------------------------------------------------
#                            CONJUGATE GRADIENT SOLVER
# ----------------------------------------------------------------------------------------

def matrix_reconstruction(A, b, x0 = None,
                                   Rtol = 1e-6, NE_Rtol = 1e-6, max_iter = 100,
                                   sigmaSq = 0.0, beta = 0.0):
    '''
    Richardson-Lucy algorithm

    Ported from the RestoreTools MATLAB package available at:
    http://www.mathcs.emory.edu/~nagy/RestoreTools/

    Input: A  -  object defining the coefficient matrix.
           b  -  Right hand side vector.

     Optional Intputs:

           x0      - initial guess (must be strictly positive); default is x0 = A.T*b
           sigmaSq - the square of the standard deviation for the
                     white Gaussian read noise (variance)
           beta    - Poisson parameter for background light level
           max_iter - integer specifying maximum number of iterations;
                      default is 100
           Rtol    - stopping tolerance for the relative residual,
                     norm(b - A*x)/norm(b)
                     default is 1e-6
           NE_Rtol - stopping tolerance for the relative residual,
                     norm(A.T*b - A.T*A*x)/norm(A.T*b)
                     default is 1e-6

    Output:
          x  -  solution

    Original MATLAB code by J. Nagy, August, 2011

    References:
    [1]  B. Lucy.
        "An iterative method for the rectication of observed distributions."
         Astronomical Journal, 79:745-754, 1974.
    [2]  W. H. Richardson.
        "Bayesian-based iterative methods for image restoration.",
         J. Optical Soc. Amer., 62:55-59, 1972.
     [3]  C. R. Vogel.
        "Computational Methods for Inverse Problems",
        SIAM, Philadelphia, PA, 2002


    '''

    # The A operator represents a large, sparse matrix that has dimensions [ nrays x nvoxels ]

    nrays = A.shape[0]
    nvoxels = A.shape[1]

    # Pre-compute some values for use in stopping criteria below
    b_norm = np.linalg.norm(b)
    trAb = A.rmatvec(b)
    trAb_norm = np.linalg.norm(trAb)
    # Start the optimization from the initial volume of a focal stack.
    if x0 is not None:
        x = x0
    else:
        x = trAb

    Rnrm = np.zeros(max_iter+1);
    Xnrm = np.zeros(max_iter+1);
    NE_Rnrm = np.zeros(max_iter+1);

    eps = np.spacing(1)
    tau = np.sqrt(eps);
    sigsq = tau;
    minx = x.min()

    # If initial guess has negative values, compensate
    if minx < 0:
        x = x - min(0,minx) + sigsq;

    normalization = A.rmatvec(np.ones(nrays)) + 1
    print(normalization.min(), normalization.max())
    c = A.matvec(x) + np.matrix(beta*np.ones(nrays)).T + np.matrix(sigmaSq*np.ones(nrays)).T;
    b = b + np.matrix(sigmaSq*np.ones(nrays)).T;

    # i =0
    SAVEFIG =1
    for i in range(max_iter):
        tic = time.time()
        x_prev = x

        # STEP 1: RL Update step
        # b / c
        # (c+1e-12).min()
        v = A.rmatvec(b / c)
        x = (np.multiply(x_prev,v)) / matrix(normalization).T;
        Ax = A.matvec(x)
        # b-Ax
        residual = b - Ax ;
        c = A.matvec(x) + np.matrix(beta*np.ones(nrays)).T + np.matrix(sigmaSq*np.ones(nrays)).T;
        # c = Ax + beta*np.ones(nrays) + sigmaSq*np.ones(nrays);

        # STEP 2: Compute residuals and check stopping criteria
        Rnrm[i] = np.linalg.norm(residual) / b_norm
        Xnrm[i] = np.linalg.norm(x - x_prev) / nvoxels
        #NE_Rnrm[i] = np.linalg.norm(trAb - A.rmatvec(Ax)) / trAb_norm # disabled for now to save on extra rmatvec

        toc = time.time()
        print('\t--> [ RL Iteration %d   (%0.2f seconds) ] ' % (i, toc-tic))
        print('\t      Residual Norm: %0.10g               (tol = %0.12e)  ' % (Rnrm[i], Rtol))
        #print '\t         Error Norm: %0.4g               (tol = %0.2e)  ' % (NE_Rnrm[i], NE_Rtol)
        print('\t        Update Norm: %0.4g                              ' % (Xnrm[i]))

        # stop because residual satisfies ||b-A*x|| / ||b||<= Rtol
        if Rnrm[i] <= Rtol:
            break

        # stop because normal equations residual satisfies ||A'*b-A'*A*x|| / ||A'b||<= NE_Rtol
        #if NE_Rnrm[i] <= NE_Rtol:
        #    break

    return x.astype(np.float32)

#---------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
