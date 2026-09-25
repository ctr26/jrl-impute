# Can use Richardson Lucy to deconvolve with a spatially varying PSF
# 1) Use spatially invariant PSF for RL on skimage image
# 2) Use spatially varying PSF for RL on skimage
#   How do abberate  - skip
# 3) Put NaNs in matrix and impute

# https://scikit-learn.org/stable/modules/impute.html
# https://en.wikipedia.org/wiki/Matrix_completion
# https://en.wikipedia.org/wiki/Netflix_Prize
# https://stackoverflow.com/questions/17982931/matrix-completion-in-python

# https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.richardson_lucy

from skimage.metrics import structural_similarity as ssim

import numpy as np
import matplotlib.pyplot as plt
# import microscPSF.microscPSF as msPSF
import PIL
import scipy

from numpy import matrix  # scipy.matrix was removed in SciPy 1.12
from scipy.sparse import coo_matrix
import time
from scipy import linalg
from skimage import color, data, restoration
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.signal import convolve2d as conv2
# import matlab.engine
import pandas as pd

import richardson_lucy

# from sklearn.preprocessing import Imputer  # removed in scikit-learn 0.22; use sklearn.impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer

import os

#%% Creating the test card image

scale = 4.0
# scale = 1.0
# int(12/scale)
psf = np.ones((int(12/scale),int(12/scale)))/int(12/scale)**2 #Boxcar
astro = rescale(color.rgb2gray(data.astronaut()),1.0/scale);
astro_blur = conv2(astro, psf, 'same')# astro_blur = rescale(astro_blur, 1.0 / 4)

# Add Noise to Image
astro_noisy = astro_blur.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro_blur.shape) - 10) / 255.
# astro_blur
# deconvolved_RL = restoration.richardson_lucy(astro_blur, psf, iterations=100)
astro_blur = astro_noisy
plt.imshow(astro_noisy)
plt.imshow(psf)

# plt.imshow(deconvolved_RL)

#%% Build measurement matrix.
print("Build measurement matrix.")
##s H = [h_i,j] each row is a line of PSF centered on x,y filled with zeros

N_v = np.ma.size(astro);N_v
N_p = np.ma.size(astro_blur);N_p
measurement_matrix = matrix(np.zeros((N_p,N_v)))

zero_image = np.zeros_like(astro)
astro_shape = astro.shape

for i in np.arange(N_v):
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    delta_PSF = scipy.ndimage.convolve(delta_image,psf)
    # NB: the point response of pixel i is column i of H (b = H x); writing it to
    # row i builds H^T, which is only equal to H for one symmetric PSF. See jrl_impute.forward_matrix.
    measurement_matrix[i,:] = delta_PSF.flatten()
    # plt.imshow(delta_image)
    # plt.show()
astro_noisy_vector = np.matrix(astro_noisy.flatten()).transpose();astro_noisy_vector
plt.imshow(measurement_matrix)
# plt.show()
plt.imshow(psf)
#%% Begin RL matrix deconvolvution
print("Build measurement matrix.")
#
# x0 = None
# Rtol = 1e-6
# NE_Rtol = 1e-6
# max_iter = 100
# sigmaSq = 0.0
# beta = 0.0

measurement_matrix_LO = scipy.sparse.linalg.aslinearoperator(measurement_matrix);measurement_matrix_LO
input_vector = astro_noisy_vector


# Raw RL, no imputation
FLAG_RAW = 1
if(FLAG_RAW):
    astro_rl_flat = richardson_lucy.matrix_reconstruction(
                    scipy.sparse.linalg.aslinearoperator(measurement_matrix),
                    input_vector,max_iter=30)
    # astro_rl = astro
    astro_rl = np.reshape(np.array(astro_rl_flat),astro_blur.shape)

    fig, ax = plt.subplots(1,3,figsize=[6.4*2, 4.8*2])

    ax[0].imshow(astro)
    ax[0].title.set_text("Raw")

    ax[1].imshow(astro_noisy)
    ax[1].title.set_text("Corrupted")

    ax[2].imshow(astro_rl)
    ax[2].title.set_text("Recovered")


#%% Matrix imputation

H_df = pd.DataFrame(measurement_matrix)

#Remove all random sampled rows
beads = 100

rows_to_nuke = np.random.choice(np.arange(measurement_matrix.shape[0]),measurement_matrix.shape[0]-beads)
# rows_to_nuke=np.random.choice(np.arange(0,measurement_matrix.shape[0]), measurement_matrix.shape[0]-beads, replace=False);rows_to_nuke
# rows_to_nuke = (np.rint(np.random.choice(H.shape[0]-beads))*H.shape[0]).astype(int)
H_nuked = measurement_matrix.copy()

for i in rows_to_nuke:
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    psf_nan = np.zeros_like(psf)*np.nan;psf_nan
    delta_PSF = scipy.ndimage.convolve(delta_image,psf_nan)
    # plt.imshow(delta_PSF)
    H_nuked[i,:] = delta_PSF.flatten()
    # delta_PSF = psf_xy
    # plt.imshow(delta_image)
    # plt.show()
plt.imshow(H_nuked)
plt.savefig("output/H_nuked.png")

imp = SimpleImputer(missing_values=np.nan, strategy='mean',verbose=1)
imp = IterativeImputer(missing_values=np.nan,verbose=1)
imp.fit(H_nuked)
H_fixed = imp.transform(H_nuked)
error = ssim(measurement_matrix,H_fixed)
# plt.show()
plt.imshow(H_fixed)
