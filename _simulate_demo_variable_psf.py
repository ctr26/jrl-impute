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

import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout, Activation, Convolution1D, Flatten, Conv1D,UpSampling1D,InputLayer,UpSampling2D,Conv2D,Reshape,Input
from keras.optimizers import SGD
from keras.utils import to_categorical

# from sklearn.preprocessing import Imputer  # removed in scikit-learn 0.22; use sklearn.impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
from scipy import signal
from sklearn.impute import SimpleImputer

import os

#%% Creating the test card image

scale = 4.0
# scale = 1.0
# int(12/scale)
static_psf = np.ones((int(12/scale),int(12/scale)))/int(12/scale)**2 #Boxcar
def psf_guass(w=10,h=10,sigma=3):
    # blank_psf = np.zeros((w,h))
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    xx,yy = np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
    return gaussian(xx,0,sigma)*gaussian(yy,0,sigma)

static_psf =  psf_guass(w=10, h=10, sigma=1/5)
plt.imshow(static_psf)


astro = rescale(color.rgb2gray(data.astronaut()),1.0/scale);
astro_blur = conv2(astro, static_psf, 'same')# astro_blur = rescale(astro_blur, 1.0 / 4)

# Add Noise to Image
astro_noisy = astro_blur.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro_blur.shape) - 10) / 255.
# astro_blur
# deconvolved_RL = restoration.richardson_lucy(astro_blur, psf, iterations=100)
astro_blur = astro_noisy
plt.imshow(astro_noisy)
plt.imshow(static_psf)

# plt.imshow(deconvolved_RL)

#%% Build measurement matrix.
print("Build measurement matrix.")
##s H = [h_i,j] each row is a line of PSF centered on x,y filled with zeros

N_v = np.ma.size(astro);N_v
N_p = np.ma.size(astro_blur);N_p
measurement_matrix = matrix(np.zeros((N_p,N_v)))

zero_image = np.zeros_like(astro)
astro_shape = astro.shape

x_astro,y_astro = astro_blur.shape
xx_astro,yy_astro = np.meshgrid(np.linspace(-1,1,x_astro),np.linspace(-1,1,y_astro))
psf_window_w,psf_window_h = (10,10)
psf_window_volume = np.full((psf_window_w,psf_window_h,N_v),np.nan)

illumination = np.cos(64/2*np.pi*xx_astro)

plt.imshow(psf_guass(w=psf_window_w,
                    h=psf_window_h,
                    sigma=sigma_scale(1)))

def sigma_scale(r_dist):
    return (r_dist+0.01)*3


r_map = np.sqrt(xx_astro**2+yy_astro**2)

psf_current = psf_guass(w=psf_window_w,
                        h=psf_window_h,
                        sigma=sigma_scale(r_map.max().max()))
plt.imshow(psf_current)

r_dist = np.empty(N_v)
sigma = np.empty(N_v)
psf_window_volume = np.empty((10,10,N_v))

plt.imshow(r_map)
plt.imshow(illumination)
# for i in np.arange(N_v):
#     coords = np.unravel_index(i,astro.shape)
#     print(r_map[coords])
for i in np.arange(N_v):
    coords = np.unravel_index(i,astro.shape)
    r_dist = r_map[coords]
    sigma = sigma_scale(r_map[coords])
    psf_current = psf_guass(w=psf_window_w,
                            h=psf_window_h,
                            sigma=sigma*illumination[coords])
    # psf_current = psf_guass(w=psf_window_w,
    #                         h=psf_window_h,
    #                         sigma=sigma[i])
    psf_window_volume[:,:,i] = psf_current
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    delta_PSF = scipy.ndimage.convolve(delta_image,psf_current)
    # NB: the point response of pixel i is column i of H (b = H x); writing it to
    # row i builds H^T, which is only equal to H for one symmetric PSF. See jrl_impute.forward_matrix.
    measurement_matrix[i,:] = delta_PSF.flatten()
    # plt.imshow(psf_current)
    # plt.imsave(f'./output/psfs/{str(i).zfill(6)}.png',psf_window_volume[:,:,i])
    plt.show()
# pd.DataFrame(measurement_matrix)
astro_noisy_vector = np.matrix(astro_noisy.flatten()).transpose();
# plt.imshow(measurement_matrix)
# plt.show()
# plt.imshow(static_psf)

#%% Begin RL matrix deconvolvution - Nuke beads


print("Regress full PSF model")

beads = 100

rows_to_nuke = np.random.choice(np.arange(measurement_matrix.shape[0]),measurement_matrix.shape[0]-beads)
# rows_to_nuke
psf_window_volume_nuked = psf_window_volume.copy()
psf_window_volume_nuked[:,:,rows_to_nuke] = np.nan

X_indices = np.array(np.unravel_index(np.arange(0,psf_window_volume.size), psf_window_volume.shape)).T
y_values = np.array(psf_window_volume_nuked.flatten())

# from scipy import interpolate
# rbfi = interpolate.Rbf(X_indices[:,0],X_indices[:,1],X_indices[:,2],y_values)

X_indices_clean = X_indices[np.isfinite(y_values)]
y_values_clean = y_values[np.isfinite(y_values)]

y_values_clean_2d = np.vstack(y_values_clean)

#%% NN models

from sklearn import linear_model
from sklearn import svm
from scipy.stats import pearsonr
from sklearn import neural_network,metrics,gaussian_process,preprocessing,svm,neighbors


def keras_model_cov():
    model = Sequential()
    model.add(InputLayer(input_shape=X_indices_clean_scaled.shape))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    # model.add(UpSampling1D(size=2))
    # model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    # model.add(UpSampling1D(size=2))
    # model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(UpSampling1D(size=2))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def keras_model_fc():
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

classifiers = [
    # svm.SVR(),
    # neural_network.MLPRegressor(hidden_layer_sizes=(64,64),
    #                             verbose=True),
    # neighbors.KNeighborsRegressor(),
    KerasRegressor(build_fn=keras_model_cov, epochs=100,nb_epoch=100,batch_size=64, verbose=1),
    KerasRegressor(build_fn=keras_model_fc, epochs=100,nb_epoch=100,batch_size=64, verbose=1)
    # svm.SVR(),
    # gaussian_process.GaussianProcessRegressor(),
    # linear_model.SGDRegressor(),
    # linear_model.BayesianRidge(),
    # linear_model.LassoLars(),
    # linear_model.ARDRegression(),
    # linear_model.PassiveAggressiveRegressor(),
    # linear_model.TheilSenRegressor(),
    # linear_model.LinearRegression()
    ]

#Need to invert scaling



#%%
y_ground_truth = psf_window_volume.flatten()

X_indices_clean_scaled = preprocessing.scale(X_indices_clean)
X_indices_scaled = preprocessing.scale(X_indices)

y_values_clean_2d_scaled = preprocessing.scale(y_values_clean_2d)
y_ground_truth_scaled = preprocessing.scale(y_ground_truth)


batch_size = 32

# DO IT ALL IN 2D

from keras import backend as K

# (*X_indices_scaled.shape,1)
# histor = the_model.fit(X_indices_clean_scaled, y_values_clean_2d_scaled)
# X_indices_scaled.shape
X_indices_scaled.shape
samples = X_indices_scaled.shape[0]
feature_size = 3

x = np.expand_dims(X_indices_scaled,-1)


model = Sequential()
layer_shape = 2
model.add(Dense(layer_shape, activation='relu'))
for i in np.arange(0,5):
    model.add(Reshape((layer_shape,1,1)))
    model.add(UpSampling2D(size=(2,1)))
    model.add(Reshape((layer_shape*2,1)))
    model.add(Conv1D(filters=1,kernel_size=2, activation='relu'))
    model.add(Flatten())
    # model.add(Dropout(0.5))

    layer_shape = layer_shape*2-1

# model.add(Reshape((3,1,1)))
# model.add(UpSampling2D(size=(2,1)))
# model.add(Reshape((6,1)))
# model.add(Conv1D(filters=1,kernel_size=2, activation='relu'))
# model.add(Flatten())
#
# model.add(Reshape((5,1,1)))
# model.add(UpSampling2D(size=(2,1)))
# model.add(Reshape((10,1)))
# model.add(Conv1D(filters=1,kernel_size=2, activation='relu'))
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
# loss = keras.optimizers.Adadelta()
# model.compile(loss='mean_squared_error',
#               optimizer=loss,
#               metrics=['accuracy'])
# model.compile(loss='logcosh',
#               optimizer='sgd',
#               metrics=['accuracy'])
model.build()

from keras import metrics
# model.compile(loss='logcosh',
#               optimizer='adam',
#               metrics=metrics.mae, metrics.accuracy])
# return model
# model.build()
# model.summary()
x = np.array(np.unravel_index(np.arange(0,astro.size),astro.shape)).T
y = np.reshape(psf_window_volume,(100,int(samples/100))).T

# plt.imshow(psf_window_volume[:,:,int(128/4*127/4)])
model.fit(x, y,
          batch_size=1,
          epochs=10)
model.summary()
u#%%

coords = np.unravel_index(i,astro.shape)

# plt.imshow()
y_predict = model.predict(x)
for i in np.arange(N_v):
    y_predict_current = y_predict[i,:].reshape((10,10))
    plt.imsave(f'./output/predict_psf/{str(i).zfill(6)}.png',y_predict_current)
# plt.imsave(,)

# np.nansum(np.sqrt(y_predict**2-y**2))

plt.imshow(a.reshape((10,10)))

plt.imshow(psf_window_volume[:,:,].reshape((10,10)))
samples = psf_window_volume_nuked.shape[2]


# model.add(Reshape(target_shape=(feature_size,1)))
# model.add(Conv1D(filters=1,kernel_size=2, activation='relu'))

# model.add(Dense(128, activation='relu'))
# model.add(Flatten())
# model.add(InputLayer(input_shape=(3,)))
# model.add(Reshape((3,1,1)))
# model.add(UpSampling2D(size=(2,1)))
# model.add(Flatten())

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='relu'))

# model.add(Reshape(target_shape=(feature_size,1,1,1)))

# upsampler =
# K.int_shape(upsampler)
# model.add(Reshape((3,1)))
# model.add(Flatten())
# model.add(UpSampling2D(size=(2,1)))
# model.add(Flatten())
# model.add(Reshape(target_shape=(feature_size*2,1)))
# model.add(Conv1D(filters=1,kernel_size=1, activation='relu',input_shape=(feature_size*2,1)))

# model.add(Reshape(target_shape=(samples,feature_size,1)))
# model.add(Flatten())
# model.add(Reshape((3,1,1)))
# model.add(Flatten())
# model.add(Reshape((1,3)))
# model.add(Flatten())
# model.add(Conv2D(filters=0, kernel_size=1, activation='relu',input_shape=(3,1,1)))
# model.add(UpSampling2D(size=(2,1)))
# model.add(UpSampling1D(size=2))
# model.add(Reshape((3,1)))
# model.add(Conv1D(filters=1, kernel_size=10 ,strides=10,
#                   input_shape=(None, 3),kernel_initializer= 'uniform',
#                   activation= 'relu'))
# # model.add(UpSampling1D(size=2))
# # model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
# # model.add(UpSampling1D(size=2))
# # model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
# model.add(UpSampling1D(size=2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))


model = keras_model_fc()
# for classifier in classifiers:
    # print(classifier)
classifier = classifiers[1]

name = classifier.__module__
print(f'{name}')
classifier.fit(X_indices_clean_scaled, y_values_clean_2d_scaled)
y_values_predict_scaled = classifier.predict(X_indices_scaled)

score = classifier.score(X_indices_scaled,y_ground_truth_scaled)
mse = metrics.mean_squared_error(y_ground_truth_scaled,y_values_predict_scaled)
r2 = metrics.r2_score(y_ground_truth_scaled,y_values_predict_scaled)
# classifier.score()
correlation,p_value = pearsonr(y_ground_truth_scaled.flatten(),y_values_predict_scaled.flatten())
print(f'Correlation: {correlation:.5f} | MSE:{mse:.5f} |  R2:{r2:.5f}  | Score:{score:.5f}')
# plt.scatter(y_ground_truth_scaled,y_values_predict_scaled)
# from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(classifier,
                        X_indices_clean_scaled,
                        y_values_clean_2d_scaled,
                        cv=kfold)

from sklearn import pipeline

estimators = []
estimators.append(('standardize', preprocessing.StandardScaler()))
estimators.append(('model', classifiers[1]))
pipeline = pipeline.Pipeline(estimators)

results = model_selection.cross_val_score(pipeline,
                        X_indices_clean,
                        y_values_clean_2d,
                        cv=kfold)

print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#%% Begin RL matrix deconvolvution
print("Build measurement matrix.")
#
# x0 = None
# Rtol = 1e-6
# NE_Rtol = 1e-6
# max_iter = 100
# sigmaSq = 0.0
# beta = 0.0

measurement_matrix_LO = scipy.sparse.linalg.aslinearoperator(measurement_matrix)
input_vector = astro_noisy_vector

# Raw RL, no imputation
FLAG_RAW = 0
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


#%% Matrix nuking

# H_df = pd.DataFrame(measurement_matrix)

#Remove all random sampled rows
beads = 100

rows_to_nuke = np.random.choice(np.arange(measurement_matrix.shape[0]),measurement_matrix.shape[0]-beads)
# rows_to_nuke=np.random.choice(np.arange(0,measurement_matrix.shape[0]), measurement_matrix.shape[0]-beads, replace=False);rows_to_nuke
# rows_to_nuke = (np.rint(np.random.choice(H.shape[0]-beads))*H.shape[0]).astype(int)
H_nuked = measurement_matrix.copy()

for i in rows_to_nuke:
    ones_image = np.ones_like(astro)
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    psf_ones = np.ones_like(static_psf);
    # psf_complex[psf_complex==1j] = np.nan
    delta_PSF = scipy.ndimage.convolve(delta_image,psf_ones)
    delta_PSF[delta_PSF!=0] = np.nan
    # plt.imshow(delta_PSF)
    H_nuked[i,:] = delta_PSF.flatten()
    # delta_PSF = psf_xy
    # plt.imshow(delta_image)
    # plt.show()

H_nuked_diag = np.diag(H_nuked)
H_nuked_diag.shape
nans_in_H = np.sum(np.isnan(H_nuked_diag))
nans_not_in_H = np.sum(~np.isnan(H_nuked_diag))

pd_nan = (nans_not_in_H-nans_in_H)/2*(nans_in_H+nans_not_in_H);
ratio = nans_in_H/H_nuked_diag.shape
print(f'Nans in H: {nans_in_H} | Ratio: {ratio}')

plt.imshow(H_nuked)
plt.imsave('./output/H_nuked.png', H_nuked)
# imp = SimpleImputer(missing_values=np.nan, strategy='mean',verbose=1)
####

image_width = np.sqrt(N_v);image_width
image_height = np.sqrt(N_v)
matrix_4d = [image_width,
                image_height,
                image_width,
                image_height]

measurement_matrix_4d = np.reshape(measurement_matrix,matrix_4d)
H_nuked_4d            = np.reshape(H_nuked,matrix_4d)

######
FLAT_IMPUTE = 0
if(FLAT_IMPUTE):
    from sklearn.linear_model import BayesianRidge
    #%% Matrix impute

    imp = IterativeImputer(missing_values=np.nan,verbose=2,estimator=BayesianRidge())
    imp.fit(H_nuked)
    H_fixed = imp.transform(H_nuked)

    error = measurement_matrix - H_fixed
    sum_error = np.sum(np.sum(error))
    sum_error
    # plt.show()
    plt.savefig("output/H_fixed.png")
    plt.imshow(H_fixed)
