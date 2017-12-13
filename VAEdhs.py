# A experiment of running VAE
# @author Zhaozhuo Xu

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cross_validation import train_test_split
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

############################################################

batch_size = 100
original_dim = 4097
latent_dim = 2
intermediate_dim = 256
epochs =500
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)


# train the VAE on MNIST digits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


import scipy.io as spio
mat= spio.loadmat('all_countries_dhs.mat', squeeze_me=True)
NGdhs=mat['dhs']
# y_train=NGdhs[:,3];
# y_test=NGdhs[:,3];
survey_X=NGdhs[:,0:2];
featureX=NGdhs[:,4:4099];
# All_feature=(survey_X,featureX)
All_feature = np.concatenate((survey_X,featureX),axis=1)
All_y=NGdhs[:,3]

print len(All_feature), len(All_y)
# All_feature=All_feature.reshape((len(All_feature), np.prod(All_feature.shape[1:])))
# x_train=np.column_stack((survey_X,featureX));
# x_test=np.column_stack((survey_X,featureX));
x_train, x_test, y_train, y_test = train_test_split(All_feature, All_y,test_size=0.33, random_state=42)
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0],x_test_encoded[:, 1],  c=y_test)
plt.colorbar()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_test_encoded, y_test)

# Make predictions using the testing set
dhs_y_pred = regr.predict(x_test_encoded)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, dhs_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, dhs_y_pred))


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
dhs_y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, dhs_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, dhs_y_pred))
# # Plot outputs
# plt.scatter(dhs_y_pred,y_test,  color='black')
# plt.plot(dhs_y_pred, y_test, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()


# from sklearn import decomposition
# pca = decomposition.PCA(n_components=2)
# pca.fit(x_test_encoded)
# X = pca.transform(x_test_encoded)
# plt.figure(figsize=(6, 6))
# plt.scatter(X[:, 0],X[:, 1],  c=y_test)
# plt.colorbar()
# plt.show()
# # build a digit generator that can sample from the learned distribution
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)
#
# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit
#
# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()