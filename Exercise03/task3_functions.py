import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# Load and preprocess MNIST data
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Load MNIST data
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    # Reshape images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    return (x_train, y_train), (x_test, y_test)


data_shape = 784 # shape of the data (28x28=784)
latent_dim = 2 # dimension of the latent space : to be changed for the last question of task 3
batch_size = 128 # batch size for training

# Create VAE
def VAE(data_shape, latent_dim) : 

    inputs = tf.keras.Input(shape=data_shape)

        # Encoder
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Mean and standard deviation outputs for the approximate posterior qφ(z|x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x) # or tf.keras.layers.Activation('softplus')(z_log_var) -> positivity insurance

    def sampling(args):
        """
        Sampling from the latent multivariate diagonal standard normal distribution as prior p(z).
        Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
            Also referred to as "Reparameterization trick".
        """
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim)) #shape=(batch_size, latent_dim)=z_mean.shape
        return z_mean + tf.exp(0.5*z_log_var) * epsilon
    
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = tf.keras.layers.Dense(256, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs_mean = tf.keras.layers.Dense(data_shape, activation='sigmoid')(x) #Outputs only the mean of likelihood pθ(x|z)
    # Implement standard deviation as a trainable variable of the model
    decoder_stddev = tf.Variable(tf.ones(shape=(data_shape,), dtype=tf.float32), name='decoder_stddev') 
    decoded_with_stddev = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([outputs_mean, decoder_stddev])
    decoder = tf.keras.Model(latent_inputs, decoded_with_stddev, name='decoder')

    outputs = decoder(encoder(inputs)[2]) 
        # Define the loss function
    # Reconstruction loss : binary cross-entropy between the input and output
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= data_shape
    reconstruction_loss += 0.5 * tf.reduce_sum(tf.square(outputs - inputs) / tf.square(decoder_stddev) + tf.math.log(tf.square(decoder_stddev))) #include the standard deviation term 
    # KL divergence loss : KL divergence between the approximate posterior qφ(z|x) and the prior p(z)
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # Total loss : sum of the reconstruction loss and the KL divergence loss
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    
            # Compile the model
    vae = tf.keras.Model(inputs, outputs, name='vae')
    vae.add_loss(vae_loss)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    vae.compile(optimizer=optimizer)
    return vae, encoder, decoder

