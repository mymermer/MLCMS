import tensorflow as tf
from tensorflow import keras
from keras import layers

def VAE(data_shape, latent_dim=2):
    """
    Variational Autoencoder (VAE) model.

    Args:
        data_shape (tuple): The shape of the input data.
        latent_dim (int, optional): The dimension of the latent space. Defaults to 2.

    Returns:
        tuple: A tuple containing the VAE model, encoder model, and decoder model.
    """
    # Encoder
    inputs = tf.keras.Input(shape=data_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Sampling function for the latent space
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs_mean = layers.Dense(data_shape, activation='sigmoid')(x)

    # Implement standard deviation as a trainable variable of the model
    decoder_stddev = tf.Variable(tf.ones(shape=(data_shape,), dtype=tf.float32), name='decoder_stddev')
    decoded_with_stddev = layers.Lambda(lambda x: x[0] * x[1])([outputs_mean, decoder_stddev])
    decoder = tf.keras.Model(latent_inputs, decoded_with_stddev, name='decoder')

    # VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = tf.keras.Model(inputs, outputs, name='vae')

    # Define the loss function
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= data_shape
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizer)

    return vae, encoder, decoder
