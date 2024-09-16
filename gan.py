import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt

# Set constants
BATCH_SIZE = 256
IMG_HEIGHT = 64
IMG_WIDTH = 64
EPOCHS = 50
BUFFER_SIZE = 60000
NOISE_DIM = 100  # Dimension of the noise vector

# Path to your local dataset
dataset_path = '/home/sudharsan/Dataset/AppleLeaf/test/Apple___Black_rot'

# Load and preprocess the dataset
def load_and_preprocess_image(file_path):
    # Load the image, resize it and normalize to [-1, 1]
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])  # Resize to 64x64



    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

def load_dataset(dataset_path):
    # List all image files in the dataset directory
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
    
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    
    # Map each file path to the loaded and preprocessed image
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

# Build the generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return model

# Build the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_HEIGHT, IMG_WIDTH, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

# Define loss functions for generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Initialize models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define a function for generating and saving images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        # Convert the tensor to numpy and then use astype
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, :].numpy() * 127.5 + 127.5).astype(np.uint8))
        plt.axis('off')
    
    plt.savefig('gan_img/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training loop
def train(dataset, epochs):
    seed = tf.random.normal([16, NOISE_DIM])
    
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        
        # Produce images after each epoch
        generate_and_save_images(generator, epoch + 1, seed)
    
        print(f'Epoch {epoch + 1} completed')

# Load the dataset from your storage
train_dataset = load_dataset(dataset_path)

# Train the GAN model
train(train_dataset, EPOCHS)

