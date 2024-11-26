import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the saved model
model_path = 'incremental_cnn_model.h5'
model = load_model(model_path)

# Print model summary
model.summary()

# Fine-tuning function for incremental learning
def incremental_fine_tune(model, data_generator, steps_per_epoch, epochs=5):
    # Unfreeze the base model for fine-tuning (optional)
    model.layers[0].trainable = True  # Unfreeze the base model

    # Compile with a low learning rate for incremental updates
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    return model

# Load new data from a local folder
def load_data_from_folder(data_dir, img_size=(224, 224), batch_size=32):
    """
    This function uses ImageDataGenerator to load images from a specified directory.
    The directory structure should be as follows:
    data_dir/
    ├── class1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── class2/
    │   ├── img3.jpg
    └── ...
    """
    datagen = ImageDataGenerator(rescale=1./255)

    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return data_generator

# Main
if __name__ == "__main__":
    # Define path to your local dataset folder
    data_dir = 'path/to/your/dataset/folder'  # e.g., './new_dataset'
    batch_size = 32
    epochs = 5

    # Load data from the local folder
    data_generator = load_data_from_folder(data_dir, batch_size=batch_size)
    steps_per_epoch = data_generator.samples // batch_size

    # Retrain the loaded model incrementally
    print("\nRetraining the loaded model with new data from local folder...")
    model = incremental_fine_tune(model, data_generator, steps_per_epoch, epochs=epochs)

    # Save the updated model
    model.save('updated_incremental_model.h5')
    print("Updated model saved successfully!")
