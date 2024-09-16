import tensorflow as tf

# Set memory growth to avoid allocating all GPU memory (if you want to allow GPU usage, otherwise skip this step)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the CPU
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError as e:
        print(e)

# Now, load and use your model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('/home/sudharsan/projects/sih_model/model/other_models/resnet.h5')
#model = tf.keras.models.load_model('/home/sudharsan/projects/sih_v2/summa/model.h5')
# Define class names (replace with your own class names)
#class_names=['cercospora','rust','healthy','leaf_blight']
class_names = ['apple_scab', 'apple_blackrot', 'apple_Cedar_rust', 'apple_healthy',
               'potato_early_blight', 'potato_healthy', 'potato_late_blight']

def img_classification(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict using the loaded model
    predictions = model.predict(img_array)

    # Get the maximum softmax probability and its corresponding class
    max_confidence = np.max(predictions, axis=1)[0]
    predicted_index = np.argmax(predictions, axis=1)[0]

    # Convert the TensorFlow/Numpy float32 to a standard Python float
    max_confidence = float(max_confidence)

    # Check against the confidence threshold
    
    predicted_class = class_names[predicted_index]
    print(f'Confidence score: {max_confidence:.2f}')
    print(f'Predicted class label: {predicted_class}')

    return predicted_class

if __name__ == "__main__":
    img_path = "00120a18-ff90-46e4-92fb-2b7a10345bd3___RS_GLSp 9357.JPG"  # Replace with the actual image path
    class_name = img_classification(img_path)
    print(f'Class: {class_name}')
