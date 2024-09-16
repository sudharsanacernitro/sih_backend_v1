import tensorflow as tf

# Set memory growth to avoid allocating all GPU memory (if you want to allow GPU usage, otherwise skip this step)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the CPU
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError as e:
        print(e)

from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model in .keras format
model = tf.keras.models.load_model('/home/sudharsan/projects/sih_model/model/other_models/fruit.keras')

# Define class names (replace with your own class names)
class_names = ['apple_blotch', 'apple_healthy', 'apple_blackrot','apple_scab']  # Update this list with your actual class names

# Path to an image for prediction
def img_classification(img_path):
    # Load and preprocess the image
    # Resize the image to the expected size for InceptionV3: 299x299
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict using the loaded model
    predictions = model.predict(img_array)
    max_confidence = np.max(predictions, axis=1)[0]

    # Get the predicted class index
    predicted_index = np.argmax(predictions, axis=1)[0]

    confidence_score = predictions[0][predicted_index]

    print(f'Confidence score: {confidence_score:.2f}')
    
        # Map the predicted index to the class label
    predicted_class = class_names[predicted_index]

    print(f'Predicted class index: {predicted_index}')
    print(f'Predicted class label: {predicted_class}')

    return predicted_class

if __name__ == "__main__":
    img_path = '/home/sudharsan/projects/sih_model/model/10.jpg'  # Replace with the actual image path
    class_name = img_classification(img_path)
    print(class_name)
