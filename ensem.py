import numpy as np
import tensorflow as tf
from scipy.stats import mode

# Load pre-trained models
def ensem_predict(img_type, crop_name, file_path):
    model1 = tf.keras.models.load_model(f'/home/sudharsan/projects/models/inception/{img_type}/{crop_name}.h5')  # Inception-based
    model2 = tf.keras.models.load_model(f'/home/sudharsan/projects/models/mobilenet/{img_type}/{crop_name}.h5')  # MobileNet-based
    model3 = tf.keras.models.load_model(f'/home/sudharsan/projects/models/resnet/{img_type}/{crop_name}.h5')  # ResNet-based
    print("Model Loaded successfully")
    # Define the labels
    ensem_label = {
        'apple-leaf': ['apple_scab', 'apple_blackrot', 'apple_Cedar_rust', 'apple_healthy'],
        'apple-fruit': ['apple_blotch', 'apple_healthy', 'apple_blackrot', 'apple_scab'],
        'potato-fruit': ['Blackspot_Bruising', 'Healthy_Potato', 'Potato_Brown_Rot', 'Potato_Dry_Rot', 'Potato_Soft_Rot'],
        'potato-leaf': ['potato_early_blight', 'potato_late_blight', 'potato_healthy'],
        'corn-leaf': ['gray_leaf_spot', 'common_rust', 'northern_leaf_blight', 'corn_healthy'],
        'grape-leaf': ['Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)', 'grape_healthy'],
        'paddy-leaf': ['Bacterial_leaf_blight', 'Brown_spot', 'Leaf_smut'],
    }
    labels = ensem_label[crop_name + "-" + img_type]

    def preprocess_image(image_path, target_size):
        """Preprocess image for model prediction."""
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0  # Normalize
        return image_array

    def ensemble_predictions(models, image_path):
        """Get ensemble predictions from multiple models."""
        # Define target sizes for each model
        model_input_sizes = {
            'inception': (299, 299),  # Inception model typically expects 299x299
            'mobilenet': (224, 224),  # MobileNet model typically expects 224x224
            'resnet': (224, 224),     # ResNet model also typically expects 224x224
        }

        # Preprocess image for each model
        preprocessed_images = {
            'inception': preprocess_image(image_path, model_input_sizes['inception']),
            'mobilenet': preprocess_image(image_path, model_input_sizes['mobilenet']),
            'resnet': preprocess_image(image_path, model_input_sizes['resnet'])
        }

        # Get predictions from each model
        predictions = []
        individual_predictions = {}
        for model, model_name in zip(models, ['Inception', 'MobileNet', 'ResNet']):
            pred = model.predict(preprocessed_images[model_name.lower()])
            confidence_score = np.max(pred)  # Get the confidence score (highest probability)
            predicted_label_index = np.argmax(pred)  # Get the predicted label index
            predicted_label = labels[predicted_label_index]
            individual_predictions[model_name] = [str(confidence_score), predicted_label]
            predictions.append(pred)

        # Convert predictions to class labels
        predictions = np.array(predictions)
        predicted_classes = np.argmax(predictions, axis=-1)

        # Ensemble prediction by majority vote
        ensemble_preds, _ = mode(predicted_classes, axis=0)
        ensemble_label = labels[ensemble_preds[0]]
        individual_predictions['ensem']=ensemble_label

        return individual_predictions

    models = [model1, model2, model3]
    image_path = file_path
    predictions = ensemble_predictions(models, image_path)

    # Output the results
    print(f'Ensemble prediction: {predictions['ensem']}')

    print('Individual model predictions with confidence scores:')

    print(f"{predictions}")

    return predictions

if __name__ == "__main__":
    ensem_predict('fruit', 'apple', '/home/sudharsan/projects/sih_model/model/2a9ne8h.jpg')
