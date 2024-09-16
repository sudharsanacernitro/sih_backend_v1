import numpy as np
import tensorflow as tf
from scipy.stats import mode
import matplotlib.pyplot as plt

# Load pre-trained models
def ensem_predict(img_type, crop_name, file_path):
    model1 = tf.keras.models.load_model(f'/home/sudharsan/projects/models/inception/{img_type}/{crop_name}.h5')  # Inception-based
    model2 = tf.keras.models.load_model(f'/home/sudharsan/projects/models/mobilenet/{img_type}/{crop_name}.h5')  # MobileNet-based
    model3 = tf.keras.models.load_model(f'/home/sudharsan/projects/models/resnet/{img_type}/{crop_name}.h5')  # ResNet-based

    # Define the labels
    ensem_label = {
        'apple-leaf': ['apple_scab', 'apple_blackrot', 'apple_Cedar_rust', 'apple_healthy'],
        'apple-fruit': ['apple_blotch', 'apple_healthy', 'apple_blackrot', 'apple_scab'],
        'potato-fruit': ['Blackspot_Bruising', 'Healthy_Potato', 'Potato_Brown_Rot', 'Potato_Dry_Rot', 'Potato_Soft_Rot'],
        'potato-leaf': ['potato_early_blight','potato_late_blight','potato_healthy' ],
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
        for model in models:
            if model == model1:
                pred = model.predict(preprocessed_images['inception'])
                individual_predictions['Inception'] = pred
            elif model == model2:
                pred = model.predict(preprocessed_images['mobilenet'])
                individual_predictions['MobileNet'] = pred
            elif model == model3:
                pred = model.predict(preprocessed_images['resnet'])
                individual_predictions['ResNet'] = pred
            predictions.append(pred)

        # Convert predictions to class labels
        predictions = np.array(predictions)
        predicted_classes = np.argmax(predictions, axis=-1)

        # Ensemble prediction by majority vote
        ensemble_preds, _ = mode(predicted_classes, axis=0)

        # Map to label names
        ensemble_label = labels[ensemble_preds[0]]
        individual_labels = {model_name: labels[np.argmax(pred)] for model_name, pred in individual_predictions.items()}

        return ensemble_label, individual_labels, individual_predictions

    def plot_predictions(individual_predictions, ensemble_label):
        """Plot individual model predictions and the ensemble."""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = ['Inception', 'MobileNet', 'ResNet', 'Ensemble']
        num_classes = len(labels)
        
        # Individual model predictions
        for model_name, pred in individual_predictions.items():
            probabilities = pred[0]
            ax.plot(labels, probabilities, label=model_name)

        #Add Ensemble prediction (mode or averaged probabilities)
        ax.axhline(y=1/num_classes, color='gray', linestyle='--', label='Ensemble Majority Vote')
        ax.set_title(f'Predictions for Each Model and Ensemble\n(Ensemble Prediction: {ensemble_label})')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Prediction Probability')
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    models = [model1, model2, model3]
    image_path = file_path
    ensemble_label, individual_labels, individual_predictions = ensemble_predictions(models, image_path)

    # Output the results
    print(f'Ensemble prediction: {ensemble_label}')

    print('Individual model predictions:')
    for model_name, pred_label in individual_labels.items():
        print(f'{model_name}: {pred_label}')

    # Plot the predictions
    plot_predictions(individual_predictions, ensemble_label)

    return ensemble_label

if __name__ == "__main__":
    ensem_predict('fruit', 'apple', '/home/sudharsan/projects/sih_model/model/2a9ne8h.jpg')
