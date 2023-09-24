import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert the images to float32 and normalize them
x_test = x_test.astype('float32') / 255.0

# Preprocess the labels
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Load the saved model architecture from the JSON file
model_architecture = open('squeeze_net_model.json', 'r').read()
sn = tf.keras.models.model_from_json(model_architecture)

# Load the learned weights from the .h5 file
sn.load_weights('image_classifier_squeeze_net.h5')

# Make predictions on the test dataset
predictions = sn.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)


print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

accuracy =accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

def plot_sample_predictions(test_images, true_labels, predicted_labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[i])
        plt.title("True: {} | Predicted: {}".format(true_labels[i], predicted_labels[i]))
        plt.axis("off")
    plt.show()

# Choose some samples from the test dataset
sample_indices = np.random.randint(0, len(x_test), size=9)
sample_images = x_test[sample_indices]
sample_true_labels = true_labels[sample_indices]
sample_predicted_labels = predicted_labels[sample_indices]

plot_sample_predictions(sample_images, sample_true_labels, sample_predicted_labels)
