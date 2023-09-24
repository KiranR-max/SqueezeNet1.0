import numpy as np
from PIL import Image
import tensorflow as tf
# from tf.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.python.keras.layers import Layer,Lambda, Conv2D, MaxPooling2D, Activation, GlobalAveragePooling2D

from tensorflow.python.keras.models import Model

# import numpy as np
# from PIL import Image
# import tensorflow as tf

# ... (The rest of your code remains unchanged)
class FireModule(Layer):
    def __init__(self, s1x1, e1x1, e3x3, **kwargs):
        super(FireModule, self).__init__(**kwargs)
        self.s1x1 = s1x1
        self.e1x1 = e1x1
        self.e3x3 = e3x3

    def build(self, input_shape):
        self.squeeze_conv = Conv2D(self.s1x1, (1, 1), padding='valid', activation='relu')
        self.expand1x1_conv = Conv2D(self.e1x1, (1, 1), padding='valid', activation='relu')
        self.expand3x3_conv = Conv2D(self.e3x3, (3, 3), padding='same', activation='relu')

    def call(self, inputs):
        x = self.squeeze_conv(inputs)
        expand1x1 = self.expand1x1_conv(x)
        expand3x3 = self.expand3x3_conv(x)
        return tf.keras.layers.concatenate([expand1x1, expand3x3], axis=-1)

def SqueezeNet(input_shape=(227, 227, 3), num_classes=1000):
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # SqueezeNet Fire modules
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = FireModule(16, 64, 64)(x)
    x = FireModule(16, 64, 64)(x)
    x = FireModule(32, 128, 128)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = FireModule(32, 128, 128)(x)
    x = FireModule(48, 192, 192)(x)
    x = FireModule(48, 192, 192)(x)
    x = FireModule(64, 256, 256)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = FireModule(64, 256, 256)(x)

    # Final layers
    x = Conv2D(num_classes, (1, 1), padding='valid', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x, name='squeezenet')
    return model


def fire_module(x, s1x1, e1x1, e3x3):
    squeeze = Conv2D(s1x1, (1, 1), padding='valid', activation='relu')(x)

    expand1x1 = Conv2D(e1x1, (1, 1), padding='valid', activation='relu')(squeeze)
    expand3x3 = Conv2D(e3x3, (3, 3), padding='same', activation='relu')(squeeze)

    x = tf.keras.layers.concatenate([expand1x1, expand3x3], axis=-1)
    return x

# Load the custom SqueezeNet model with pre-trained weights
model = SqueezeNet()

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((227, 227))  # SqueezeNet takes 227x227 images as input
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)
    return image

# Perform the classification
def classify_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    return predictions

# Replace 'path/to/your/image.jpg' with the path to your image
image_path = 'bird.jpg'
predictions = classify_image(image_path)

# Load ImageNet labels (1000 classes)
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_PATH = "imagenet-simple-labels.json"
import requests
import json

response = requests.get(LABELS_URL)
with open(LABELS_PATH, 'wb') as f:
    f.write(response.content)

with open(LABELS_PATH) as f:
    labels = json.load(f)

# Get the top predicted label
top_prediction_idx = np.argmax(predictions)
predicted_label = labels[top_prediction_idx]

print(f"Predicted class: {predicted_label}")

