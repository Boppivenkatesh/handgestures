import tensorflow as tf
import numpy as np
import cv2
model = tf.keras.models.load_model('model.h5')
def preprocess_image(image_path):
  image = cv2.imread(image_path)
  image = cv2.resize(image, (224, 224))
  image = image.astype('float32') / 255.0
  image = np.expand_dims(image, axis=0)
  return image
def predict_image_class(image_path):
  image = preprocess_image(image_path)
  predictions = model.predict(image)
  class_index = np.argmax(predictions)
  return class_index
image_path = 'test_image.jpg'
class_index = predict_image_class(image_path)
print('The predicted class index is:', class_index)