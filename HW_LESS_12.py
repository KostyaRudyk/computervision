import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(__file__)#вказуємо щро цей файл є головнимм

TRAIN_PATH = os.path.join(BASE_DIR, "data", "train")
TEST_PATH = os.path.join(BASE_DIR, "data", "test")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH, image_size=(128, 128), batch_size = 20,
    label_mode = "categorical"
)#

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH, image_size=(128, 128), batch_size = 32,
    label_mode = "categorical"
)

model = models.Sequential()

model.add(layers.Rescaling(1./255, input_shape=(128, 128, 3)))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))#зменшує картинку прибиаючи неважливі елементи тобто шум залишаючи тільки важливе
#це все перший шар

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))#зменшує картинку прибиаючи неважливі елементи тобто шум залишаючи тільки важливе


model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))#зменшує картинку прибиаючи

model.add(layers.Flatten())# перетворює картинку в числа

model.add(layers.Dense(64, activation='relu'))#шось там перетворює
model.add(layers.Dense(6, activation='softmax'))

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)
#розписуємо як навчаємо в скільки епох і тд

model.fit(train_ds,epochs=20, validation_data = test_ds)

test_photo = os.path.join(BASE_DIR, 'image', "test.jpg")

if os.path.exists(test_photo):
    img = image.load_img(test_photo, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    class_name = sorted(os.listdir(TRAIN_PATH))

    result_ind = np.argmax(predictions[0])

    print(f'Результат; {class_name[result_ind]}')