import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\flat_structure\train"
valid_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\flat_structure\valid"
test_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\flat_structure\test"


assert os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0, "Train directory is empty or does not exist."
assert os.path.exists(valid_dir) and len(os.listdir(valid_dir)) > 0, "Validation directory is empty or does not exist."
assert os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0, "Test directory is empty or does not exist."

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {valid_generator.samples}")
print(f"Testing samples: {test_generator.samples}")

num_classes = len(train_generator.class_indices)

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)

predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=valid_generator, epochs=10)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

model.save('hyperkvasir_model.h5')
