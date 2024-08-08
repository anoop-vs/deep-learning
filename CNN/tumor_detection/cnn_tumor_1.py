import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib


train_dir = pathlib.Path('artifacts/cleanedbrainimage/train')
test_dir = pathlib.Path('artifacts/cleanedbrainimage/test')
batch_size = 64
img_height = 180
img_width = 180
AUTOTUNE = tf.data.AUTOTUNE

print("--------------------------------------\n")
print("Data Loading Process\n")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=None,
    seed=101,
    image_size=(img_height, img_width),
    batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  validation_split=None,
  seed=101,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)


train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# data_augmentation = tf.keras.Sequential(
#   [
#     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_width,img_height,3)),
#     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
#     tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
#   ]
# )

model = tf.keras.Sequential([
  # data_augmentation, #Add if augmentation is needed.
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(num_classes),
])


print("--------------------------------------\n")
# model.summary()
print("--------------------------------------\n")

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(train_ds,validation_data=val_ds,epochs=10)
print("Training Finished.\n")
print("--------------------------------------\n")


# Plot and save accuracy
plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('CNN/tumor_detection/results/tumor_accuracy_plot.png')

# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('CNN/tumor_detection/results/tumor_loss_plot.png')


