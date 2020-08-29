import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage import transform
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from keras.callbacks.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, to_categorical

#parameters
num_channels = 1
num_classes = 10
num_epochs = 600
batch_size = 64
transformed_w = 28
transformed_h = 28

# load images
Xload = np.load('X_train.npy')
Yload =np.load('y_train.npy')


#spliting dataset into train and validation
train_digits, val_digits, train_labels, val_labels = train_val_split(Xload, Yload, test_size=0.2, shuffle=True)

#reshaping to add 1 channel
original_h, original_w = train_digits.shape[1], train_digits.shape[2]
train_data = np.reshape(train_digits,(train_digits.shape[0], original_h, original_w, num_channels))
val_data = np.reshape(val_digits,(val_digits.shape[0],original_h, original_w, num_channels))

#reshaping data from 300x300 to 28x28
train_data = train_data.transpose((1, 2, 3, 0))
train_data = transform.resize(train_data.reshape(original_h, original_w, -1), (transformed_h, transformed_w))
train_data = train_data.reshape(transformed_h, transformed_w, 1, -1)
train_data = train_data.transpose((3, 0, 1, 2))

#reshaping data from 300x300 to 28x28
val_data = val_data.transpose((1, 2, 3, 0))
val_data = transform.resize(val_data.reshape(original_h, original_w, -1), (transformed_h, transformed_w))
val_data = val_data.reshape(transformed_h, transformed_w, 1, -1)
val_data = val_data.transpose((3, 0, 1, 2))

# one-hot-encode the labels
train_labels_cat = to_categorical(train_labels,num_classes)
val_labels_cat = to_categorical(val_labels,num_classes)

def build_model():
    model = Sequential()
    # adding Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same',
                     input_shape=(28, 28,num_channels)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=100, kernel_size=(3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    # Densely connected layers
    model.add(Dense(128, activation='relu'))

    # output layer
    model.add(Dense(num_classes, activation='softmax'))

    # compiling with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#load model
model = build_model()

#using data aumentagtion to generate more data
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range = 1.2)

#saving the model with best validation accuracy
checkpointer = ModelCheckpoint(filepath='trained_model.hdf5', verbose=1, save_best_only=True, monitor='val_accuracy' )

# fitting the model on batches with real-time data augmentation:
history = model.fit_generator(datagen.flow(train_data, train_labels_cat, batch_size=batch_size),
                    steps_per_epoch=len(train_data) / batch_size,
                    epochs=num_epochs,
                    validation_data=(val_data,val_labels_cat),
                    callbacks=[checkpointer])

val_loss, val_accuracy = model.evaluate(val_data, val_labels_cat, batch_size=batch_size)
print('Val loss: %.3f accuracy: %.3f' % (val_loss, val_accuracy))

#plotting confusion matrix
val_pred = model.predict(val_data)
y_predicted = np.argmax(val_pred, axis=1)
y_true = np.argmax(val_labels_cat,axis=1)

cm = confusion_matrix(y_true, y_predicted)
plt.figure(figsize=(9,7))
plt.title('Confusion Matrix')
plt.ylabel('Predicted')
plt.xlabel('True')
sns.heatmap(cm,fmt="", annot=True, cmap='Blues')
plt.savefig('cm.png')

#saving model image
plot_model(model, to_file='model.png')
