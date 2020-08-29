import numpy as np
from skimage import transform

from keras.models import load_model
from keras.utils import to_categorical

# parameters
num_classes = 10
num_channels = 1  
batch_size = 64
transformed_w = 28
transformed_h = 28

def test_model():

    test_data = np.load('X_train.npy') #change to test data path
    test_labels = np.load('y_train.npy') # change to test labels path

    original_h, original_w = test_data.shape[1], test_data.shape[2]
    #reshaping to add 1 channel
    test_data = np.reshape(test_data,(test_data.shape[0], original_h, original_w, num_channels))

    #reshaping data from 300x300 to 28x28
    test_data = test_data.transpose((1, 2, 3, 0)) 
    test_data = transform.resize(test_data.reshape(original_h, original_w, -1), (transformed_h, transformed_w)) 
    test_data = test_data.reshape(transformed_h, transformed_w, 1, -1)
    test_data = test_data.transpose((3, 0, 1, 2))

    # one-hot-encode the labels
    test_labels_cat = to_categorical(test_labels,num_classes)

    #load model
    model = load_model('trained_model.hdf5')

    #getting test loss and accuracy 
    test_loss, test_accuracy = model.evaluate(test_data, test_labels_cat, batch_size=batch_size)
    print('Test loss: %.3f Test accuracy: %.3f' % (test_loss, test_accuracy))

    # predicting 
    test_pred = model.predict(test_data)
    y_predicted = np.argmax(test_pred, axis=1)
    y_true = np.argmax(test_labels_cat,axis=1)

    return(y_predicted)
    
test_model()