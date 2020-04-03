pip install keras

from keras.datasets import cifar10
(x_train,y_train), (x_test,y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

import matplotlib.pyplot as mt
img = mt.imshow(x_train[50])
print('the label of the image is:',y_train[50])

from keras.utils import to_categorical 
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print(y_train_one_hot) 

print('the new label is',y_train_one_hot[2])

x_train = x_train/255
y_train = y_train/255


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
model = Sequential()
model.add( Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)) )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add( Flatten() )
model.add( Dense(1000, activation='relu'))
model.add( Dense(10, activation ='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=14, validation_split=0.3 )

model.evaluate(x_test, y_test_one_hot)[1]

mt.plot(hist.history['acc'])
mt.plot(hist.history['val_acc'])
mt.title('Model accuracy')
mt.ylabel('Accuracy')
mt.xlabel('Epoch')
mt.legend(['Train', 'Val'], loc='upper left')
mt.show()


mt.plot(hist.history['loss'])
mt.plot(hist.history['val_loss'])
mt.title('Model loss')
mt.ylabel('Loss')
mt.xlabel('Epoch')
mt.legend(['Train', 'Val'], loc='upper right')
mt.show()


#Load the data
from google.colab import files # Use to load data on Google Colab
uploaded = files.upload() # Use to load data on Google Colab
my_image = mt.imread("truck.jfif") #Read in the image (3, 14, 20)

img = mt.imshow(my_image)

from skimage.transform import resize
my_image_resized = resize(my_image, (32,32,3)) #resize the image to 32x32 pixel with depth = 3
img = mt.imshow(my_image_resized) #show new image

import numpy as np
probabilities = model.predict(np.array( [my_image_resized] ))

array([[7.9247387e-05, 7.2782481e-05, 1.4453316e-03, 1.5812383e-03,
        3.3377426e-05, 2.3285209e-03, 4.5265850e-02, 6.3021121e-06,
        1.7083034e-05, 9.4917023e-01]], dtype=float32)
        
number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])
print("Sixth most likely class:", number_to_class[index[4]], "-- Probability:", probabilities[0,index[4]])
print("Seventh most likely class:", number_to_class[index[3]], "-- Probability:", probabilities[0,index[3]])
print("Eighth most likely class:", number_to_class[index[2]], "-- Probability:", probabilities[0,index[2]])
print("Ninth most likely class:", number_to_class[index[1]], "-- Probability:", probabilities[0,index[1]])
print("Tenth most likely class:", number_to_class[index[0]], "-- Probability:", probabilities[0,index[0]])
        
