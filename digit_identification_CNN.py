#importing packages
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

#reading the file containing labels.
df_train = pd.read_csv('train.csv')
#assigning labels to y
y = df_train.iloc[:,1].values
#creating one hot encoded labels
y = to_categorical(y, num_classes = 10)

df_test = pd.read_csv('test.csv')
#function for loading images into a numpy array
def load_data(path , filenames):
    data = []
    for file in filenames:
        img = cv2.imread(path+'/'+file , 0)
        data.append(img)
        
    data = np.array(data)
    data = data/255.0
    data = np.reshape(data , (-1,28,28,1))
    return data

path_train = 'F:/ML/Analytics Vidhya/Digits/data/train'  #complete path for labeled data
path_test = 'F:/ML/Analytics Vidhya/Digits/data/test'    #complete path for unlabeled data

train = load_data(path_train,df_train['filename'])   # loading images for labeled data
test = load_data(path_test,df_test['filename'])  #loading images for unlabeled data

#splitting the labeled images into training and testing data to check accuracy
X_train,X_test,y_train,y_test = train_test_split(train,y,test_size = 0.2 ,random_state = 0)

#visualising some loaded images
plt.imshow(train[5][:,:,0] , cmap = 'gray') #training data

plt.imshow(test[2][:,:,0] , cmap = 'gray')  #testing_data

#defining the convolutional neural network model
classifier = Sequential()
classifier.add(Conv2D(64, (3, 3),input_shape = (28,28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 256, activation = 'relu' , input_dim = 256))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 256, activation = 'relu' , input_dim = 256))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='acc', patience=3, verbose=1,factor=0.5, 
                                            min_lr=0.00001)
callbacks = [EarlyStopping(monitor="loss", min_delta=0 , patience=3 , verbose=0 , mode='auto') ,
             learning_rate_reduction]

print(classifier.summary())  #printing the structure of the CNN model

#fitting the CNN model on the data
classifier.fit(X_train , y_train , batch_size = 128 , epochs = 20 , callbacks = callbacks)

#computing loss and accuracy
loss, acc = classifier.evaluate(X_test, y_test, verbose=0)
#predicitng the labels of labeled testing data
pred = classifier.predict_classes(X_test)                 

#checking the predictions by plotting the image
num_check = 6  #index number to check
plt.imshow(X_test[num_check][:,:,0] , cmap = 'gray')
plt.title('Predicted Value for this digit : {}'.format(pred[num_check]))

#predicting the labels of unlabeled data
y_pred = classifier.predict_classes(test)

# checking predictions on unlabeled data
num_check = 6
plt.imshow(test[num_check][:,:,0] , cmap = 'gray')
plt.title('Predicted Value for this digit : {}'.format(y_pred[num_check]))



    