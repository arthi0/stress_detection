from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
import os
import pandas as pd
import numpy as np
import pickle
from keras_preprocessing.image import load_img
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import matplotlib.pyplot as plt

Train_dir = r'C:\Users\Arthi\Downloads\archive (4)\train'
Test_dir=r'C:\Users\Arthi\Downloads\archive (4)\test'
def createdataframe(dir):
    image_path=[]
    labels=[]
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_path.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label,"completed")
    return image_path,labels
train=pd.DataFrame()
train['image'],train['label']=createdataframe(Train_dir)
test=pd.DataFrame()
test['image'],test['label']=createdataframe(Test_dir)
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')  # Specify color_mode='grayscale' for loading images in grayscale
        img = np.array(img)
        features.append(img)
    return features
train_features = extract_features(train['image'])
test_features = extract_features(test['image'])
train_features_array = np.array(train_features)

# Perform division by 255.0
x_train = train_features_array / 255.0
test_features_array = np.array(test_features)

# Perform division by 255.0
x_test = test_features_array / 255.0
le=LabelEncoder()
log=le.fit(train['label'])
pickle.dump(log,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
y_train=le.transform(train['label'])
y_test=le.transform(test['label'])
y_train=to_categorical(y_train,num_classes=7)
y_test=to_categorical(y_test,num_classes=7)
model=Sequential()
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_json=model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")
json_file=open("D:\gc\emotiondetector.json",'r')
model_json=json_file.read()
json_file.close()
model=model_from_json(model_json)
model.load_weights("D:\gc\emotiondetector.h5")
label=['angry','disgust','fear','happy','neutral','sad','surprise']
def ef(image):
    img = load_img(image, color_mode='grayscale')  # Use color_mode='grayscale' to load the image in grayscale mode
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature

image = r'C:\Users\Arthi\Downloads\archive (4)\train\angry\Training_143373.jpg'
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
image = r'C:\Users\Arthi\Downloads\archive (4)\train\angry\Training_143373.jpg'
img = ef(image) 
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')

