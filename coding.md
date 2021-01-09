# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image


X=[]
Z=[]
IMG_SIZE=150
bayam='../input/traindata/train/bayam'
brokoli='../input/traindata/train/brokoli'
buncis='../input/traindata/train/buncis'
kangkung='../input/traindata/train/kangkung'
kubis='../input/traindata/train/kubis'
pare='../input/traindata/train/pare'
seledri='../input/traindata/train/seledri'
singkong='../input/traindata/train/singkong'
terong='../input/traindata/train/terong'
timun='../input/traindata/train/timun'
tomat='../input/traindata/train/tomat'
wortel='../input/traindata/train/wortel'






    def assign_label(img,vegetable_type):
    return vegetable_type
    
    
    
    
        for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,vegetable_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
        
        
        
        
        
        
        
make_train_data('Bayam',bayam)
print(len(X))

make_train_data('Brokoli',brokoli)
print(len(X))

make_train_data('Buncis',buncis)
print(len(X))

make_train_data('Kangkung',kangkung)
print(len(X))

make_train_data('Kubis',kubis)
print(len(X))

make_train_data('Pare',pare)
print(len(X))

make_train_data('Seledri',seledri)
print(len(X))

make_train_data('Singkong',singkong)
print(len(X))

make_train_data('Terong',terong)
print(len(X))

make_train_data('Timun',timun)
print(len(X))

make_train_data('Tomat',tomat)
print(len(X))

make_train_data('Wortel',wortel)
print(len(X))

fig,ax=plt.subplots(12,2)
fig.set_size_inches(15,15)
for i in range(12):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Vegetable: '+Z[l])
        
plt.tight_layout()



le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,12)
X=np.array(X)
X=X/255


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)





# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (12,12),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(12, activation = "softmax"))



batch_size=150
epochs=50

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)



datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)





model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])







History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
epochs = epochs, validation_data = (x_test,y_test),
verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
#model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))









model.save('/kaggle/working/modelsayur.h5')


model.summary()


new_model = tf.keras.models.load_model('/kaggle/working/modelsayur.h5')






new_model.summary()






plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()





plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()




# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)




# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break
        
        
        
        
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
      
        count+=1
        
        
        
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        plt.tight_layout()
        count+=1
        
        
        
        
        


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(12,2)
fig.set_size_inches(15,15)
for i in range (12):
    for j in range (0):
        ax[i,j].imshow(x_test[prop_class[count]])
        ax[i,j].set_title("Vegetable : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"bayam : "+str(le.inverse_transform(np.argmax([y_test[prop_class[count]]]))))
        plt.tight_layout()
        count+=1
        
        
        




# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('/kaggle/working/modelsayur.h5')

# Show the model architecture
new_model.summary()




new_model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))



# Create Models TFLite

!pip install tflite-model-maker




import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task.model_spec import mobilenet_v2_spec
from tensorflow_examples.lite.model_maker.core.task.model_spec import ImageModelSpec

import matplotlib.pyplot as plt




data = ImageClassifierDataLoader.from_folder('../input/model-sayur1/train')
train_data, test_data = data.split(0.8)



model = image_classifier.create(train_data)







loss, accuracy = model.evaluate(test_data)





model.export(export_dir='.', with_metadata=False)
