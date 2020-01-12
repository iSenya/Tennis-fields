import numpy as np
import os, time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#load data
data_path = r"/home/kseniia/Pictures/"
data_dir_list = os.listdir(data_path)
img_data_list = []
for dataset in data_dir_list:
    img_list = np.random.choice(os.listdir(data_path + '/' + dataset), 2000)
    print('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for i in img_list:
        img_path = data_path +'/'+ dataset + '/' + i
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_data_list.append(x)

img_data = np.array(img_data_list)
print (img_data.shape)
img_data = np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

#define classes
num_classes = 2
num_of_samples = img_data.shape[0] 
labels = np.ones((num_of_samples,), dtype = 'int64')

labels[0:2000] = 0
labels[2000:4000]= 1
names = ['tennis court','other']

#data shuffle
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(img_data, Y, random_state=2)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

# Custom_vgg_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor =image_input, include_top = True, weights='imagenet')
model.summary()
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation = 'softmax', name = 'output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-1]:
    layer.trainable = False

custom_vgg_model.layers[3].trainable
custom_vgg_model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
t=time.time()
#t = now()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

custom_vgg_model.save('model_v1.h5')

#visualization

train_loss=hist.history['loss']
val_loss = hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize = (7, 5))
plt.plot (xc, train_loss)
plt.plot (xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.legend(['train','val'])
#print plt style
plt.style.use(['classic'])

plt.figure(2,figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])

plt.show()
