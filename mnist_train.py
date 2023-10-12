from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam

(x_train, y_train),(x_test,y_test) = load_data()

x_train = x_train.astype('float32')
x_train = x_train/255

x_test = x_test.astype('float32')
x_test = x_test/255

print(x_train.shape)
print(x_test.shape)

x_train=x_train.reshape(60000,98,8)
x_test=x_test.reshape(10000,98,8)
print(x_train.shape)
print(x_test.shape)

i = randint(0, x_train.shape[0])
plt.imshow(x_train[i])
plt.show()

#모델 생성
model = Sequential()

model.add(Conv2D(32,(5,5), activation='relu', padding='same', input_shape=(98,8,1)))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Conv2D(64,(5,5), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Conv2D(128,(5,5), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same')) #AE와 동일하게 하기 위함

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

opt=Adam()
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history=model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test), verbose=1)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(1, len(loss)+1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('test accuracy: %.3f%%' % (test_acc * 100))

y_pred_test = model.predict(x_test)
print(y_pred_test.shape)

prediction_test = np.argmax(y_pred_test, axis=1)
print(prediction_test.shape)

y_pred_test = model.predict(x_test)
prediction_test = np.argmax(y_pred_test, axis=1)

#혼동행렬로 확인하기
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, prediction_test)
sns.heatmap(cm,annot=True)
plt.show()
