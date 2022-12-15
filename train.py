
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
plt.style.use('dark_background')


x_train = np.load('x_train.npy').astype(np.float32)
y_train = np.load('y_train.npy').astype(np.float32)
x_val = np.load('x_val.npy').astype(np.float32)
y_val = np.load('y_val.npy').astype(np.float32)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


plt.subplot(2, 1, 1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0].reshape((26, 34)), cmap='gray')
plt.subplot(2, 1, 2)
plt.title(str(y_val[4]))
plt.imshow(x_val[4].reshape((26, 34)), cmap='gray')


#이미지 처리를 도와주는 클래스
train_datagen = ImageDataGenerator(
    rescale=1./255, #0-255의 값을 가지는데 255를 곱해줘서 0아니면 1의 값을 가진다
    rotation_range=10, #돌리기
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2 #이미지 비틀기
)

#데이터알규먼트 - 데이터가 부족할때 쓰는 방법 . 모델이 더 잘 학습할 수 있다. 이미지의 형태가 다양해지니까

#모델이 잘 학습했는지 확인하는 작업
val_datagen = ImageDataGenerator(rescale=1./255) #0에서 1사이 값


#데이터제너레이터, flow메소드를 사용하면 쉬움
train_generator = train_datagen.flow(
    x=x_train, y=y_train,
    batch_size=32, #32개씩 꺼내와
    shuffle=True
)

val_generator = val_datagen.flow(
    x=x_val, y=y_val,
    batch_size=32,
    shuffle=False 
)


#모델링
inputs = Input(shape=(26, 34, 1))

#32사이즈로 3번 돌리기
net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

#1차원으로 길게 펼치기
net = Flatten()(net)

#
net = Dense(512)(net)
net = Activation('relu')(net)
net = Dense(1)(net) #0-1사이 값이니까 최종결과값은 1개만 출력
outputs = Activation('sigmoid')(net)

#모델정의
model = Model(inputs=inputs, outputs=outputs)

#모델을 어떻게 학습할건지, 이름은 adam, 0아니면 1 이라서 binary형식을 사용함.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) 

model.summary()

#
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

#학습시키기 , 제너레이터로 불러왔으면 핏제너레이터로 학습시키는게 좋음
model.fit_generator(
    train_generator, epochs=50, validation_data=val_generator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' % (start_time), monitor='val_acc', save_best_only=True, mode='max', verbose=1), #좋으면 저장
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05) #안좋으면 버려
    ]
)


#모델이 얼마나 잘 학습했는지
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns #그래프 패키지

model = load_model('models/%s.h5' % (start_time)) #케라스 모델을 로드한다

y_pred = model.predict(x_val/255.) #모델을 통해 데이터를 예측한다
y_pred_logical = (y_pred > 0.5).astype(np.int) #0.5보다 크면 뜬거 작으면 감은거 를 int로 바꿈

print ('test acc: %s' % accuracy_score(y_val, y_pred_logical)) #정확도 체크
cm = confusion_matrix(y_val, y_pred_logical)
sns.heatmap(cm, annot=True)


ax = sns.distplot(y_pred, kde=False)
