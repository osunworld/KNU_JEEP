import os
import cv2
import glob
import numpy as np
# from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

import os

# 에러 메시지에 나온 그 경로를 그대로 복사해서 넣으세요.
os.environ['LD_PRELOAD'] = '/usr/local/lib/python3.8/dist-packages/tensorflow/python/platform/../../../tensorflow_cpu_aws.libs/libgomp-cc9055c7.so.1.0.0'

print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU'))}개")

# 데이터셋 경로 설정
DATA_PATH = 'repo/new_Right'
image_files = glob.glob(os.path.join(DATA_PATH, '*.jpg'))

print(f"Found {len(image_files)} images")

label = []
img = []

for file in image_files:
    try:
        # 파일명에서 좌표값 추출 (예: 100_200.jpg)
        filename = os.path.basename(file)
        label.append([float(filename.split('_')[0]), float(filename.split('_')[1].split('.')[0])])
    except:
        pass
    
    X = cv2.imread(file)
    if X is not None:
        img.append(X[120:270, :])

label = np.array(label)
img = np.array(img)

print(f"Label shape: {label.shape}, Image shape: {img.shape}")

# 정규화
label = label / 400

# 기존 모델이 있으면 로드, 없으면 새로 생성
MODEL_PATH = 'repo/Track_Model_new_right.h5'

if os.path.exists(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH}")
    try:
        model = keras.models.load_model(MODEL_PATH, custom_objects={'mse': 'mse'})
        print("Model loaded successfully!")
    except:
        print("Failed to load model, creating new one...")
        # 모델 구축
        input1 = keras.layers.Input(shape=(150, 400, 3,))
        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(input1)
        norm1 = keras.layers.BatchNormalization()(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(norm1)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(pool1)
        norm2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm2)
        norm3 = keras.layers.BatchNormalization()(conv3)
        add1 = keras.layers.Add()([norm2, norm3])
        conv4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add1)
        norm4 = keras.layers.BatchNormalization()(conv4)
        conv5 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm4)
        norm5 = keras.layers.BatchNormalization()(conv5)
        add2 = keras.layers.Add()([norm4, norm5])
        conv6 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add2)
        norm6 = keras.layers.BatchNormalization()(conv6)
        conv7 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm6)
        norm7 = keras.layers.BatchNormalization()(conv7)
        add3 = keras.layers.Add()([norm6, norm7])
        conv8 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add3)
        norm8 = keras.layers.BatchNormalization()(conv8)
        conv9 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(norm8)
        norm9 = keras.layers.BatchNormalization()(conv9)
        flat1 = keras.layers.Flatten()(norm9)
        dense1 = keras.layers.Dense(128, activation="swish")(flat1)
        norm10 = keras.layers.BatchNormalization()(dense1)
        dense2 = keras.layers.Dense(64, activation="swish")(norm10)
        norm11 = keras.layers.BatchNormalization()(dense2)
        dense3 = keras.layers.Dense(64, activation="swish")(norm11)
        norm12 = keras.layers.BatchNormalization()(dense3)
        dense4 = keras.layers.Dense(2, activation="sigmoid")(norm12)
        model = keras.models.Model(inputs=input1, outputs=dense4)
else:
    print("Creating new model...")
    # 모델 구축
    input1 = keras.layers.Input(shape=(150, 400, 3,))
    conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(input1)
    norm1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(norm1)
    conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(pool1)
    norm2 = keras.layers.BatchNormalization()(conv2)
    conv3 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm2)
    norm3 = keras.layers.BatchNormalization()(conv3)
    add1 = keras.layers.Add()([norm2, norm3])
    conv4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add1)
    norm4 = keras.layers.BatchNormalization()(conv4)
    conv5 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm4)
    norm5 = keras.layers.BatchNormalization()(conv5)
    add2 = keras.layers.Add()([norm4, norm5])
    conv6 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add2)
    norm6 = keras.layers.BatchNormalization()(conv6)
    conv7 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm6)
    norm7 = keras.layers.BatchNormalization()(conv7)
    add3 = keras.layers.Add()([norm6, norm7])
    conv8 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add3)
    norm8 = keras.layers.BatchNormalization()(conv8)
    conv9 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(norm8)
    norm9 = keras.layers.BatchNormalization()(conv9)
    flat1 = keras.layers.Flatten()(norm9)
    dense1 = keras.layers.Dense(128, activation="swish")(flat1)
    norm10 = keras.layers.BatchNormalization()(dense1)
    dense2 = keras.layers.Dense(64, activation="swish")(norm10)
    norm11 = keras.layers.BatchNormalization()(dense2)
    dense3 = keras.layers.Dense(64, activation="swish")(norm11)
    norm12 = keras.layers.BatchNormalization()(dense3)
    dense4 = keras.layers.Dense(2, activation="sigmoid")(norm12)
    model = keras.models.Model(inputs=input1, outputs=dense4)
    model.save('repo/Track_Model_new_right.h5')
    print("Model saved to repo/Track_Model_new_right.h5") 

# 모델 컴파일
adam = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=adam, loss="mse")

# Early Stopping 콜백
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, min_delta=1e-4)

# 모델 학습
print("Starting training...")
model.fit(x=img, y=label, epochs=100, batch_size=32, validation_split=0.1, callbacks=[es])

# 모델 저장
model.save('repo/Track_Model_new_right.h5')
print("Model saved to repo/Track_Model_new_right.h5")