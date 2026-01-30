from autocar3g.AI import Track_Follow_TF
from autocar3g.camera import Camera 
from autocar3g.driving import Driving

import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_tensor_float_32_execution(False)

import cv2, time

import os

# 에러 메시지에 나온 그 경로를 그대로 복사해서 넣으세요.
os.environ['LD_PRELOAD'] = '/usr/local/lib/python3.8/dist-packages/tensorflow/python/platform/../../../tensorflow_cpu_aws.libs/libgomp-cc9055c7.so.1.0.0'

cam = Camera()
cam.start() 

car = Driving()
throttle = 10

TF = Track_Follow_TF(cam)
TF.load_model("repo/Track_Model_new_origin_right.h5")

while True:
    try:
        car.throttle = 0
        ret_tf = TF.run()
        if ret_tf is not None:
            steer=(ret_tf['x']-0.5)*2
            print(f'steer: {steer}')
            car.steering=steer 
    except KeyboardInterrupt:
        car.throttle = 0
        car.steering = 0 
        break