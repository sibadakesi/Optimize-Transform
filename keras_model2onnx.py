import keras
import keras2onnx
import onnx
from keras import optimizers
from keras.models import load_model
from keras.layers import Input, Dense
from keras.applications.mobilenetv2 import MobileNetV2
from keras import Model
from keras import losses
from keras.callbacks import *
import cv2
from timeit import default_timer as timer

model = load_model('result.h5')
# model = MobileNetV2()
# model.save('result.h5')
onnx_model = keras2onnx.convert_keras(model, model.name, channel_first_inputs=['input_1'], target_opset=12)
temp_model_file = 'result.onnx'
onnx.save_model(onnx_model, temp_model_file)


# 计算推断时间
# model = MobileNetV2()
# img = cv2.imread('test.jpg')
# img = cv2.resize(img, (224, 224))
# img = np.expand_dims(img, 0)
# n = 10000
# tic = timer()
# while n:
#     out = model.predict(img)
#     n -= 1
# toc = timer()
# print(toc - tic)  # 输出的时间，秒为单位