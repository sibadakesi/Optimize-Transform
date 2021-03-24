import keras
import keras2onnx
import onnx
from keras import optimizers
from keras.models import load_model
from keras.layers import Input,Dense
from keras.applications.mobilenetv2 import MobileNetV2
from keras import Model
from keras import losses
from keras.callbacks import *

https://forums.developer.nvidia.com/t/tensorrts-onnx-parser-cant-parse-the-output-layer-correctly/71252
https://zhuanlan.zhihu.com/p/129622586


model = load_model('mobile_model-05.h5')
#model = load_model('result.h5')
#model = MobileNetV2()
#model.save('result.h5')
onnx_model = keras2onnx.convert_keras(model, model.name, channel_first_inputs=['input_1'], target_opset=12)
temp_model_file = 'result.onnx'
onnx.save_model(onnx_model, temp_model_file)
