import keras
import keras2onnx
import onnx
from keras.models import load_model
from keras.layers import Input
from keras.applications.resnet50 import ResNet50



model = load_model('mobile_model-05.h5')
#model = ResNet50()
onnx_model = keras2onnx.convert_keras(model, model.name,channel_first_inputs=['input_1'],target_opset=12)
temp_model_file = 'result.onnx'
onnx.save_model(onnx_model, temp_model_file)
