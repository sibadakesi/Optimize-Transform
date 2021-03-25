import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import keras
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras import Model
from keras import losses
from keras.callbacks import *
import time
import glob
import keras2onnx
import onnx
from keras.models import load_model
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
from timeit import default_timer as timer
import cv2

explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class ModelData(object):
    MODEL_PATH = "result.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume([1, 3, 224, 224]), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume([1, 1000]), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()


# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network,
                                                                                                               TRT_LOGGER) as parser:
        builder.max_batch_size = 1
        config = builder.create_builder_config()
        builder.max_workspace_size = common.GiB(1)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print(parser.get_error(0).desc())
        profile = builder.create_optimization_profile()
        profile.set_shape("input_1", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)
        # last_layer = network.get_layer(network.num_layers - 1)
        # network.mark_output(last_layer.get_output(0))
        # return builder.build_cuda_engine(network)
        return builder.build_engine(network, config)


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = cv2.resize(image, (w, h)).transpose([2, 0, 1]).astype(
            trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(cv2.imread(test_image)))
    return test_image


def main():
    # Build a TensorRT engine.
    with build_engine_onnx('result.onnx') as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.

        with engine.create_execution_context() as context:
            context.set_binding_shape(0, (1, 3, 224, 224))
            # Load a normalized test case into the host input page-locked buffer.
            n = 10000
            tic = timer()
            while n:
                test_case = load_normalized_test_case('test.jpg', h_input)
                do_inference(context, h_input, d_input, h_output, d_output, stream)
                n -= 1
            toc = timer()
            print(toc - tic)  # 输出的时间，秒为单位
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            pred = [np.argmax(h_output)]
            print(pred)


if __name__ == '__main__':
    main()
