import numpy as np
import chainer
import chainercv.links as L
import onnx_chainer
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import os

folder = 'model'
if not os.path.exists(folder):
    os.makedirs(folder)

image_folder = 'data'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Download synset.txt that contain class labels
mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/synset.txt', dirname='model')

# Get a default image for inference
mx.test_utils.download('https://s3.amazonaws.com/onnx-mxnet/examples/Penguin.jpg', dirname='data')


def convert_model_to_onnx(input_shape, onnx_file_path):
    # Export Chainer model to ONNX
    model = L.VGG16(pretrained_model='imagenet')

    # Pseudo input
    x = np.zeros(input_shape, dtype=np.float32)

    # Don't forget to set train flag off!
    chainer.config.train = False

    onnx_chainer.export(model, x, filename=onnx_file_path)


def import_onnx_to_mxnet(onnx_file_path):
    return onnx_mxnet.import_model(onnx_file_path)


def save_mxnet_model(sym, arg, aux, input_name, input_shape):
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

    mod = mx.mod.Module(symbol=sym, data_names=[input_name], label_names=None, context=ctx)
    mod.bind(for_training=False, data_shapes=[(input_name, input_shape)], label_shapes=mod._label_shapes)
    mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)
    mod._symbol.save('model/vgg16-symbol.json')
    mod.save_params('model/vgg16-0000.params')


if __name__ == '__main__':
    shape = (1, 3, 224, 224)

    # Step 1: Convert Chainer model to ONNX
    convert_model_to_onnx(shape, 'model/chainer_vgg16.onnx')

    # Step 2: Import ONNX model to MXNet
    sym, arg_params, aux_params = import_onnx_to_mxnet('model/chainer_vgg16.onnx')

    # Step 3: Save MXNet model's symbol and params
    save_mxnet_model(sym, arg_params, aux_params, 'Input_0', shape)
