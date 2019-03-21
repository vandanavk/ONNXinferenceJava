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

# Step 1: Convert Chainer model to ONNX

# Export Chainer model to ONNX
model = L.VGG16(pretrained_model='imagenet')

# Pseudo input
x = np.zeros((1, 3, 224, 224), dtype=np.float32)

# Don't forget to set train flag off!
chainer.config.train = False

onnx_chainer.export(model, x, filename='model/chainer_vgg16.onnx')


# Step 2: Import ONNX model to MXNet
sym, arg_params, aux_params = onnx_mxnet.import_model('model/chainer_vgg16.onnx')


# Step 3: Save MXNet model's symbol and params
mod = mx.mod.Module(symbol=sym, data_names=['Input_0'], label_names=None)
mod.bind(for_training=False, data_shapes=[('Input_0', [1, 3, 224, 224])], label_shapes=mod._label_shapes)
mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
mod._symbol.save('model/vgg16-symbol.json')
mod.save_params('model/vgg16-0000.params')

# Download synset.txt that contain class labels
mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/synset.txt', dirname='model')

# Get a default image for inference
mx.test_utils.download('https://s3.amazonaws.com/onnx-mxnet/examples/mallard_duck.jpg', dirname='data')
