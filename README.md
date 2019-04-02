# ONNX model inference using MXNet's Java API

### Recommendations:
* Use Python 3.5+ as onnx_chainer requires Python 3.5+
* Chainer is not guaranteed to work on MacOS and Windows ([Reference](https://docs.chainer.org/en/stable/install.html#recommended-environments))
* Use [conda](https://pypi.org/project/conda/) to create a virtual environment for Python

### Installation:

1. Clone this repo with `git clone https://github.com/vandanavk/ONNXinferenceJava.git`
2. Create and activate a conda environment


```
pip install conda
conda create -n mxnet_blog python=3.6

conda activate mxnet_blog
```

3. Install dependencies

```
pip install mxnet==1.4.0
pip install chainer==5.3.0
pip install chainercv==0.12.0
pip install onnx==1.3.0
pip install onnx_chainer==1.3.3
```

4. For Java, install the pre-requisites mentioned in [here](https://github.com/apache/incubator-mxnet/blob/master/docs/tutorials/java/mxnet_java_on_intellij.md)


### Code execution:

1. Execute the Python file to convert the Chainer model to an MXNet model

`python chainer_onnx_mxnet.py`

Check model/ folder for 
```
chainer_vgg16.onnx, vgg16-symbol.json, vgg16-0000.params, synset.txt
```

Check data/ folder for `Penguin.jpg`

2. Open ONNXInferenceJava in IntelliJ, build the project and run ONNXMXNetJava.java


Deactivate conda environment

`conda deactivate mxnet_blog`