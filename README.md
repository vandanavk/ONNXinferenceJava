# ONNX model inference using MXNet's Java API

1. Clone this repo with git clone https://github.com/vandanavk/ONNXinferenceJava.git
2. Create a conda environment


`pip install conda`

`conda create -n mxnet_blog python=3.6`
 
`conda activate mxnet_blog`

3. Install dependencies

`pip install mxnet`

`pip install chainer==5.3.0`

`pip install chainercv==0.12.0`

`pip install onnx==1.3.0`

`pip install onnx_chainer==1.3.3`

4. Execute the Python file to convert the Chainer model to an MXNet model

`python chainer_onnx_mxnet.py`

Check model/ for 

`chainer_vgg16.onnx`

`vgg16-symbol.json`

`vgg16-0000.params`

Check data/ for

`Penguin.jpg`

5. Install the pre-requisites mentioned in here[https://github.com/apache/incubator-mxnet/blob/master/docs/tutorials/java/mxnet_java_on_intellij.md]

6. Open ONNXInferenceJava in IntelliJ, build the project and run ONNXMXNetJava.java

7. Deactivate conda environment

`conda deactivate mxnet_blog`