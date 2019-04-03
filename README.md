# ONNX model inference using MXNet's Java API

### Recommendations:
* Use Python 3.5+ as onnx_chainer requires Python 3.5+
* Chainer is not guaranteed to work on MacOS and Windows ([Reference](https://docs.chainer.org/en/stable/install.html#recommended-environments))
* Use [conda](https://pypi.org/project/conda/) to create a virtual environment for Python

### Installation:

1. Clone this repo with 

```
git clone https://github.com/vandanavk/ONNXinferenceJava.git
cd ONNXinferenceJava/
```

2. Create and activate a conda environment


```
pip install conda
conda create -n mxnet_blog python=3.6

conda activate mxnet_blog
```

3. Install Python dependencies

```
pip install mxnet==1.4.0
pip install chainer==5.3.0
pip install chainercv==0.12.0
pip install onnx==1.3.0
pip install onnx_chainer==1.3.3
```

4. For Java, install the pre-requisites [Reference](https://mxnet.apache.org/versions/master/install/java_setup.html)

For macOS

```
brew update
brew tap caskroom/versions
brew cask install java8
brew install maven
```

For Ubuntu

`sudo apt-get install openjdk-8-jdk maven`


### Code execution:

1. Execute the Python file to convert the Chainer model to an MXNet model

`python chainer_onnx_mxnet.py`

Check model/ folder for 
```
chainer_vgg16.onnx, vgg16-symbol.json, vgg16-0000.params, synset.txt
```

Check data/ folder for `Penguin.jpg`

2. Execute the Java file to perform inference

```
mvn clean dependency:copy-dependencies install
java -Xmx8G -cp target/ONNXJava-1.0-SNAPSHOT.jar:target/dependency/* mxnet.ONNXMXNetJava --model-path-prefix model/vgg16 --input-image data/Penguin.jpg
```

Alternatively, open ONNXInferenceJava in an IDE, build the project and run ONNXMXNetJava.java


Deactivate conda environment

`conda deactivate mxnet_blog`


#####Tip:
* If you are unable to activate the conda environment, [try this](https://github.com/conda/conda/issues/7980#issuecomment-441358406)