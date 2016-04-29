![Deep](http://fastml.com/images/tensorflow/i_dont_always_use_deep_learning.jpg "Deep")


# First Steps - Interactive DataViz - TensorFlow in your browser
http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.39171&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification
http://cs.stanford.edu/people/karpathy/convnetjs/

# Another demo
http://vision.stanford.edu/teaching/cs231n/linear-classify-demo/

# Save yourself the headache of installing TensorFlow
https://hub.docker.com/r/tensorflow/tensorflow/

# AWS GPU
https://gist.github.com/AlexJoz/1670baf0b32573ca7923
http://eatcodeplay.com/installing-gpu-enabled-tensorflow-with-python-3-4-in-ec2/
http://eugenezhulenev.com/blog/2016/02/01/deep-learning-with-tensorflow-on-ec2-spot-instances/
Public AMI in US-East(N. Virginia): ami-9d0f3ff7




# Upgrade to Version 0.8 Distributed TensorFlow
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/tensorflow-0.8.0rc0-py2-none-any.whl
https://databricks.com/blog/2016/01/25/deep-learning-with-spark-and-tensorflow.html


![Hot](https://i.imgflip.com/12i14k.jpg "So hot right now")

# sklearners can find the TensofFlow sklearn API
"To smooth the transition from the `Scikit Learn world of one-liner machine learning` into the more open world of building different shapes of ML models. You can start by using fit/predict and slide into TensorFlow APIs as you are getting comfortable."
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn
http://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/
https://github.com/tensorflow/tensorflow/tree/e39d8feebb9666a331345cd8d960f5ade4652bba/tensorflow/examples/skflow
https://github.com/tensorflow/tensorflow/blob/e39d8feebb9666a331345cd8d960f5ade4652bba/tensorflow/examples/skflow/iris_save_restore.py

# tensorflow models in production with serving
http://googleresearch.blogspot.ca/2016/02/running-your-models-in-production-with.html

# Learn the stuff off Udacity for Youtube learners
Some of the other videos (you know who you are...are pure crap watch me learn TensorFlow videos...lotsa clicks doesnt mean good)
https://www.udacity.com/course/deep-learning--ud730
https://www.youtube.com/watch?v=iQdWX1327XQ

# Stanford Deep Learning Computer Vision CS231n
https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC
http://cs231n.github.io/linear-classify/

# CS224 Youtube
https://www.youtube.com/watch?v=kZteabVD8sU
http://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf

# Learn and use these notebooks to implement basic to advanced TensorFlow
https://github.com/mbernico/CS570/tree/master/deepLearning

# Tutorial controlling backprop
https://nbviewer.jupyter.org/github/rdipietro/tensorflow-notebooks/blob/master/tensorflow_scan_examples/tensorflow_scan_examples.ipynb

# Submit to Kaggle
Try you hand at using TensorFlow for https://www.kaggle.com/c/digit-recognizer

# Visualizing MNIST data
http://colah.github.io/posts/2014-10-Visualizing-MNIST/

# Moar
https://github.com/317070/kaggle-heart

# Nerd out - Different papers MNIST
http://yann.lecun.com/exdb/mnist/
http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf
http://www.cs.toronto.edu/~hinton/absps/tsne.pdf
http://arxiv.org/pdf/1502.03167.pdf
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
http://arxiv.org/pdf/1502.03492.pdf
http://arxiv.org/pdf/1603.07285v1.pdf

# Stats for deep learning
http://joanbruna.github.io/stat212b/

# First NN read
http://neuralnetworksanddeeplearning.com/index.html



![Numbers](http://cs.stanford.edu/people/karpathy/convnetjs/mnist.png "Numbers")


# TensorFlow Examples
Code examples for some popular machine learning algorithms, using TensorFlow library. This tutorial is designed to easily dive into TensorFlow, through examples. It includes both notebook and code with explanations.

## Tutorial index

#### 1 - Introduction
- Hello World ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1%20-%20Introduction/helloworld.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1%20-%20Introduction/helloworld.py))
- Basic Operations ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1%20-%20Introduction/basic_operations.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1%20-%20Introduction/basic_operations.py))

#### 2 - Basic Classifiers
- Nearest Neighbor ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2%20-%20Basic%20Classifiers/nearest_neighbor.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2%20-%20Basic%20Classifiers/nearest_neighbor.py))
- Linear Regression ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2%20-%20Basic%20Classifiers/linear_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2%20-%20Basic%20Classifiers/linear_regression.py))
- Logistic Regression ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2%20-%20Basic%20Classifiers/logistic_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2%20-%20Basic%20Classifiers/logistic_regression.py))

#### 3 - Neural Networks
- Multilayer Perceptron ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/multilayer_perceptron.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/multilayer_perceptron.py))
- Convolutional Neural Network ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/convolutional_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/convolutional_network.py))
- AlexNet ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/alexnet.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/alexnet.py))
- Recurrent Neural Network (LSTM) ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/reccurent_network.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py))
- Bidirectional Recurrent Neural Network (LSTM) ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/bidirectional_rnn.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/bidirectional_rnn.py))

#### 4 - Multi GPU
- Basic Operations on multi-GPU ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/4%20-%20Multi%20GPU/multigpu_basics.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4%20-%20Multi%20GPU/multigpu_basics.py))

#### 5 - User Interface (Tensorboard)
- Graph Visualization ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5%20-%20User%20Interface/graph_visualization.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5%20-%20User%20Interface/graph_visualization.py))
- Loss Visualization ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5%20-%20User%20Interface/loss_visualization.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5%20-%20User%20Interface/loss_visualization.py))


## More Examples
The following examples are coming from [TFLearn](https://github.com/tflearn/tflearn), a library that provides a simplified interface for TensorFlow. You can have a look, there are many [examples](https://github.com/tflearn/tflearn/tree/master/examples) and [pre-built operations and layers](http://tflearn.org/doc_index/#api).

#### Basics
- [Linear Regression](https://github.com/tflearn/tflearn/blob/master/examples/basics/linear_regression.py). Implement a linear regression using TFLearn.
- [Logical Operators](https://github.com/tflearn/tflearn/blob/master/examples/basics/logical.py). Implement logical operators with TFLearn (also includes a usage of 'merge').
- [Weights Persistence](https://github.com/tflearn/tflearn/blob/master/examples/basics/weights_persistence.py). Save and Restore a model.
- [Fine-Tuning](https://github.com/tflearn/tflearn/blob/master/examples/basics/finetuning.py). Fine-Tune a pre-trained model on a new task.
- [Using HDF5](https://github.com/tflearn/tflearn/blob/master/examples/basics/use_hdf5.py). Use HDF5 to handle large datasets.
- [Using DASK](https://github.com/tflearn/tflearn/blob/master/examples/basics/use_dask.py). Use DASK to handle large datasets.

#### Computer Vision
- [Multi-layer perceptron](https://github.com/tflearn/tflearn/blob/master/examples/images/dnn.py). A multi-layer perceptron implementation for MNIST classification task.
- [Convolutional Network (MNIST)](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py). A Convolutional neural network implementation for classifying MNIST dataset.
- [Convolutional Network (CIFAR-10)](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py). A Convolutional neural network implementation for classifying CIFAR-10 dataset.
- [Network in Network](https://github.com/tflearn/tflearn/blob/master/examples/images/network_in_network.py). 'Network in Network' implementation for classifying CIFAR-10 dataset.
- [Alexnet](https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py). Apply Alexnet to Oxford Flowers 17 classification task.
- [VGGNet](https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py). Apply VGG Network to Oxford Flowers 17 classification task.
- [RNN Pixels](https://github.com/tflearn/tflearn/blob/master/examples/images/rnn_pixels.py). Use RNN (over sequence of pixels) to classify images.
- [Residual Network (MNIST)](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_mnist.py). A residual network with shallow bottlenecks applied to MNIST classification task.
- [Residual Network (CIFAR-10)](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py). A residual network with deep bottlenecks applied to CIFAR-10 classification task.
- [Auto Encoder](https://github.com/tflearn/tflearn/blob/master/examples/images/autoencoder.py). An auto encoder applied to MNIST handwritten digits.

#### Natural Language Processing
- [Reccurent Network (LSTM)](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm.py). Apply an LSTM to IMDB sentiment dataset classification task.
- [Bi-Directional LSTM](https://github.com/tflearn/tflearn/blob/master/examples/nlp/bidirectional_lstm.py). Apply a bi-directional LSTM to IMDB sentiment dataset classification task.
- [City Name Generation](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_cityname.py). Generates new US-cities name, using LSTM network.
- [Shakespeare Scripts Generation](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_shakespeare.py). Generates new Shakespeare scripts, using LSTM network.

## Dependencies
```
tensorflow
numpy
matplotlib
cuda (to run examples on GPU)
tflearn (if using tflearn examples)
```
For more details about TensorFlow installation, you can check [Setup_TensorFlow.md](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/Setup_TensorFlow.md)

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples (with input_data.py).
MNIST is a database of handwritten digits, with 60,000 examples for training and 10,000 examples for testing. (Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))
