# Challenge 2:  Deep in the Woods

## Background

Your data science team wants to use *deep learning* techniques to build an image classification model that can identify each class of product the company sells. Specifically, the team wants to develop a *convolutional neural network* (CNN) model. You can use your choice of deep learning framework (*PyTorch* or *Keras* are most common...and I understand both and can provide assistance.)  `pyTorch` seems to be the way the industry is moving as of 2021 so consider using that.

### PyToch vs Keras

Keras:

* is an abstraction over either TensorFlow. Keras makes it easy to define neural networks in a very readable way. One of Keras` top sponsors is Google.  
* generally used on top of tensorflow but can also use `theano`.  more like traditional data scientists would use with vectors and matrices.  

pytorch:

* Created and backed by Facebook and MSFT 
* more like traditional programming (classes, functions)
* port of the popular Torch framework where the binaries are wrapped in GPU accelerated python

Both are fundamentally the same and if you are brand-new it doesn't really matter.

**Both *should* work for this hack, but I am more comfortable with pyTorch and have tested it a bit better.**

## Prerequisites

* The resized  ***gear*** image data fom the previous challenges.
* make sure this is in a folder called `notebooks/resized_images`

## Lab Instructions

There are three tasks in this challenge:

1. Explore a sample convolutional neural network model.  Let's get something simple working first.  
2. Create and train a convolutional neural network (CNN) model on your `resized_images`.  
3. Use your model with new data.


### 1. Explore a sample convolutional neural network model

Explore the notes and code in the [notebooks/02-ImageClassification.ipynb](notebooks/02-ImageClassification.ipynb) notebook.  

* You may see some errors when installing libraries.  Simply rerun the cell and see if that solves the issue.  These packages are constantly changing and I've done my best to ensure they are totally rerunnable, but they aren't.  

### 2. Create and train a convolutional neural network (CNN) model

Open [notebooks/Lab2-GearClassifier.ipynb](notebooks/Lab2-GearClassifier.ipynb).  We will create a CNN that predicts the class of an image based on the images in `notebooks/resized_images`.

* The code is NOT complete.  You will need to find/change the `# TODO` entries and complete the code.  
* The architecture of your model should consist of a series of *convolutional*, *pooling*, *drop*, and *fully-connected* layers that you define.
* The input layer of your model must match the size and shape of the training image arrays.
* The output layer of your model must include an output for each class the model is designed to predict.
* Randomly split the data into training and validation subsets with which to train and validate the model.
* For each epoch in your training process, you should record the average *loss* for both the training and validation data; and when training is complete you should plot the training and loss values like this:

    ![Training and Validation Loss](images/loss.png)

* Use the **Python 3.5** kernel, or higher
* Base your initial solution on the code in the sample notebook (shapes classifier) and use my starter code merely as a guide.  
* To improve the model's performance, try adding more convolutional and pool layers, or using more training epochs.
* Try to avoid *overfitting* your model to the training data. One sign of this is that after your training and validation loss metrics converge, the training loss continues to drop but your validation loss stays the same or rises (as shown in the image above). The end result is a model that performs well when predicting the classes of images that it has been trained on, but which does not generalize well to new images.
* Techniques to help avoid overfitting include:
  * Including *drop* layers to randomly remove some features from the model.
  * Augmenting the data with re-oriented, skewed, or otherwise altered versions of training images.
    * try "rotating" your images a bit and re-train

### 3. Use your model with new data

Use the model to predict the class of at least five relevant images that are not included in the ***gear*** dataset. You can find these images by using Bing to search for appropriate terms, for example:

* <a href="https://www.bing.com/images/search?q=ski+helmet" target="_blank">https://www.bing.com/images/search?q=ski+helmet</a>
* <a href="https://www.bing.com/images/search?q=climbing+axe" target="_blank">https://www.bing.com/images/search?q=climbing+axe</a>
* <a href="https://www.bing.com/images/search?q=tent" target="_blank">https://www.bing.com/images/search?q=tent</a>
* <a href="https://www.bing.com/images/search?q=carabiner" target="_blank">https://www.bing.com/images/search?q=carabiner</a>
* <a href="https://www.bing.com/images/search?q=insulated+jacket" target="_blank">https://www.bing.com/images/search?q=insulated+jacket</a>

* You can download the new images by using the `curl` command.
* Don't forget to resize and pre-process the new images to match the images with which the model was trained.

## Success Criteria

* Successfully train a convolutional neural network model
* Plot the average training and validation loss observed when training your model
* Achieve model accuracy of **0.80** (80%) or greater using your test data set, if you can.  The next Lab will show you an easier way to get a really good accuracy with very little coding.  

## References

### CNN Concepts

* <a href="https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/" target="_blank">An Intuitive Explanation of Convolutional Neural Networks</a>
* <a href="https://www.youtube.com/watch?v=FmpDIaiMIeA" target="_blank">How Convolutional Neural Networks work</a> (video)
* <a href="https://youtu.be/k-K3g4FKS_c" target="_blank">Demystifying AI</a> (video)

### Deep Learning Frameworks

* **<a href="https://pytorch.org/" target="_blank">PyTorch</a>**
  * <a href="https://pytorch.org/docs/stable/index.html" target="_blank">Documentation</a>
  * <a href="https://pytorch.org/tutorials/" target="_blank">Tutorials</a>
* **<a href="https://keras.io/" target="_blank">Keras</a>** (an abstraction layer that uses a TensorFlow or CNTK backend)
  * <a href="https://keras.io/" target="_blank">Documentation</a>
  * <a href="https://github.com/fchollet/keras-resources" target="_blank">Tutorials</a>
