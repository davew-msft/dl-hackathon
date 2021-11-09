# Lab 1: Getting A Working Development Environment

## Background

Before working on a computer vision solution, your team wants to standardize on a data science development environment.

After provisioning the development environment, you can use it to explore the images that Adventure Works has collected, and prepare them for use in training a machine learning model.

## Success Criteria

1. a working development environment
2. familiarization with the images and content by working with a sample notebook for this workshop.  

There are MANY ways to get a working development environment.  Here are 3 that work well:

* Use JupyterHub in Azure Machine Learning Service.  Considerations:
  * we can all work together and share code easily (if we use a shared AMLS workspace)
  * we can scale our solution (using GPUs) as needed
  * to control costs, just make sure you shutdown your Compute when it's not being used
* Use a local python development environment
  * Cheaper
  * Harder to setup
  * "it works on my machine" 
* Use a Data Science Virtual Machine in Azure (Ubuntu)
  * has all of the necessary tooling
  * requires you to connect to the Ubuntu desktop over VNC, which can be challenging

### Option 1:  Use Jupyter in AMLS

* We will review this together.  

### Option 2:  Provision a Data Science Virtual Machine in Azure

This Azure virtual machine image includes essential data science tools, including the Jupyterhub notebook environment; in which you can create and run Python code.

The following DSVM configuration has been found to work well, and is the recommended environment for this hack:

- DSVM Image: *Data Science Virtual Machine for Linux (Ubuntu)*
- Region: _Any available region_
- Size:  _NC6_ (Filter by **Family** to list **GPU** enabled images)
- Authentication type:  _Password_
- Username: _(Specify a **lowercase** user name of your choice)_
- Password: _(Specify a complex password)_

** the ability to create a GPU-based VM may be restricted to specific regions. If you find that no GPU images are available, go back, change regions, and try again.  You also may not be able to use GPU VMs based on your Azure subscription limitations.  Just use a CPU vm in that case. 

**You must use Password auth type.  There are some issues when you *setup* a Linux DSVM with ssh.  You can add your ssh public key AFTER you have the DSVM provisioned.**

  * When provisioning the DSVM, specify a **lowercase** user name and be sure to choose **Password** as the authentication type.

**Make sure you SHUTDOWN this vm when not in use...it is a GPU VM so it's a little pricey.**

After the DSVM has been created, connect to Jupyterhub and log in using the username and password you specified when provisioning the DSVM.

_Why use GPUs?_  GPUs are meant for training neural networks.  They tend to train orders of magnitude faster than CPUs.  We won't be training with a lot of data so it's OK if you don't use a GPU for training.  

You *may* need to connect to the DSVM remotely.  It's best to use `x2go client`.  Use `xfce` as the connection type.  Ensure you can connect.  

- Jupyterhub is at **https://*your.dsvm.ip.address*:8000**. 
- To get to the Jupyterhub, you must click through the non-private connection warnings in browser - this is expected behavior.
- If  Jupyterhub takes a while to load, click the **jupyter** logo to open the folder tree page.
- JupyterHub is available at https://*ip*:8000/user/<user>/lab 
  - this is a much improved interface

### Option 3:  Local laptop setup

_The goal is to have a working Jupyter notebook environment. However you can do that on your local laptop is fine._

There are a lot of options here and each has a "complexity tradeoff", and also will depend on whether you are Mac or Windows.  
* dev ready containers
* install python directly in Windows
* use WSL2 

You will likely need, minimally:  
* vscode (or any IDE)
  * python extension 
  * jupyter notebook extension


## Explore your Jupyter/Notebook Environment

By now you should have a working Jupyter notebook environment.  

You will want to clone this repo so that you have access to the notebooks we will use.  

```shell
mkdir -p git
cd git
git clone https://github.com/davew-msft/dl-hackathon
```


## Prepare Image Data for Machine Learning

1. Open [notebooks/01-DataPrep.ipynb](notebooks/01-DataPrep.ipynb) in your Jupyter environment. Examine the notes and code it contains. Run each code cell, and review the output. The code in the notebook

1. Downloads and extracts a folder hierarchy of image files that you will use in subsequent challenges.
2. Displays the first image in each folder - each folder represents a category or *class* of product image.
3. Standardizes the images so that they are a common format and size.

> **Note**: In this Lab, the code has been provided for you to enable you to get familiar with the Jupyter notebook environment. However, you should take the time to review the code and ensure you understand it, because in later challenges you will need to write your own code to perform similar tasks!

> **Important:  Since I can't guarantee what environment you are running on, I can't guarantee the code will work without modification.  Carefully ensure each cell is running and generating the expected output.**

- The <a href="https://docs.python.org/3.6/tutorial/stdlib.html#operating-system-interface" target="_blank">**os** Python module</a> includes functions for interacting with the file system.
- The <a href="https://matplotlib.org/2.0.2/index.html" target="_blank">**matplotlib** Python library</a> provides functions for plotting visualizations and images.
- To ensure that plots are displayed in a notebook, you must run the following *magic* command before creating the first plot:

    `%matplotlib inline`

- Images are essentially just numeric arrays. In the case of color images, they are three-dimensional arrays that contain a two-dimensional array of pixels for each color channel. For example, a 128x128 Jpeg image is represented as three 128x128 pixel arrays (one each for the red, green, and blue color channels). The Python <a href="https://docs.scipy.org/doc/numpy/reference/arrays.html" target="_blank">*NumPy* library</a> provides a great way to work with multidimensional arrays. For example, you can use:
  - `numpy.array(my_img)` to exlicitly convert an image object to a numpy array.
  - `my_array.shape` to determine the size of the array dimensions - an image has three dimensions (height, width, and channels)
- There are several Python libraries for working with images, as noted in the **References** section. You can use whatever combination of these packages works best to process your images, and rely on the **numpy** array data type as an intermediary format.
- The <a href="https://pillow.readthedocs.io" target="_blank">*PIL* library</a> uses a native format for images, but you can easily convert PIL images to numpy arrays using the `numpy.array()` function, and you can convert a numpy array to a PIL Image object by using the `Image.fromarray()` function. You can also convert PIL images between image formats (for example, from a 4-channel PNG to a 3-channel JPG) using the `my_img.convert()` function.
- To open a file as a PIL Image object, use the `Image.open()` function. To save a PIL image as a file, use the `my_img.save()` function.
- A common strategy to resize an image while maintaining its aspect ratio is to:
  1. Scale the image so that its largest dimension (height or width) is set to the target size for that dimension. You can use the PIL `my_image.thumbnail()` method to accomplish this.
  2. Create a new image of the required size and shape with an appropriate background color. You can use the PIL `Image.new()` function to accomplish this.
  3. Paste the rescaled image into the center of the new background image. You can use the PIL `my_bg_img.paste()` function to accomplish this.
- When using <a href="https://matplotlib.org/2.0.2/users/image_tutorial.html" target="_blank">*matplotlib*</a> to plot multiple images in a grid format, create a figure and add a subplot for each image by using the `my_figure.add_subplot()` function. The parameters for this function are:
  - The total number of *rows* in the grid.
  - The total number of *columns* in the grid.
  - The *ordinal position* of this subplot in the grid (starting with 1 in the top-left cell).

## Success Criteria

To complete this Lab successfully, you must run the code in the notebook in your environment. The final code cell in the notebook should display the original and resized version of the first image in each folder, similar to the following:

![Gear Images](images/resized_gear_images.png)

## References

- <a href="https://matplotlib.org/2.0.2/users/image_tutorial.html" target="_blank">Using *matplotlib* for image I/O and plotting</a>
- <a href="https://pillow.readthedocs.io/en/5.3.x/reference/Image.html" target="_blank">Using the *PIL Image* module for I/O and more </a>
- <a href="http://pillow.readthedocs.io/en/5.3.x/reference/ImageOps.html" target="_blank">Using the *PIL ImageOps* module for image manipulation</a>
- <a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.io.html" target="_blank">Using *NumPy* for image I/O</a>
- <a href="http://www.scipy-lectures.org/advanced/image_processing/" target="_blank">Using *NumPy* for image manipulation/processing/visualization</a>


