#!/usr/bin/env python
# coding: utf-8

# ## Exercise 3
# In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
# 
# I've started the code for you -- you need to finish it!
# 
# When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
# 

# In[1]:


import tensorflow as tf
from os import path, getcwd, chdir
import matplotlib.pyplot as plt
# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
#path = f"{getcwd()}/.../tmp/mnist.npz"
path = f"{getcwd()}/mnist-numpy/mnist.npz"
print(path)
# In[2]:
'''config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)'''

# In[3]:


# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    callbacks = myCallback()
    
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images, training_labels, epochs=20, callbacks=[callbacks]
        # YOUR CODE ENDS HERE
    )
    # model fitting

    
    #printing loss and accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_loss, test_acc)

    
    return history.epoch, history.history['acc'][-1]


# In[ ]:


_, _ = train_mnist_conv()

#Visualizing The Convolutions and Pooling
print(test_labels[:100])
f,axarr=plt.subplots(3,4)
f_img=0
s_img=23
t_img=28
CONVOLUTION_NUMBER=1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = models.input, outputs = layer_outputs)
for x in range(0,4):
    f1 = activation_model.predict(test_images(f_img).reshape(1, 28, 28, 1))[x]
    axarr[0,x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0,x].grid(False)

    f2 = activation_model.predict(test_images(s_img).reshape(1, 28, 28, 1))[x]
    axarr[1,x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1,x].grid(False)

    f3 = activation_model.predict(test_images(t_img).reshape(1, 28, 28, 1))[x]
    axarr[2,x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2,x].grid(False)




# In[ ]:
#to apply in jupytor notebook
'''
get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')
'''
