References:

# Autoencoder reference:


# Papers/blogs
https://www.jeremyjordan.me/autoencoders/

#Keras blog
https://blog.keras.io/building-autoencoders-in-keras.html

#Feature Extraction
https://machinelearningmastery.com/autoencoder-for-classification/


#Explanation with mathematical equations
https://yaledatascience.github.io/2016/10/29/autoencoders.html

#Blog
https://ff12.fastforwardlabs.com/

#Cross entropy loss:
https://gombru.github.io/2018/05/23/cross_entropy_loss/

#Structured Similarity Measurement Index
https://en.wikipedia.org/wiki/Structural_similarity

# Lecture reference;
https://github.com/jeffheaton/t81_558_deep_learning


#2mins paper
https://www.youtube.com/watch?v=Rdpbnd0pCiI



# Autoencoder Types:
        1. Simple Autoencoder 
        2. Sparse Autoencoder
        3. Deep Fully Connected Autoencoder
        4. Deep convolutional Autoencoder
        5. Image Denoising 
        6. Sequence-to-sequence Autoencoder
        7. Variational autoencoder
        
        
 
 ## Autoencoder
 
 An autoencoder is a neural network that is trained to attempt to copy its input to its output.
 
 
 ![](https://i.imgur.com/Bw03yPG.png)

 
 **Encoder**: learns to map input data to a low-dimensional representation
 **Decoder**: learns to map this low-dimensional representation back to the original input data.
 
Autoencoder are **lossy**.
Autoencoder are **data specific**.

Reconstruction error i.e. differences of error between output and input

Reconstruction error can be used for anomoly detection:
 
        
        
### Objective of autoencoder 
    To minimize the reconstruction error: 
        
 
Autoencoders are not a true unsupervised learning technique, they are a self-supervised technique, a specific instance of supervised learning where the targets are generated from the input data.
  



        
  
 # Anamoly 

Anomalys are rare events, items, or observations which are suspicious because they differ significantly from standard behaviors or patterns.

Anomalies in data are also called standard deviations, outliers, noise, novelties, and exceptions.

Graphical Examples:

![](https://i.imgur.com/dYJFt3s.png)

![](https://i.imgur.com/kVaP0ex.png)


Common usecase:
    - Can be used for Data preprocessing: Outliner detections
    - Credit card fault detection with the help of users behaviour
    - Noise detection with the certain threshold
    - Manufacturing/Data Centers: Exceptions or outliners detections



 