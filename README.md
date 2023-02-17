# Deep AutoEncoder
I designed a deep AE to perform feature extraction and dimension reduction.

An auto-encoder (AE) is a neural network that learns to copy its input to its output. It has an internal (hidden) layer that describes a code used to represent the input, and it is formed by two main parts: an encoder that maps the input into the code (latent space), and a decoder that maps the code to a reconstruction of the original input. One of the main usage of AEs is to reduce the dimension by extracting meaningful features in the latent space (code layer). Representing data in a lower-dimensional space can improve performance on different tasks, such as classification and clustering. In the following you can see a standard deep auto-encoder:

![image](https://user-images.githubusercontent.com/24508376/219633309-1dc20416-1b46-44c6-8861-cff79b3002e1.png)


here I design a deep AE to perform feature extraction and dimension reduction on a given dataset (Lung Cancer Microarray Dataset) which contains 1626 genes and each of them has 181 features.

At first I read the data and normalize the dataset and chunk test train for 0.3 percent 

Then use 2,3,4,5 latent dim for code layer and 181 for input layer

Then use 3 encode and 3 decode layer at first depth with 64 and then in depth 2 with 16 and then in to latent dim in 2 3 4 5 nerun and then decode 16 in depth 3 and use 64 in depth 2 and then 181 in output layer

Then plot train and test and get kMeans for test in latent layer and plot them

And get the kMeans DBS (Davies–Bouldin index) at the end
I comment all of description in my code too

![image](https://user-images.githubusercontent.com/24508376/219633748-a21e6702-a5a3-4307-b040-e7ee26d64c5f.png)


![image](https://user-images.githubusercontent.com/24508376/219633827-465473a7-da4a-4339-88d0-f40bb426a1de.png)


![image](https://user-images.githubusercontent.com/24508376/219633892-231787bb-536a-45f4-a04f-9a811f22a9d7.png)

