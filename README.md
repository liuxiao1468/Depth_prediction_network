# Depth_prediction_network
This is the repository for a skip-connection and autoencoder based depth-prediction network. 
This work is in contribution to RA-L paper submitted to ICRA 2021 with Interactive Lab at ASU.

About the network structure:

The depth prediction network is built upon an dense autoencoder with skip connection from encoder to decoder.
The bottlenect has size 64
The building blocks include regular conv layers (with kernel 3x3 and 1x1), conv_blocks and identity_blocks (from Resnet).
Loss function: MSE + gradient + Total variance (TV)

Input: 180x320 RGB images
Output: 180x320 depth images

Sample Results:

Inputs:
![alt text](https://github.com/liuxiao1468/Depth_prediction_network/blob/master/RGB_images.png)

Outputs:
![Depth_prediction](https://user-images.githubusercontent.com/25230143/89096237-04575000-d38a-11ea-8c12-5fac633765e8.png)


Ground truth:
![Ground_truth](https://user-images.githubusercontent.com/25230143/89096253-251fa580-d38a-11ea-94ef-f2bad884045f.png)


