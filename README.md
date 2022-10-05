# feedforward-neural-network

![img.png](img.png)

![img_1.png](img_1.png)

1. Pooling layer will reduce the spatial dimensions of the given input
2. Batch normalization fit in the certain area of the network. Which should help stabalized trainig

stack layers together to form a CNN:
INPUT > CONV > RELU > FC > SOFTMAX

- If the stride is 2, we are not learning anything. Skipping the value too much
- while increasing the value of stride, you are reducing the volume
- By using zero padding we can preserve the output dimensions

![img_2.png](img_2.png)

![img_3.png](img_3.png)

![img_4.png](img_4.png)

![img_5.png](img_5.png)

![img_6.png](img_6.png)

![img_7.png](img_7.png)

![img_8.png](img_8.png)

Batch Normalization:
    will allow us to train our model in most stable manner. It's basic scalling in normalization
1. Performing this batch normalization makes our training more stable
2. Batch normalization tipycally allow you to train your network in less no of epochs
3. Batch normalization typically go after the activation
INPUT > CONV > RELU > BN

![img_9.png](img_9.png)

![img_10.png](img_10.png)

![img_11.png](img_11.png)

![img_12.png](img_12.png)