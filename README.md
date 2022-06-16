
# Gait Deep learning

The deep learning training of the GAIT dataset generated from the gait-logger-server and gait-logger-firmware are available in the src folder.
The gait-deep learning mostly studies on using three architectures CNN, CNN-Statistics and CNN-LSTM whose implementation and training is done using 
a Keras API. Then using different parameters and benchmarking factors, the best model is generated, and converted using tensor-js library to be deployed 
to the Raspberry Pi for inference.

The overall deeplearning architecture for training all the data was trained using this particular flow. Email sanjevsakya@gmail.com for the full dataset.

![image](https://user-images.githubusercontent.com/9900412/174111181-9c888ad0-3513-4e96-b479-bdcbb4bd9dcf.png)

The detailed implementation of CNN architecture is shown below:

![image](https://user-images.githubusercontent.com/9900412/174111776-b0efaaa0-d1c8-4023-97ee-2e13dc5c4be2.png)

The detail implementation of CNN-statistics is shown below:

![image](https://user-images.githubusercontent.com/9900412/174111928-546fc0c2-e11d-46b4-9f39-f088c3f9447b.png)

the statistical features are computed using 

![image](https://user-images.githubusercontent.com/9900412/174112060-6040f28a-938e-4d86-bbf6-ec52c53c68bb.png)

And the final CNN architecture is given by

![image](https://user-images.githubusercontent.com/9900412/174112153-a8d84bb9-f16c-41f0-a8df-787f820de44c.png)


These were the results of the best models of 
a. CNN 
![image](https://user-images.githubusercontent.com/9900412/174112371-47be0e31-28fe-41d4-80c7-29eb8e9f538c.png)

b. CNN-stats
![image](https://user-images.githubusercontent.com/9900412/174112429-afeb1723-1acd-4e7c-85ae-d0858bde64b1.png)

c. CNN-LSTM
![image](https://user-images.githubusercontent.com/9900412/174112484-a4f62a6f-237f-452a-b4a5-56d87720d780.png)

All the three architecture were deployed to the single board computer Raspberry Pi and the inference results were as follows:

![image](https://user-images.githubusercontent.com/9900412/174112585-6ed46ab1-3579-427a-9c0f-bd4b528c2f19.png)
