# 2016_Kaggle_DataScience_Bowl
Calculating Left Ventricle volume from MRI data using CNN

There are two approaches.
## First approach:
This approach follows closely with the provided Deep Learning tutorial from Kaggle.
Using Sunnybrook MRI dataset, which provide contour label, we train our net to recognize left ventricle
Then we use the trained net to predict contour on each slice of the testing data.
For each study, we predict contour on each slice at each time point. 
Then we calculate the Diastole volume using the time point with the largest area in each slice.
We calculate the Systole volume using the time point with the smallest area in each slice.

## Second approach:
Using Kaggle's training dataset, we feed in each slice's 30 time points together at a time. 
We encode the label data to be a vector of length 600, representing the CDF format that the final submission need to be in.
Basic model structure:
Standard Conv/relu/pooling layers + sigmoid activation and CrossEntropy loss. Prediction data is pulled out of the sigmoid activation layer.


Caffe's Prototxt files are included.