
# Pump failure Prediction


## Data

We have 1000 data points, cotains 2 classes like failure and non-failure classes. Each data point is collected at 1 Hr intervel, 

As per our analysis, 
- its a Multivariate time series dataset, ie dataset have multiple features (variables) recorded over a single time.
- No Missing values
- Data is Highly imbalanced with More that 95% non failure and remaining failure cases
- pressure, flow rate Level and Failure: Shows a moderate positive correlation, indicating that higher pressure levels may lead to pump failure.


## Data preprocessing

Correlation: Find the Correlation between the Features to calculate how effective each features on the target feature.

class_weights : Since the data is Time series data, and the need to carefully handle the imbalanced data, in the normal case we may introduce over sampling and under sampling using the SMOT method, here this method is inefficient as we require the data in order and cant introduce any synthetic data which may decreas the uality of actual data. class_weights is dictionary, with key is class and value will be its relative importance.



## Steps for Preparing Data:

### Feature Engineering : 
LSTM models expect the input to be in the form of 3D arrays: [samples, time steps, features]

Where:

samples: Number of sequences (i.e., rows of data representing each example in the dataset).
time steps: The number of time steps (i.e., how many previous data points to use for each prediction).
features: The number of variables recorded at each time step (i.e., the number of columns in your dataset).

To create this 3D shape, you typically use a sliding window approach where you take fixed-length sequences of your time series as input, and associate each sequence with the correct output label.


### Normalize the data :
so that all features have a similar scale. LSTMs perform better when the inputs are scaled, as it helps in stable convergence during training.
A common method is Min-Max scaling where values are scaled between 0 and 1.

### Create Sequences (Sliding Window)
To prepare the data for LSTM, you need to create sequences (subsets of data) from the original time series, where each sequence will be used as an input to predict a label

