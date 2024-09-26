
# Pump failure Prediction


## Data

We have 1000 data points, cotains 2 classes like failure and non-failure classes. Each data point is collected at 1 Hr intervel, 

As per our analysis, 
- its a Multivariate time series dataset, ie dataset have multiple features (variables) recorded over a single time.
- No Missing values
- Data is Highly imbalanced with More that 95% non failure and remaining failure cases
- pressure, flow rate Level and Failure: Shows a moderate positive correlation, indicating that higher pressure levels may lead to pump failure.


## 1. Data preprocessing

### class_weights: 
Since the data is Time series data, and the need to carefully handle the imbalanced data, in the normal case we may introduce over sampling and under sampling using the SMOT method, here this method is inefficient as we require the data in order and cant introduce any synthetic data which may decreas the uality of actual data. class_weights is dictionary, with key is class and value will be its relative importance.



## Steps for Preparing Data:

### Feature Engineering : 

LSTM models expect the input to be in the form of 3D arrays: [samples, time steps, features], Where:

samples: Number of sequences (i.e., rows of data representing each example in the dataset).
time steps: The number of time steps (i.e., how many previous data points to use for each prediction).
features: The number of variables recorded at each time step (i.e., the number of columns in your dataset).

To create this 3D shape, you typically use a sliding window approach where you take fixed-length sequences of your time series as input, and associate each sequence with the correct output label.

### Normalize the data :
so that all features have a similar scale. LSTMs perform better when the inputs are scaled, as it helps in stable convergence during training.
A common method is Min-Max scaling where values are scaled between 0 and 1.

### Create Sequences (Sliding Window)
To prepare the data for LSTM, you need to create sequences (subsets of data) from the original time series, where each sequence will be used as an input to predict a label

### Train-Test Split
When working with time series data, it's important to respect the temporal order. Always split the dataset in a way that prevents data leakage from the future into the past.
Note: Do not shuffle the time series data as this will break the temporal dependencies.


## 2.	Exploratory Data Analysis (EDA)

### Correlation:
Find the Correlation between the Features to calculate how effective each features on the target feature.
Observation : pressure, flow rate Level and Failure: Shows a moderate positive correlation, indicating that higher pressure levels may lead to pump failure.

![image](https://github.com/user-attachments/assets/c73e0a0f-052c-4722-a689-f020d7561d70)


### Distribution

![image](https://github.com/user-attachments/assets/33aed1c5-b6c5-485b-9331-7ecbb7a4d4ae)

vibration_level    0.116976
temperature_C     -0.049396
pressure_PSI       0.061247
flow_rate_m3h     -0.002121

Vibration Level: The distribution is centered around 0.5 with slight variation, suggesting most pumps experience low to moderate vibration levels, and no extreme outliers are immediately visible.
Temperature (°C): The temperature distribution appears roughly normal, centered around 70°C, with slight variations, suggesting most pumps operate within a narrow temperature range. slight left skewness is visible.
Pressure (PSI): The pressure distribution is also fairly normal, centered around 95 PSI, indicating that most pumps are operating within a standard pressure range, with few instances of extreme high or low pressure.
Flow Rate (m³/h): The flow rate shows a slightly skewed distribution, with most values concentrated between 40 and 60 m³/h. This may indicate that pumps typically operate within a consistent flow range.



