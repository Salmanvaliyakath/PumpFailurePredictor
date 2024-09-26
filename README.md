
# Pump failure Prediction

## Table of Contents
1. [Objective](#Objective)
2. [Exploratory Data Analysis](#Exploratory Data Analysis)

    
## Objective
Build, Evaluate, and Simulate the deployment of a machine learning model for predictive maintenance, Use the time series sensor readings from an industrial pump.

## Exploratory Data Analysis

The dataset consists of 1000 data points, representing two classes: failure and non-failure. Each data point is collected at 1-hour intervals.

### Key observations from the analysis:

- The dataset is a multivariate time series, meaning it contains multiple features (variables) recorded over the same time period.
- There are no missing values.
- The data is highly imbalanced, with over 95% of the points representing non-failure cases and the rest representing failure cases.
- Pressure and flow rate level show a moderate positive correlation with failure, suggesting that higher pressure levels may contribute to pump failures.

### Correlation:
Find the Correlation between the Features to calculate how effective each features on the target feature.
Observation : pressure, flow rate Level and Failure: Shows a moderate positive correlation, indicating that higher pressure levels may lead to pump failure.

![image](https://github.com/user-attachments/assets/c73e0a0f-052c-4722-a689-f020d7561d70)


### Distribution

![image](https://github.com/user-attachments/assets/33aed1c5-b6c5-485b-9331-7ecbb7a4d4ae)

Based on the correlation matrix, 

- Vibration Level: The distribution is centered around 0.5 with slight variation, suggesting most pumps experience low to moderate vibration levels, and no extreme outliers are immediately visible.
- Temperature (°C): The temperature distribution appears roughly normal, centered around 70°C, with slight variations, suggesting most pumps operate within a narrow temperature range. slight left skewness is visible.
- Pressure (PSI): The pressure distribution is also fairly normal, centered around 95 PSI, indicating that most pumps are operating within a standard pressure range, with few instances of extreme high or low pressure.
- Flow Rate (m³/h): The flow rate shows a slightly skewed distribution, with most values concentrated between 40 and 60 m³/h. This may indicate that pumps typically operate within a consistent flow range.


## Data preprocessing

### Feature Engineering : 

LSTM models expect the input to be in the form of 3D arrays: [samples, time steps, features], Where:

- samples: Number of sequences (i.e., rows of data representing each example in the dataset).
- time steps: The number of time steps (i.e., how many previous data points to use for each prediction).
- features: The number of variables recorded at each time step (i.e., the number of columns in your dataset).

To create this 3D shape, you typically use a sliding window approach where you take fixed-length sequences of your time series as input, and associate each sequence with the correct output label.

### class_weights: 
To address the imbalance in time series data, typical approaches like oversampling and undersampling (e.g., using SMOTE) are not suitable here due to the need to maintain the sequential order of the data. Introducing synthetic data could compromise the integrity of the original dataset. Instead, we use class weights, which is a dictionary where the key represents the class and the value indicates its relative importance in training.

### Normalize the data :
so that all features have a similar scale. LSTMs perform better when the inputs are scaled, as it helps in stable convergence during training.
A common method is Min-Max scaling where values are scaled between 0 and 1.

### Create Sequences (Sliding Window)
To prepare the data for LSTM, you need to create sequences (subsets of data) from the original time series, where each sequence will be used as an input to predict a label

### Train-Test Split
When working with time series data, it's important to respect the temporal order. Always split the dataset in a way that prevents data leakage from the future into the past.
Note: Do not shuffle the time series data as this will break the temporal dependencies.


## Model Development
Once the data is prepared, you can now build and train the LSTM model.
Use the Keras to create a simple LSTM Network, and added dropout to reduce overfit, configure the input and output. and compile the model.

Once the model Build, compile and Trianed with the Training data, The trained model wil store for the fure use

## Model Evaluation
Since it is a classification problem, we used the mtrics such as Precision, Recall and F1 score to evaluate the LSTM model.

Attachin a sample evaluation metrics here with,
![image](https://github.com/user-attachments/assets/2c5daa27-576d-400b-83fc-648dad8948ed)


The F1 score need to be improved to get the model production ready, Observations are.

- Positive class : failure (1)
   - TP : Model predicts failure, actually failure
   - FP : model predicts failure, actually not failure
- Negative class : non failure (0)
   - TN : Model predict non failure, actual non failure
   - FN : Model predicts non failure, but actual failrue

By looking above, it is very clear that, FN need to be reduced, so the Recall is very important.

## Data Visualization and Reporting
Relationship between different features and target 'Failure' is shouwn in the Jupyter notebook.

## Model Deployment Simulation
By running the notebook successfully, the model will be build, trained, evaluated and stored in the folder.
To deploy the model use the main.py file which is a FastAPI app, 

Open the python terminal, open the folder,

RUN the "fastapi dev main.py"

Once the endpoint is up and running, I used the Postman to test the API

<img width="959" alt="image" src="https://github.com/user-attachments/assets/4d215563-7cc7-4b41-aa42-94348da4c548">

Input : path to a csv file
Output: A new csv file will be created with an extra predicted column

Sample predicted file provide in this repo.

## Bonus Task 

I have used the class_weights approach to encounter the imbalanced data. 

Use the features such as rolling averages, trend indicators to improve the model efficiency
Also wanted to try other model like XGBoost model with appropriate feature engineering.

## How to Run

Clone this repo:

- https://github.com/Salmanvaliyakath/PumpFailurePredictor.git
Open the terminal

Copy your training and testing data to this folder.

install all the dependencies
- pip install -r requirements.txt

To build the Dataset, Train, Evaluate and store the model, Open the model_train.ipynb notebook and run one by one cell,

To Deploy the Model, Run the main.py script, "fastapi dev main.py" to up and run the endpoint (API).

Open the postman application to test the API and pass the input URL and Body.

Or open the command prompt (Ctrl+R, type cmd, press) and 

curl --location 'http://127.0.0.1:8000/predict' \
--header 'Content-Type: application/json' \
--data '{"filepath":"hypothetical_pump_failure_dataset_test.csv"}' 

