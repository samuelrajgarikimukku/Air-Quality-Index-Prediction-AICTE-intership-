# Air Quality Index (AQI) Prediction Model 🌬️

A machine learning project to predict the Air Quality Index (AQI) using daily atmospheric pollutant data. This model analyzes various pollutants to forecast the overall air quality, providing a valuable tool for environmental monitoring and public health awareness.

## Table of Contents
- [Problem Statement](https://github.com/samuelrajgarikimukku/Air-Quality-Index-Prediction-AICTE-intership-/blob/main/README.md#problem-statement)
- [Project Workflow](ProjectWorkflow)
- [Dataset](Dataset)
- [Technology Stack](TechnologyStack)
- [Installation and Usage](InstallationandUsage)
- [Results and Conclusion](InstallationandUsage)
- [Future Scope]()

 ## Problem Statement
Air pollution is a critical environmental and health issue worldwide. High AQI values are linked to severe health problems, including respiratory illnesses and cardiovascular diseases. The ability to accurately predict the AQI in advance allows for timely public health warnings and proactive measures to mitigate pollution.

This project aims to develop a robust regression model that can predict the numerical AQI value based on the concentrations of key air pollutants.

## Project Workflow
The project followed a structured machine learning pipeline:

1. Data Cleaning & Preprocessing:
   Handled missing values by imputing the median, ensuring data integrity. Outliers were identified using box plots and treated using the Interquartile Range (IQR) method to prevent them from skewing the model's performance.

3. Exploratory Data Analysis (EDA): Performed in-depth analysis using visualizations to understand the dataset's structure, distributions, and relationships. Key visuals generated during this phase include:

<img width="935" height="528" alt="image" src="https://github.com/user-attachments/assets/02b9b29d-69b5-4849-8366-60b87e287146" />

A heatmap showing the correlation between different pollutants and the AQI. This helped identify which features were most influential.

<img width="489" height="489" alt="image" src="https://github.com/user-attachments/assets/bd5ca42f-bb06-4373-9158-941c86ae2961" />

A histogram of the AQI target variable. This showed the frequency of different air quality levels in the dataset.

3. Feature Engineering: Standardized all numerical features using StandardScaler to bring them to a common scale, which is essential for the performance of many machine learning algorithms.

4. Model Training: Trained and evaluated four different regression models to find the most effective one:

   - Linear Regression
   - K-Nearest Neighbors (KNN) Regressor
   - Decision Tree Regressor
   - Random Forest Regressor

5. Model Evaluation: Assessed the models based on two key metrics: R-squared $`R^2`$ Score to measure the proportion of variance explained by the model, and Root Mean Squared Error (RMSE) to measure prediction error.

 ## Dataset
The dataset used for this project is air quality data.csv, containing daily air quality measurements from various cities.

Features **(Pollutants)**: PM_2.5, PM_10, NO, NO_2, NO_x, NH_3, CO, SO_2, O_3, Benzene, Toluene, Xylene.

**Target Variable**: AQI (Air Quality Index - numerical value).

## Technology Stack
- Language: Python 3.x
- Libraries:
  - Pandas (for data manipulation)
  - bNumPy (for numerical operations)
  - Scikit-learn (for model building and evaluation)
  - Matplotlib & Seaborn (for data visualization)
  - Environment: Jupyter Notebook / Google Colab

## Installation and Usage
To run this project on your local machine, follow these steps:

### 1. lone the repository:
    git clone https://github.com/your-username/your-repo-name.git
      cd your-repo-name
## 2. Create a requirements.txt file with the following content:
      numpy
      pandas
      matplotlib
      seaborn
      scikit-learn
### 3. Install the required libraries:
    pip install -r requirements.txt 
## 4. Run the Jupyter Notebook:
    Open the .ipynb file in Jupyter Notebook or Google Colab and run the cells sequentially to see the entire workflow from data cleaning to model evaluation.

## Results and Conclusion
The models were thoroughly evaluated on the test dataset to measure their real-world performance. The results clearly indicate a top-performing model.

[-> INSERT GRAPH HERE: Actual vs. Predicted Values Plot]
A scatter plot comparing the actual AQI values against the predicted values from our best model (KNN). The tight clustering of points along the diagonal line visually confirms the model's high accuracy.

The table below summarizes the performance of each model on the test data:

**Model	Test R² Score	Verdict**
K-Neighbors Regressor	0.847 (84.7%)	✅ Best Performer
Decision Tree Regressor	0.746 (74.6%)	❌ Overfitting
Linear Regression	0.744 (74.4%)	🆗 Good Baseline

Export to Sheets
The K-Nearest Neighbors (KNN) model emerged as the most effective, with an R² score of 0.847. This indicates that the model can successfully explain approximately 85% of the variance in the AQI based on the provided pollutant data, making it a robust and reliable predictor.

## Future Scope
**While the current model performs well, there are several avenues for future improvement:**

  - Advanced Models: Implement more complex models like XGBoost, LightGBM, or CatBoost, which often yield higher accuracy.
  
  - Hyperparameter Tuning: Use techniques like GridSearchCV or RandomizedSearchCV to find the optimal parameters for the KNN model and further boost its performance.
  
  - Additional Features: Incorporate meteorological data (e.g., temperature, humidity, wind speed) as these factors significantly influence air quality.
  
  - Deployment: Deploy the final model as a simple web application using Flask or Streamlit to make it accessible to a wider audience.
