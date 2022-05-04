# SDA-Project

The objective of this project was to implement the various techniques learned throughout the course (Statistical Data Analysis) to extract information from data. The data which I used is a multivariate time-series. [This report](https://github.com/Prahitha/SDA-Project/blob/main/Group54_S20180010108_Project_Report.pdf) discusses how I approached the data and the different methods used during the analysis and prediction phases. I used the normal test to check for normality, the Durbin Watson Test to check for collinearity, the Granger Causality Test to check for association between the variables, and the Augmented-Dickey Fuller Test to check for stationarity. Furthermore, I observed the features of the data using time series analysis and applied regression techniques like VAR(OLS), and ARIMA to predict future outcomes.

## Dataset Description
<img width="1009" alt="Screenshot 2022-05-04 at 9 52 04 PM" src="https://user-images.githubusercontent.com/44160152/166726289-b318935d-02f1-47e0-9cf5-454695efda9d.png">


## EDA
1. Basic preprocessing steps like adding column names, changing the epochs to DateTime format
2. Drop the columns which don’t add valuable information to the data (like, experiment)
3. Split the data into training and validation sets
4. Identify missing values and verify the quality of the data
5. Determine likely approaches to modeling, which might yield a predictive function

## Results

### Device-1
![image](https://user-images.githubusercontent.com/44160152/166725681-cc7b700b-6535-4e09-833e-26aaf560dc06.jpeg)

### Device-2
![image](https://user-images.githubusercontent.com/44160152/166725739-5db67051-55e0-4e4f-857f-68b9992e879c.jpeg)

### Device-3
![image](https://user-images.githubusercontent.com/44160152/166725801-19660006-1ab9-4b40-8876-230d50bacb52.jpeg)

### Device-4
![image](https://user-images.githubusercontent.com/44160152/166725838-154af872-08c2-4084-93b5-b3d0ee3f52db.jpeg)

