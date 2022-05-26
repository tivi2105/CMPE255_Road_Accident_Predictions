
# Introduction
This is a countrywide traffic accident dataset, which covers 49 states of the United States. The data is continuously being collected from February 2016, using several data providers, including multiple APIs that provide streaming traffic event data. These APIs broadcast traffic events captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road- networks. Currently, there are about 2.8 million accident records in this dataset. The economic and social impact of traffic accidents cost U.S. citizens hundreds of billions of dollars every year. And a large part of losses is caused by a small number of serious accidents.

![alt text](https://github.com/tivi2105/CMPE255_Road_Accident_Predictions/blob/main/Road_Accident_Severity_Prediction_architecture_diagram.png?raw=true)

# Steps to download and run Road Accident Predictions program
## Step 1:
Go to 'https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents' and download 'US_Accidents_Dec21_updated.csv' file. Place it in 'Road_Accident_Prediction\drive\MyDrive' folder, please create two folders MyDrive in drive folder and drive in Road_Accident_Prediction folder.

## Step 2:
We could not updaload 'Random forest regression' model as the file is more than 100MB, we could not upload it to the github. So, got to 'https://drive.google.com/file/d/1k4gEUQdKvn6qXEPlknp_YsgbevuyxZzX/view?usp=sharing' and download 'RandomForest_model.sav' file and place it in the folder where dash_app.py is located (basically the base folder).

## Step 3:
Now open 'predict.py' and goto line 33 and change the model to one of the following 'minmax_scaler' for minmaxscaler model, 'Logistic_regression_model' for logistic regression, 'RandomForest_model' for random forest, 'Support_vector_machine_model' for SVM or 'city_binary_encoder_model'.

## Step 4:
Now run dash_app.py and go to 'http://127.0.0.1:3000/' and go to 'Predictions' tab, provide inputs and click on submit you can see the severity meter changes from its default value to the predicted value (Please be patient on the first try for the prediction to load as the data set is huge sometimes it might take a while to just load the dataset).
