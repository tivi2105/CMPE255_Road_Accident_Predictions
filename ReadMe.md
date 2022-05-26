
#Introduction
This is a countrywide traffic accident dataset, which covers 49 states of the United States. The data is continuously being collected from February 2016, using several data providers, including multiple APIs that provide streaming traffic event data. These APIs broadcast traffic events captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road- networks. Currently, there are about 2.8 million accident records in this dataset. The economic and social impact of traffic accidents cost U.S. citizens hundreds of billions of dollars every year. And a large part of losses is caused by a small number of serious accidents. Reducing traffic accidents, especially serious accidents, is nevertheless always an important challenge. The proactive approach, one of the two main approaches for dealing with traffic safety problems, focuses on preventing potential unsafe road conditions from occurring in the first place. For the effective implementation of this approach, accident prediction and severity prediction are critical. If we can identify the patterns of how these serious accidents happen and the key factors, we might be able to implement well-informed actions and better allocate financial and human resources. Features like weather, traffic volume, road conditions, time of the day of previous accidents are utilized from the dataset. Machine learning algorithms like Logistic Regression, Decision Tree, Neural Networks and random forest classifiers are used and their results are compared to provide the best prediction.

# Steps to download and run Road Accident Predictions program
###Step 1:
Go to 'https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents' and download 'US_Accidents_Dec21_updated.csv' file. Place it in 'Road_Accident_Prediction\drive\MyDrive' folder, please create two folders MyDrive in drive folder and drive in Road_Accident_Prediction folder.

###Step 2:
We could not updaload 'Random forest regression' model as the file is more than 100MB, we could not upload it to the github. So, got to 'https://drive.google.com/file/d/1k4gEUQdKvn6qXEPlknp_YsgbevuyxZzX/view?usp=sharing' and download 'RandomForest_model.sav' file and place it in the folder where dash_app.py is located (basically the base folder).

###Step 3:
Now open 'predict.py' and goto line 33 and change the model to one of the following ''
