import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import category_encoders as ce
import pickle

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

#Read data from the CSV file
df = pd.read_csv("drive/MyDrive/US_Accidents_Dec21_updated.csv")
df.head()

#copy data to another variable
X = df.copy()
X.head()

# Convert Start_Time to Year, Month, Day, Hour, Minute and add back to the dataframe
X["Start_Time"] = pd.to_datetime(X["Start_Time"])
X["Year"] = X["Start_Time"].dt.year
X["Month"] = X["Start_Time"].dt.month
X["Weekday"] = X["Start_Time"].dt.weekday
X["Day"] = X["Start_Time"].dt.day
X["Hour"] = X["Start_Time"].dt.hour
X["Minute"] = X["Start_Time"].dt.minute

# Drop Irrelevent features
features_to_drop = ["ID", "Start_Time", "End_Time", "End_Lat", "End_Lng", "Description", "Number", "Street", "County", "State", "Zipcode", "Country", "Timezone", "Airport_Code", "Weather_Timestamp", "Wind_Chill(F)", "Turning_Loop", "Sunrise_Sunset", "Nautical_Twilight", "Astronomical_Twilight"]
X = X.drop(features_to_drop, axis=1)
X.head()
print("Number of rows:", len(X.index))
X.drop_duplicates(inplace=True)
print("Number of rows after drop of duplicates:", len(X.index))

# Remove empty and 0 values
X = X[X["Side"] != " "]
X = X[X["Pressure(in)"] != 0]
X = X[X["Visibility(mi)"] != 0]


#Change weather feature values to the below
X.loc[X["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
X.loc[X["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
X.loc[X["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
X.loc[X["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
X.loc[X["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
X.loc[X["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
X.loc[X["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
X.loc[X["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
X.loc[X["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
X.loc[X["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
X.loc[X["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan

print(X["Weather_Condition"].unique())

#Change wind direction paramenter to simpler values
X.loc[X["Wind_Direction"] == "CALM", "Wind_Direction"] = "Calm"
X.loc[X["Wind_Direction"] == "VAR", "Wind_Direction"] = "Variable"
X.loc[X["Wind_Direction"] == "East", "Wind_Direction"] = "E"
X.loc[X["Wind_Direction"] == "North", "Wind_Direction"] = "N"
X.loc[X["Wind_Direction"] == "South", "Wind_Direction"] = "S"
X.loc[X["Wind_Direction"] == "West", "Wind_Direction"] = "W"

X["Wind_Direction"] = X["Wind_Direction"].map(lambda x : x if len(x) != 3 else x[1:], na_action="ignore")
X["Wind_Direction"].unique()

#Fill the empty features with mean values
features_to_fill = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
X[features_to_fill] = X[features_to_fill].fillna(X[features_to_fill].mean())
X.dropna(inplace=True)


# Check for the imbalance in the dataset
severity_counts = X["Severity"].value_counts()

plt.figure(figsize=(10, 8))
plt.title("Histogram for the severity")
sns.barplot(severity_counts.index, severity_counts.values)
plt.xlabel("Severity")
plt.ylabel("Value")
plt.show()

# dataset is highly unbalanced and subset having Severity 2, Sevirity 1 have the highest and lowest severity respectively
# Change the dataset such that there are same no of records for each severity level(Acheived by undersampling the dataset)
size = len(X[X["Severity"]==1].index)
df = pd.DataFrame()
for i in range(1,5):
    S = X[X["Severity"]==i]
    df = df.append(S.sample(size, random_state=42))
X = df


scaler = MinMaxScaler()
features = ['Temperature(F)','Distance(mi)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Start_Lng','Start_Lat', 'Year', 'Month','Weekday','Day','Hour','Minute']
X[features] = scaler.fit_transform(X[features])
X.head()
# Save the scaler for preprocessing before inference
filename = 'minmax_scaler.sav'
pickle.dump(scaler, open(filename, 'wb'))


categorical_features = set(["Side", "City", "Wind_Direction", "Weather_Condition", "Civil_Twilight"])
for cat in categorical_features:
    X[cat] = X[cat].astype("category")
X.info()

print("Unique classes for each categorical feature:")
for cat in categorical_features:
    print("{:15s}".format(cat), "\t", len(X[cat].unique()))

#Replace True/False values on the dataset with 1 and 0
X = X.replace([True, False], [1, 0])
X.head()

# Remove city because it will be encoded later
onehot_cols = categorical_features - set(["City"])
X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)
X.head()

# Binary encode the City values in the dataset to obtain 11 binary encoded columns where each column represents one bit.
# We do this because there are around 6000 cities in the dataset
binary_encoder = ce.binary.BinaryEncoder()
city_binary_enc = binary_encoder.fit_transform(X["City"])

# Save the binary encoder model to use during preprocessing before inference
with open('city_binary_encoder_model.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(binary_encoder, file)

X = pd.concat([X, city_binary_enc], axis=1).drop("City", axis=1)



#######################################################################################################################################
####################################################################################################################################

# Training

# Set validation dictionaries to be filled for each model
accuracy = dict()
f1 = dict()

def logisticRegression(X):
    # Set validation dictionaries to be filled for each model

    # Split the dataset to test and train dataset
    X, X_test = train_test_split(X, test_size=.2, random_state=42)
    print(X.shape, X_test.shape)
    sample = X

    # Drop the label column
    y_sample = sample["Severity"]
    X_sample = sample.drop("Severity", axis=1)

    X_train, X_validate, y_train, y_validate = train_test_split(X_sample, y_sample, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_validate.shape, y_validate.shape)

    lr = LogisticRegression(random_state=42, n_jobs=-1)
    params = {"solver": ["newton-cg", "sag", "saga"]}

    # Hyperparameter tuning
    grid = GridSearchCV(lr, params, n_jobs=-1, verbose=5)
    grid.fit(X_train, y_train)

    # Model Evaluation
    print("Best parameters scores:")
    print(grid.best_params_)
    print("Train score:", grid.score(X_train, y_train))
    print("Validation score:", grid.score(X_validate, y_validate))

    print("Default scores:")
    lr.fit(X_train, y_train)
    print("Train score:", lr.score(X_train, y_train))
    print("Validation score:", lr.score(X_validate, y_validate))

    y_pred = lr.predict(X_validate)

    accuracy["Logistic Regression"] = accuracy_score(y_validate, y_pred)
    f1["Logistic Regression"] = f1_score(y_validate, y_pred, average="macro")

    print(classification_report(y_train, lr.predict(X_train)))
    print(classification_report(y_validate, y_pred))

    y_pred = lr.predict(X_validate)
    confmat = confusion_matrix(y_true=y_validate, y_pred=y_pred)

    index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
    columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    filename = 'Logistic_regression_model.sav'
    pickle.dump(lr, open(filename, 'wb'))

    print(X_validate.columns.values.tolist())

def supportVectorMachine():
    parameters = [{"kernel": ["linear", "rbf", "sigmoid"], "C": [.2, .5, .8, 1.]},
                  {"kernel": ["poly"], "C": [.2, .5, .8, 1.], "degree": [2, 3, 4]}]
    svc = svm.SVC(verbose=5, random_state=42)
    grid = GridSearchCV(svc, parameters, verbose=5, n_jobs=-1)

    sample = X.sample(5_000, random_state=42)
    y_sample = sample["Severity"]
    X_sample = sample.drop("Severity", axis=1)
    grid.fit(X_sample, y_sample)

    print("Best parameters scores:")
    print(grid.best_params_)
    print("Train score:", grid.score(X_sample, y_sample))

    print("Default scores:")
    svc.fit(X_sample, y_sample)
    print("Train score:", svc.score(X_sample, y_sample))

    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score")

    sample = X.sample(10_000, random_state=42)
    y_sample = sample["Severity"]
    X_sample = sample.drop("Severity", axis=1)

    X_train, X_validate, y_train, y_validate = train_test_split(X_sample, y_sample, test_size=.2, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_validate.shape, y_validate.shape)

    svc = svm.SVC(**grid.best_params_, random_state=42)
    svc.fit(X_train, y_train)

    print("Train score:", svc.score(X_train, y_train))
    print("Validation score:", svc.score(X_validate, y_validate))

    y_pred = svc.predict(X_validate)

    accuracy["SVM"] = accuracy_score(y_validate, y_pred)
    f1["SVM"] = f1_score(y_validate, y_pred, average="macro")

    print(classification_report(y_train, svc.predict(X_train)))
    print(classification_report(y_validate, y_pred))
    y_pred = svc.predict(X_validate)
    confmat = confusion_matrix(y_true=y_validate, y_pred=y_pred)

    index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
    columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix - Support Vector Machine")
    plt.show()

    filename = 'Support_vector_machine_model.sav'
    pickle.dump(svc, open(filename, 'wb'))

def decisionTreeeClassifier():
    sample = X
    y_sample = sample["Severity"]
    X_sample = sample.drop("Severity", axis=1)

    X_train, X_validate, y_train, y_validate = train_test_split(X_sample, y_sample, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_validate.shape, y_validate.shape)

    dtc = DecisionTreeClassifier(random_state=42)
    parameters = [{"criterion": ["gini", "entropy"], "max_depth": [15]}]
    grid = GridSearchCV(dtc, parameters, verbose=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best parameters scores:")
    print(grid.best_params_)
    print("Train score:", grid.score(X_train, y_train))
    print("Validation score:", grid.score(X_validate, y_validate))

    print("Default scores:")
    dtc.fit(X_train, y_train)
    print("Train score:", dtc.score(X_train, y_train))
    print("Validation score:", dtc.score(X_validate, y_validate))

    y_pred = dtc.predict(X_validate)

    accuracy["Decision Tree"] = accuracy_score(y_validate, y_pred)
    f1["Decision Tree"] = f1_score(y_validate, y_pred, average="macro")

    print(classification_report(y_train, dtc.predict(X_train)))
    print(classification_report(y_validate, y_pred))

    y_pred = dtc.predict(X_validate)
    confmat = confusion_matrix(y_true=y_validate, y_pred=y_pred)

    index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
    columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dtc, max_depth=4, fontsize=10, feature_names=X_train.columns.to_list(), class_names=True, filled=True)
    plt.show()

    filename = 'DecisionTree_model.sav'
    pickle.dump(dtc, open(filename, 'wb'))

def randomForestClassifier():
    sample = X
    y_sample = sample["Severity"]
    X_sample = sample.drop("Severity", axis=1)

    X_train, X_validate, y_train, y_validate = train_test_split(X_sample, y_sample, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_validate.shape, y_validate.shape)

    rfc = RandomForestClassifier(random_state=42)
    parameters = [{"n_estimators": [30, 40], "max_depth": [ 15, 30]}]
    grid = GridSearchCV(rfc, parameters, verbose=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best parameters scores:")
    print(grid.best_params_)
    print("Train score:", grid.score(X_train, y_train))
    print("Validation score:", grid.score(X_validate, y_validate))

    print("Default scores:")
    rfc.fit(X_train, y_train)
    print("Train score:", rfc.score(X_train, y_train))
    print("Validation score:", rfc.score(X_validate, y_validate))

    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score")

    y_pred = rfc.predict(X_validate)

    accuracy["Random Forest"] = accuracy_score(y_validate, y_pred)
    f1["Random Forest"] = f1_score(y_validate, y_pred, average="macro")

    print(classification_report(y_train, rfc.predict(X_train)))
    print(classification_report(y_validate, y_pred))

    y_pred = rfc.predict(X_validate)
    confmat = confusion_matrix(y_true=y_validate, y_pred=y_pred)

    index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
    columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    filename = 'RandomForest_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))





randomForestClassifier()


