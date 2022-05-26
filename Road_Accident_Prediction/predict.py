import pickle
import pandas as pd

scalar_dict = {
    'Temperature(F)': 36.9,
    'Distance(mi)': 0.747,
    'Humidity(%)': 91.0,
    'Pressure(in)': 29.68,
    'Visibility(mi)': 10.0,
    'Wind_Speed(mph)': 15.0,
    'Precipitation(in)': 0.02,
    'Start_Lng': -84.0628,
    'Start_Lat': 39.86542,
}

surroundings_map = {
    'Amenity': 0.000000,
    'Bump': 0.000000,
    'Crossing': 0.000000,
    'Give_Way': 0.000000,
    'Junction': 0.000000,
    'No_Exit': 0.000000,
    'Railway': 0.000000,
    'Roundabout': 0.000000,
    'Station': 0.000000,
    'Stop': 0.000000,
    'Traffic_Calming': 0.000000,
    'Traffic_Signal': 0.000000,
}

civiltwilight_map = {'Civil_Twilight_Night': 0.000000}

model = "Logistic_regression_model"


def predict_sevirity(scalar_dict=scalar_dict, Start_Time='2016-02-09 06:10:59', surroundings_map=surroundings_map,
                     weather_given='Smoke', winddir_given='Variable', city_name="Dayton", side_r=0, civiltwilight_map = civiltwilight_map):

    print('---Inputs---')
    print('scalar_dict--{}'.format(scalar_dict))
    print('Start_Time---{}'.format(Start_Time))
    print('surroundings_map--{}'.format(surroundings_map))
    print('weather_given--{}'.format(weather_given))
    print('winddir_given---{}'.format(winddir_given))
    print('city_name--{}'.format(city_name))
    print('side_r--{}'.format(side_r))
    print('civiltwilight_map--{}'.format(civiltwilight_map))
    series1 = pd.Series(scalar_dict)

    # Cast Start_Time to datetime
    Start_Time = pd.to_datetime(Start_Time)
    timedict = {}
    # Extract year, month, weekday and day
    timedict["Year"] = Start_Time.year
    timedict["Month"] = Start_Time.month
    timedict["Weekday"] = Start_Time.weekday()
    timedict["Day"] = Start_Time.day
    # Extract hour and minute
    timedict["Hour"] = Start_Time.hour
    timedict["Minute"] = Start_Time.minute
    series2 = pd.Series(timedict)
    series2

    series12 = pd.Series()

    series12 = series12.append(series1)
    series12 = series12.append(series2)
    series12

    # use the save scaler to preprocess
    scaler = pickle.load(open("minmax_scaler.sav", 'rb'))
    series12 = pd.Series(scaler.transform([series12]).squeeze(), index=list(series12.index.values))
    series12

    series3 = pd.Series(surroundings_map)
    series4 = pd.Series(civiltwilight_map)
    series4

    weather_map = {
        'Weather_Condition_Cloudy': 0.000000,
        'Weather_Condition_Fog': 0.000000,
        'Weather_Condition_Hail': 0.000000,
        'Weather_Condition_Rain': 0.000000,
        'Weather_Condition_Sand': 0.000000,
        'Weather_Condition_Smoke': 0.000000,
        'Weather_Condition_Snow': 0.000000,
        'Weather_Condition_Thunderstorm': 0.000000,
        'Weather_Condition_Windy': 0.000000,
    }

    print('weather_given-----------'+weather_given)

    for weather in weather_map:
        if (weather.split('_')[-1].casefold() == weather_given.casefold()):
            weather_map[weather] = 1.0
    weather_map
    series5 = pd.Series(weather_map)
    series5

    winddir_map = {
        'Wind_Direction_E': 0.000000,
        'Wind_Direction_N': 0.000000,
        'Wind_Direction_NE': 0.000000,
        'Wind_Direction_NW': 0.000000,
        'Wind_Direction_S': 0.000000,
        'Wind_Direction_SE': 0.000000,
        'Wind_Direction_SW': 0.000000,
        'Wind_Direction_Variable': 0.000000,
        'Wind_Direction_W': 0.000000
    }
    for winddir in winddir_map:
        if (winddir.split('_')[-1].casefold() == winddir_given.casefold()):
            winddir_map[winddir] = 1.0
    series6 = pd.Series(winddir_map)
    series6

    data = [city_name]
    city_df = pd.DataFrame(data, columns=['City'])
    binary_encoder = pickle.load(open("city_binary_encoder_model.pkl", 'rb'), fix_imports=True)
    series7 = binary_encoder.transform(city_df['City']).iloc[0]

    Side_R_map = {

        'Side_R': 1
    }
    if side_r == 0:
        Side_R_map['Side_R'] = side_r
    series8 = pd.Series(Side_R_map)

    final = pd.Series()
    final = final.append(series12)
    final = final.append(series3)
    final = final.append(series4)
    final = final.append(series5)
    final = final.append(series6)
    final = final.append(series7)
    final = final.append(series8)
    final = final.reindex(
        index=['Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
               'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
               'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
               'Year', 'Month', 'Weekday', 'Day', 'Hour', 'Minute', 'Weather_Condition_Cloudy', 'Weather_Condition_Fog',
               'Weather_Condition_Hail', 'Weather_Condition_Rain', 'Weather_Condition_Sand', 'Weather_Condition_Smoke',
               'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm', 'Weather_Condition_Windy',
               'Wind_Direction_E', 'Wind_Direction_N', 'Wind_Direction_NE', 'Wind_Direction_NW', 'Wind_Direction_S',
               'Wind_Direction_SE', 'Wind_Direction_SW', 'Wind_Direction_Variable', 'Wind_Direction_W',
               'Civil_Twilight_Night', 'Side_R', 'City_0', 'City_1', 'City_2', 'City_3', 'City_4', 'City_5', 'City_6',
               'City_7', 'City_8', 'City_9', 'City_10', 'City_11', 'City_12'])
    lr = pickle.load(open("{}.sav".format(model), 'rb'), fix_imports=True)

    severity = lr.predict([final])
    print(final)
    print(severity)
    return severity


#print(predict_sevirity())
