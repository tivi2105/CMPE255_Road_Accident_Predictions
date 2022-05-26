import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
from datetime import date
import plotly.graph_objects as go
from Road_Accident_Prediction.predict import predict_sevirity
import base64
import numpy as np
from PIL import Image

df = pd.read_csv('Road_Accident_Prediction/drive/MyDrive/US_Accidents_Dec21_updated.csv')
#df = pd.read_csv('TestData.csv')
df.Start_Time = pd.to_datetime(df.Start_Time)
df.End_Time = pd.to_datetime(df.End_Time)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

amenities = {}

intro_text = '''Around the world, road safety is a major concern. The number of casualties is rising at a 
                rate of more than 4% per year across all age categories. The percentage of fatalities due to 
                traffic accidents is expected to increase by 8% through 2030. Allowing civilians to die in traffic 
                accidents is both acceptable and regrettable. As a result, an in-depth investigation is required to 
                deal with this type of circumstance. In this project, we attempt to address this issue by 
                developing a tool that can anticipate accident risk, allowing users to make more informed 
                decisions regarding their travel routes. This software gives you information on the severity 
                of an accident at a certain area based on parameters like time and weather. This program provides 
                drivers with a dashboard where they can view the city's accident rate. 
                This is a countrywide car accident dataset, which covers 49 states of the USA. 
                The accident data are collected from February 2016 to December 2020, using multiple APIs 
                that provide streaming traffic incident (or event) data. These APIs broadcast traffic data 
                captured by a variety of entities, such as the US and state departments of transportation, 
                law enforcement agencies, traffic cameras, and traffic sensors within the road-networks. 
                Currently, there are about 1.5 million accident records in this dataset. Check here to 
                learn more about this dataset.'''

intro_text1 = '''This is a countrywide traffic accident dataset, which covers 49 states of the United States. 
              The data is continuously being collected from February 2016, using several data providers, including multiple 
              APIs that provide streaming traffic event data. These APIs broadcast traffic events captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road- networks. Currently, there are about 2.8 million accident records in this dataset.
              The economic and social impact of traffic accidents cost U.S. citizens hundreds of billions of dollars every year. 
              And a large part of losses is caused by a small number of serious accidents. Reducing traffic accidents, especially 
              serious accidents, is nevertheless always an important challenge. The proactive approach, one of the two 
              main approaches for dealing with traffic safety problems, focuses on preventing potential unsafe road conditions 
              from occurring in the first place. For the effective implementation of this approach, accident prediction and 
              severity prediction are critical. If we can identify the patterns of how these serious accidents happen and the 
              key factors, we might be able to implement well-informed actions and better allocate financial and human resources. 
              Features like weather, traffic volume, road conditions, time of the day of previous accidents are utilized from the 
              dataset. Machine learning algorithms like Logistic Regression, Decision Tree, Neural Networks and random forest 
              classifiers are used and their results are compared to provide the best prediction.
              '''

dataset_text = '''This is a nationwide car accident dataset that spans 49 states in the United States. The accident data was acquired using different APIs that give streaming traffic incident (or event) data from February 2016 to December 2020. These APIs transmit traffic data recorded by a range of entities within the road-networks, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors. This dataset currently contains approximately 1.5 million accident records.
              '''

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("MENU", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Predictions", href="/predict", active="exact"),
                dbc.NavLink("Dashboard", href="/page-1", active="exact"),
                dbc.NavLink("About", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('Road Accidents Severity Prediction',
                        style={'textAlign':'center', 'color': 'blue', 'text-decoration': 'underline'}, className='p-3'),
                html.H1('',
                        style={'textAlign':'center'}, className='p-3'),
                #dcc.Graph(id='bargraph',
                #         figure=px.bar(df, barmode='group', x='Years',
                #         y=['Girls Kindergarten', 'Boys Kindergarten'])),
                get_content_for_home_page()
                ]
    elif pathname == "/page-1":
        return [
                html.H1('Road Accidents Data',
                        style={'textAlign':'center'}),
                #dcc.Graph(id='bargraph',
                #         figure=px.bar(df, barmode='group', x='Years',
                #         y=['Girls Kindergarten', 'Boys Kindergarten'])),
                get_graphs_for_home_page()
                ]
        #return [
        #        html.H1('Grad School in Iran',
        #                style={'textAlign':'center'}),
        #        dcc.Graph(id='bargraph',
        #                 figure=px.bar(df, barmode='group', x='Years',
        #                 y=['Girls Grade School', 'Boys Grade School']))
        #        ]
    elif pathname == "/page-2":
        return [
                html.H1('Contributions',
                        style={'textAlign':'center', 'color': 'blue'}),
                #dcc.Graph(id='bargraph',
                #         figure=px.bar(df, barmode='group', x='Years',
                #         y=['Girls High School', 'Boys High School']))
                get_content_for_about_page()
                ]
    elif pathname == "/predict":
        return [
                html.H1('Predictions', style={'color': 'red', 'textAlign':'center'}),
                #get_input_box(),
                #get_input('test1', 'Test2', 'Test2'),
                get_inputs_for_predictions(),
                get_submit_button(),
                get_severity_graph()
                ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )




def get_input_box():
    return html.Div([
        html.Div([html.Label(['Input: '],style={'font-weight': 'bold', "text-align": "center"})],
                 style={'padding-right': '5px'}),
        dcc.Input(
            id='my_txt_input',
            type='text',
            debounce=True,
            pattern=r"^[A-Za-z].*",
            spellCheck=True,
            inputMode='latin',
            name='text',
            list='browser',
            n_submit=0,
            n_submit_timestamp=-1,
            autoFocus=True,
            n_blur=0,
            n_blur_timestamp=-1,
        ),
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

def get_input(box_type, field_name, label_name, value):
    return html.Div([
        html.Div([html.Label([label_name+': '],style={'font-weight': 'bold', "text-align": "center"})],
                 style={'padding-right': '5px'}),
        dcc.Input(
            id='{}_id'.format(field_name),
            type=box_type,
            debounce=True,
            min=-100,
            max=100,
            minLength=0,
            maxLength=10,
            autoComplete='on',
            disabled=False,
            readOnly=False,
            required=False,
            size="20",
            value=value
        ),
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'},
    className='p-3')

def get_slider(box_type, field_name, label_name, min, max, value):
    return html.Div([
        html.Div([html.Label([label_name+': '],style={'font-weight': 'bold', "text-align": "center"})],
                 style={'padding-right': '5px'}),
        dcc.Input(
            id='{}_id'.format(field_name),
            type='range',
            min=min,
            max=max,
            value=value,
            step=0.02
        ),
        html.Div(id='{}_output'.format(field_name), children='', className='p-3')
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'})

def get_graphs_for_home_page():
    city_df = pd.DataFrame(df['City'].value_counts()).reset_index().rename(columns={'index':'City', 'City':'Cases'})
    state_df = pd.DataFrame(df['State'].value_counts()).reset_index().rename(columns={'index':'State', 'State':'Cases'})
    top_10_cities = pd.DataFrame(city_df.head(10))
    year_df = pd.DataFrame(df.Start_Time.dt.year.value_counts()).reset_index().rename(columns={'index':'Year', 'Start_Time':'Cases'}).sort_values(by='Cases', ascending=True)
    weather_df = df.head(200)

    fig = go.Figure(data = go.Choropleth(
    locations = state_df['State'], # spatial coordinates
    z = state_df['Cases'].astype(int), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'hot_r',
    showscale = True,
    colorbar_title = 'Number Of Cases',
    ))
    fig.update_layout(
        title_text = 'US Road Accident Cases By State', 
        geo_scope='usa',
        )

    pie_fig = px.pie(weather_df,names='Weather_Condition')
    pie_fig.update_traces(textinfo='percent+label')
    pie_fig.update_layout(title={'text':'Raod Accident cases in various Weather Condition',
                          'font':{'size':28},'x':0.5,'xanchor':'center'})
    pie_fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    bar_fig = px.bar(year_df, x='Year', y='Cases')
    bar_fig.update_layout(title={'text':'Yearly Road Accident Cases',
                          'font':{'size':28},'x':0.5,'xanchor':'center'})
    bar_fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    c_bar_fig = px.bar(df, x=top_10_cities['City'], y=top_10_cities['Cases'], labels=dict(x="City", y="Cases"))
    c_bar_fig.update_layout(title={'text':'Road Accident Cases City wise',
                          'font':{'size':28},'x':0.5,'xanchor':'center'})
    c_bar_fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    return html.Div([
            html.Div([
                    dcc.Graph(id='bargraph',
                         figure=fig),
                ]),
            html.Div([
                dcc.Graph(id='bargraph',
                         figure=c_bar_fig),
                dcc.Graph(id='linegraph', 
                         figure=bar_fig),
                ], style={'display': 'flex'}),
            html.Div([
                dcc.Graph(figure=pie_fig),
                ]),
        ], style={'align-items': 'center', 'justify-content': 'center'})

def get_inputs_for_predictions():
    return html.Div([
            get_slider('number', 'temp', 'Temperature(F)', -100, 100, value='30.9'),
            get_input('number', 'distance', 'Distance(mi)', value='1.585'),
            get_slider('number', 'humidity', 'Humidity(%)', 0, 100, value='92'),
            get_input('number', 'pressure', 'Pressure(in)', value='29.26'),
            get_slider('number', 'visibility', 'Visibility(mi)', 0, 100, value='1'),
            get_input('number', 'wind_speed', 'Wind Speed(mph)', value='19.6'),
            get_slider('number', 'precipitation', 'Precipitation(in)', 0, 100, value='0.06'),
            get_input('number', 'start_lang', 'Longitude', value='-81.71182'),
            get_input('number', 'start_lat', 'Latitude', value='41.47461'),
            html.Div([
                html.Div([html.Label(['Date: '],style={'font-weight': 'bold', "text-align": "center"})],
                              style={'padding-right': '5px'}),
                    dcc.DatePickerSingle(id='date_selected', date=date(2016, 2, 9)),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'}),
            html.Div([
                html.Div([html.Label(['City: '],style={'font-weight': 'bold', "text-align": "center"})],
                     style={'padding-right': '5px'}),
                
                dcc.Input(id='city', type='text')
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'}),
            html.Div([
                    html.Div([html.Label(['Day/Night: '],style={'font-weight': 'bold', "text-align": "center"})],
                              style={'padding-right': '5px'}),
                    dcc.RadioItems(id="day_night",
                            options = [
                                {'label': 'Day', 'value': '0'},
                                {'label': 'Night', 'value': '1'}
                            ],
                            value = '0',
                            className="p-3",
                            style={'display': 'flex', 'flex-direction': 'column'}
                        )
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            html.Div([
                    html.Div([html.Label(['Right/Left: '],style={'font-weight': 'bold', "text-align": "center"})],
                              style={'padding-right': '5px'}),
                    dcc.RadioItems(id="right_left",
                            options = [
                                {'label': 'Right', 'value': '1'},
                                {'label': 'Left', 'value': '0'}
                            ],
                            value = '0',
                            className="p-3",
                            style={'display': 'flex', 'flex-direction': 'column'}
                        )
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'}),
            html.Div([
                html.Div([html.Label(['Wind Direction: '],style={'font-weight': 'bold', "text-align": "center"})],
                     style={'padding-right': '5px'}),
                dcc.Dropdown(id='wind_direction', 
                    options=[
                        {'label': 'East', 'value': 'E'},
                        {'label': 'West', 'value': 'W'},
                        {'label': 'North', 'value': 'N'},
                        {'label': 'South', 'value': 'S'},
                        {'label': 'N-East', 'value': 'NE'},
                        {'label': 'N-West', 'value': 'NW'},
                        {'label': 'S-East', 'value': 'SE'},
                        {'label': 'S-West', 'value': 'SW'},
                        {'label': 'Variable', 'value': 'Variable'}
                    ],
                    optionHeight=35,
                    value='W',
                    style={'width': '100%'}
                    )
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'}),
             html.Div([
                html.Div([html.Label(['Weather Condition: '],style={'font-weight': 'bold', "text-align": "center"})],
                     style={'padding-right': '5px'}),
                dcc.Dropdown(id='weather_condition', 
                    options=[
                        {'label': 'Cloudy', 'value': 'Cloudy'},
                        {'label': 'Fog', 'value': 'Fog'},
                        {'label': 'Hail', 'value': 'Hail'},
                        {'label': 'Rain', 'value': 'Rain'},
                        {'label': 'Sand', 'value': 'Sand'},
                        {'label': 'Smoke', 'value': 'Smoke'},
                        {'label': 'Snow', 'value': 'Snow'},
                        {'label': 'Thunderstorm', 'value': 'Thunderstorm'},
                        {'label': 'Windy', 'value': 'Windy'}
                    ],
                    optionHeight=35,
                    value='Cloudy',
                    style={'width': '100%'}
                    )
            ], style={'display': 'flex', 'align-items': 'center',
                      'justify-content': 'center', 'padding-top': '5px', 'min-widht': '50px'}),
             html.Div([
                html.Div([html.Label(['Amenities: '],style={'font-weight': 'bold', "text-align": "center"})],
                     style={'padding-right': '5px'}),
                html.Div([
                    html.Div([
                        dcc.Checklist(id='Bump',
                        options=[
                            {'label': 'Bump', 'value': 'Bump'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Crossing',
                        options=[
                            {'label': 'Crossing', 'value': 'Crossing'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Give_Way',
                        options=[
                            {'label': 'Give_Way', 'value': 'Give_Way'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Junction',
                        options=[
                            {'label': 'Junction', 'value': 'Junction'},
                        ], className='p-3'
                        )
                    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                html.Div([
                    dcc.Checklist(id='No_Exit', 
                        options=[
                            {'label': 'No_Exit', 'value': 'No_Exit'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Railway', 
                        options=[
                            {'label': 'Railway', 'value': 'Railway'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Roundabout', 
                        options=[
                            {'label': 'Roundabout', 'value': 'Roundabout'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Station', 
                        options=[
                            {'label': 'Station', 'value': 'Station'},
                        ], className='p-3'
                        ),
                    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                html.Div([
                    dcc.Checklist(id='Stop', 
                        options=[
                            {'label': 'Stop', 'value': 'Stop'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Traffic_Calming', 
                        options=[
                            {'label': 'Traffic_Calming', 'value': 'Traffic_Calming'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='Traffic_Signal', 
                        options=[
                            {'label': 'Traffic_Signal', 'value': 'Traffic_Signal'},
                        ], className='p-3'
                        ),
                        dcc.Checklist(id='amenity_id', 
                        options=[
                            {'label': 'Amenity', 'value': 'Amenity'},
                        ], className='p-3'
                        )
                    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
                    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 
                              'justify-content': 'center', 'border-style': 'solid'}),
                html.Div(id='amenities_output', children='')
            ], style={'display': 'flex', 'align-items': 'center',
                      'justify-content': 'center', 'padding-top': '5px', 'min-widht': '50px'})
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'})

def get_submit_button():
    return html.Div([html.Div([
                    html.Button('Submit', id='submit_val', n_clicks=0),
                    ]),
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-top': '5px'})

def get_severity_graph():
    return html.Div([
                        dcc.Graph(id = 'output_graph',
                            figure=go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = 0,
            mode = "gauge+number+delta",
            title = {'text': "Severity"},
            delta = {'reference': 0},
            gauge = {'axis': {'range': [None, 4]},
            'steps' : [
            {'range': [0, 2], 'color': "lightgray"},
            {'range': [2, 3], 'color': "gray"},
            {'range': [3, 4], 'color': "orangered"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0}})))
                    ])    

@app.callback(
Output('output_graph', 'figure'),
Input('submit_val', 'n_clicks'),
[State('temp_id', 'value'), State('day_night', 'value'),
 State('distance_id', 'value'), State('humidity_id', 'value'), State('pressure_id', 'value'),
 State('visibility_id', 'value'), State('wind_speed_id', 'value'), State('precipitation_id', 'value'),
 State('start_lang_id', 'value'), State('start_lat_id', 'value'), State('city', 'value'), 
 State('wind_direction', 'value'), State('weather_condition', 'value'), State('right_left', 'value'),
 State('date_selected', 'date')]
)
def update_output(value, temp_id, day_night, distance_id, humidity_id, pressure_id,
                  visibility_id, wind_speed_id, precipitation_id, start_lang_id, start_lat_id,
                  city_name, wind_direction, weather_condition, right_left, date_selected):
    scalar_dict = {
        'Temperature(F)': temp_id,
        'Distance(mi)': distance_id,
        'Humidity(%)': humidity_id,
        'Pressure(in)': pressure_id,
        'Visibility(mi)': visibility_id,
        'Wind_Speed(mph)': wind_speed_id,
        'Precipitation(in)': precipitation_id,
        'Start_Lng': start_lang_id,
        'Start_Lat': start_lat_id,
    }
    surroundings_map = {
        'Amenity': amenities['amenity_id'],
        'Bump': amenities['Bump'],
        'Crossing': amenities['Crossing'],
        'Give_Way': amenities['Give_Way'],
        'Junction': amenities['Junction'],
        'No_Exit': amenities['No_Exit'],
        'Railway': amenities['Railway'],
        'Roundabout': amenities['Roundabout'],
        'Station': amenities['Station'],
        'Stop': amenities['Stop'],
        'Traffic_Calming': amenities['Traffic_Calming'],
        'Traffic_Signal': amenities['Traffic_Signal'],
    }
    civiltwilight_map = {'Civil_Twilight_Night': day_night}
    weather_given = weather_condition
    winddir_given = wind_direction
    side_r = right_left
    Start_Time = date_selected+' 17:12:00'
    severity = predict_sevirity(scalar_dict, Start_Time,  surroundings_map,
                     weather_given, winddir_given, city_name, side_r, civiltwilight_map)
    print(severity.squeeze())
    figure=go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = severity[0],
            mode = "gauge+number+delta",
            title = {'text': "Severity"},
            delta = {'reference': 0},
            gauge = {'axis': {'range': [None, 4]},
            'steps' : [
            {'range': [0, 2], 'color': "lightgray"},
            {'range': [2, 3], 'color': "gray"},
            {'range': [3, 4], 'color': "orangered"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': severity[0]}}))
    return figure
    

@app.callback(
Output('amenities_output', 'children'),
[Input('Bump', 'value'), Input('Crossing', 'value'), Input('Give_Way', 'value'),
 Input('Junction', 'value'), Input('No_Exit', 'value'), Input('Railway', 'value'),
 Input('Roundabout', 'value'), Input('Station', 'value'), Input('Stop', 'value'),
 Input('Traffic_Calming', 'value'), Input('Traffic_Signal', 'value'), Input('amenity_id', 'value')]
)
def add_amenity(Bump, Crossing, Give_Way, Junction, No_Exit, Railway,
                Roundabout, Station, Stop, Traffic_Calming, Traffic_Signal, amenity_id):
    amenities['Bump'] = 0 if Bump == None else len(Bump)
    amenities['Crossing'] = 0 if Crossing == None else len(Crossing)
    amenities['Give_Way'] = 0 if Give_Way == None else len(Give_Way)
    amenities['Junction'] = 0 if Junction == None else len(Junction)
    amenities['No_Exit'] = 0 if No_Exit == None else len(No_Exit)
    amenities['Railway'] = 0 if Railway == None else len(Railway)
    amenities['Roundabout'] = 0 if Roundabout == None else len(Roundabout)
    amenities['Station'] = 0 if Station == None else len(Station)
    amenities['Stop'] = 0 if Stop == None else len(Stop)
    amenities['Traffic_Calming'] = 0 if Traffic_Calming == None else len(Traffic_Calming)
    amenities['Traffic_Signal'] = 0 if Traffic_Signal == None else len(Traffic_Signal)
    amenities['amenity_id'] = 0 if amenity_id == None else len(amenity_id)
    print("value: {}".format(amenities))

def get_content_for_home_page():
    img = np.array(Image.open('Road_Accident_Severity_Prediction_architecture_diagram.png'))
    fig = px.imshow(img, color_continuous_scale="gray")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    intro_img = np.array(Image.open('background.png'))
    intro_fig = px.imshow(intro_img, color_continuous_scale="gray")
    intro_fig.update_layout(coloraxis_showscale=False)
    intro_fig.update_xaxes(showticklabels=False)
    intro_fig.update_yaxes(showticklabels=False)


    return html.Div([
                html.Div([
                        #dcc.Graph(figure=intro_fig, style={'width': '100%'}),
                        html.H2('Introduction', style={'color': 'blue'}),
                        html.P(intro_text1,
                            style={'font-weight': '0.5', 'font-size': '20px'}),
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
                html.Div([
                        html.H2('DataSet (US Accident Analysis)', style={'color': 'blue'}),
                        html.P(dataset_text, style={'font-weight': '0.5', 'font-size': '20px'}),
                        html.A('Check here to learn more about this dataset', href='https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents',
                            style={'font-weight': 'bold', 'font-size': '20px', 'color': 'red'}),
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
                html.Div([
                        html.H2('Approach', style={'color': 'blue'}),
                        html.P('Data cleaning was first performed to detect and handle corrupt or missing records. EDA (Exploratory Data Analysis) and feature engineering were then done over most features. Finally, Logistic regression, Support Vector Classifier, Decision Tree, Random Forest Classifier, and MLP classifier were used to develop the predictive model. Out of all the models the one which has best accuracy is chosen and saved to be used in our web-based dash application for predictions.',
                            style={'font-weight': '0.5', 'font-size': '20px'}),
                        
                        dcc.Graph(figure=fig, style={'width': '150vh', 'height': '110vh'}),
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
            ], style={'display': 'flex', 'flex-direction': 'column'})

def get_content_for_about_page():
    return html.Div([
                html.Div([
                        html.H2('Contributers', style={'color': 'blue'}),
                        html.Table([
                                html.Tr([
                                        html.Th([html.Label(['Name'], style={'font-size': '20px', 'color': 'blue'})], style={'border': '1px solid', 'textAlign': 'center'}),
                                        html.Th([html.Label(['Student ID'], style={'font-size': '20px', 'color': 'blue'})], style={'border': '1px solid', 'textAlign': 'center'}),
                                        html.Th([html.Label(['Email'], style={'font-size': '20px', 'color': 'blue'})], style={'border': '1px solid', 'textAlign': 'center'}),
                                    ]),
                                html.Tr([
                                        html.Th([html.Label(['Yashwanth Reddy Samala'])], style={'border': '1px solid', 'textAlign': 'center'}),
                                        html.Th([html.Label(['016014583'])], style={'border': '1px solid', 'textAlign': 'center'}),
                                        html.Th([html.Label(['yashwanthreddy.samala@sjsu.edu'])], style={'border': '1px solid', 'textAlign': 'center'}),
                                    ]),
                                html.Tr([
                                        html.Th([html.Label(['Trivikram Thopugunta'])], style={'border': '1px solid', 'textAlign': 'center'}),
                                        html.Th([html.Label(['015955602'])], style={'border': '1px solid', 'textAlign': 'center'}),
                                        html.Th([html.Label(['trivikram.thopugunta@sjsu.edu'])], style={'border': '1px solid', 'textAlign': 'center'}),
                                    ]),
                            ])
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
                html.Div([
                        html.H2('Project Componets', style={'color': 'blue'}),
                        get_content_for_components(),
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
                html.Div([
                        html.H2('Exploratory and Data Analysis', style={'color': 'blue'}),
                        html.P("Exploratory data analysis (EDA) has been widely utilized in research, with literature describing how to do early investigations on datasets using various graphical representations and statistical techniques. EDA is a well-known approach for examining datasets in order to reveal hidden patterns and answer certain crucial questions (Martinez et al., 2010). The goal of EDA is to learn about the dataset's context so that an appropriate prediction model may be developed. The EDA method can be used to find key variables, outliers, and anomalies in a dataset (Martinez et al., 2010). The non-graphical univariate and multivariate approaches largely require the generation of summary statistics, whereas the graphical univariate and multivariate methods use some graphical ways to summarize, analyze, and show the dataset (Chambers, 2018; DuToit et al., 2012). Furthermore, univariate approaches focus on two or more variables at the same time to identify their relationship, whereas multivariate methods focus just on two variables, or can grow to more than two variables in some circumstances. EDA is a best practice that can be used in a variety of fields, including anomaly detection, speech recognition, and fraud detection.",
                            style={'font-weight': '0.5', 'font-size': '20px'}),
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
                html.Div([
                        html.H2('Creating and Training models', style={'color': 'blue'}),
                        html.P('In our project we developed total five models using "Logistic Regression", "Minmax Scaler", "MLP", "RandomForest Regression", "Support Vector Machine". We created and trained these five models using sklearn python library, out of these 5 models we found that "Random Forest regression" performed well and has high accuracy compared to other models. We also used hiper parameter training for training our models. Finally, we saved our best model integrated into our dash web application, where users can provide various inputs governing most road accidents like weather conditions, temperature, etc.',
                            style={'font-weight': '0.5', 'font-size': '20px'}),
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
                html.Div([
                        html.H2('Creating dashboard using "Dash" in python', style={'color': 'blue'}),
                        html.P('For creating a dashboard website we used "Dash" library in python. Dash is an open-source Python library for developing interactive web applications. Dash is an excellent tool for generating a graphical user interface for our data analysis and for allowing people to experiment with our dashboard application. The best part about Dash is that we can create these things entirely in Python. Without writing any HTML or Javascript code, we can develop web components.',
                            style={'font-weight': '0.5', 'font-size': '20px'}),
                    ], style={'display': 'flex','flex-direction': 'column'}, className='p-3'),
            ], style={'display': 'flex', 'flex-direction': 'column'})

def get_content_for_components():
    return html.Div([
            html.P('In our project there are three major components', style={'font-weight': '0.5', 'font-size': '20px'}),
            html.P('1. Exploratory and Data Analysis', style={'font-weight': '0.5', 'font-size': '20px'}),
            html.P('2. Creating and Training models', style={'font-weight': '0.5', 'font-size': '20px'}),
            html.P('3. Creating dashboard using "Dash" in python', style={'font-weight': '0.5', 'font-size': '20px'}),
            
        ], style={'className': 'p-3'})

@app.callback(
    Output('temp_output', 'children'),
    Input('temp_id', 'value'))
def update_slider_text(value):
    return value

@app.callback(
    Output('humidity_output', 'children'),
    Input('humidity_id', 'value'))
def update_slider_text(value):
    return value

@app.callback(
    Output('visibility_output', 'children'),
    Input('visibility_id', 'value'))
def update_slider_text(value):
    return value

@app.callback(
    Output('precipitation_output', 'children'),
    Input('precipitation_id', 'value'))
def update_slider_text(value):
    return value


if __name__=='__main__':
    app.run_server(debug=True, port=3000)



#def get_inputs_for_predictions():
#    return html.Div([
#            dcc.Slider(0, 20, value=10, id='my-slider'),
#            html.Div(id='slider-output-container')
#        ])
#
#@app.callback(
#    Output('slider-output-container', 'children'),
#    Input('my-slider', 'value'))
#def update_output(value):
#    return 'You have selected "{}"'.format(value)