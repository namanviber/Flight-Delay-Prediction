import streamlit as st
import pandas as pd 
import requests
import pickle

Operating_Airline= ["American Airlines", "Delta Air Lines", "American Eagle Airlines", "United Airlines", "Southwest Airlines"]
Origin = ["Hartsfield-Jackson Atlanta International Airport", "Charlotte Douglas International Airport", "Denver International Airport", "Dallas/Fort Worth International Airport", "George Bush Intercontinental Airport", "Los Angeles International Airport", "Chicago O'Hare International Airport", "Phoenix Sky Harbor International Airport", "San Francisco International Airport"]
Dest = ["Hartsfield-Jackson Atlanta International Airport", "Charlotte Douglas International Airport", "Denver International Airport", "Dallas/Fort Worth International Airport", "George Bush Intercontinental Airport", "Los Angeles International Airport", "Chicago O'Hare International Airport", "Phoenix Sky Harbor International Airport", "San Francisco International Airport"]

airports = {
  "Hartsfield-Jackson Atlanta International Airport": "ATL",
  "Charlotte Douglas International Airport": "CLT",
  "Denver International Airport": "DEN",
  "Dallas/Fort Worth International Airport": "DFW",
  "George Bush Intercontinental Airport": "IAH",
  "Los Angeles International Airport": "LAX",
  "Chicago O'Hare International Airport": "ORD",
  "Phoenix Sky Harbor International Airport": "PHX",
  "San Francisco International Airport": "SFO"
}

airlines = {
  "American Airlines": "AA",
  "Delta Air Lines": "DL",
  "American Eagle Airlines": "OO", 
  "United Airlines": "UA",
  "Southwest Airlines": "WN"
}


data_pivot = {
   'origin': ['ATL', 'ATL', 'ATL', 'ATL', 'ATL', 'ATL', 'ATL', 'CLT', 'CLT', 'CLT', 'CLT', 'CLT', 'CLT', 'CLT', 'DEN', 'DEN', 'DEN', 'DEN', 'DEN', 'DEN', 'DEN', 'DFW', 'DFW', 'DFW', 'DFW', 'DFW', 'DFW', 'IAH', 'IAH', 'IAH', 'IAH', 'IAH', 'IAH', 'LAX', 'LAX', 'LAX', 'LAX', 'LAX', 'LAX', 'LAX', 'ORD', 'ORD', 'ORD', 'ORD', 'ORD', 'ORD', 'ORD', 'PHX', 'PHX', 'PHX', 'PHX', 'PHX', 'PHX', 'PHX', 'SFO', 'SFO', 'SFO', 'SFO', 'SFO', 'SFO', 'SFO'],
   'dest': ['CLT', 'DEN', 'DFW', 'IAH', 'LAX', 'ORD', 'PHX', 'ATL', 'DEN', 'DFW', 'IAH', 'LAX', 'ORD', 'PHX', 'ATL', 'CLT', 'DFW', 'IAH', 'LAX', 'ORD', 'PHX', 'ATL', 'CLT', 'DEN', 'IAH', 'LAX', 'ORD', 'ATL', 'CLT', 'DEN', 'DFW', 'LAX', 'ORD', 'ATL', 'CLT', 'DEN', 'DFW', 'IAH', 'ORD', 'PHX', 'ATL', 'CLT', 'DEN', 'DFW', 'IAH', 'LAX', 'PHX', 'ATL', 'CLT', 'DEN', 'DFW', 'IAH', 'LAX', 'SFO', 'ATL', 'CLT', 'DEN', 'DFW', 'IAH', 'LAX', 'ORD'],
   'distance': [226.0, 1199.0, 731.0, 689.0, 1947.0, 606.0, 1587.0, 226.0, 1337.0, 936.0, 912.0, 2125.0, 599.0, 1773.0, 1199.0, 1337.0, 641.0, 862.0, 862.0, 888.0, 602.0, 731.0, 936.0, 641.0, 224.0, 1235.0, 801.0, 689.0, 912.0, 862.0, 224.0, 1379.0, 925.0, 1947.0, 2125.0, 862.0, 1235.0, 1379.0, 1744.0, 370.0, 606.0, 599.0, 888.0, 802.0, 925.0, 1744.0, 1440.0, 1587.0, 1773.0, 602.0, 868.0, 1009.0, 370.0, 651.0, 2139.0, 2296.0, 967.0, 1464.0, 1635.0, 337.0, 1846.0]
}

df_pivot = pd.DataFrame(data_pivot)
pivot_table = pd.pivot_table(df_pivot, values='distance', index=['origin'], columns=['dest'], fill_value=0)

filename = "trained_models/rf.pkl"

with open(filename, "rb") as pickle_file:
  model = pickle.load(pickle_file)

airport_codes = {
    'LAX': 'USW00023174',
    'IAH': 'USW00012960',
    'DEN': 'USW00003017',
    'ORD': 'USW00094846',
    'ATL': 'USW00013874',
    'SFO': 'USW00023234',
    'DFW': 'USW00003927',
    'PHX': 'USW00023183',
    'CLT': 'USW00013881'
}

def processResponse(a):
    data = a.text.replace('"', ' ').splitlines()
    data = [line.strip() for line in data]

    header = data[0].split(',')
    header = [line.strip() for line in header]

    rows = [row.split(',') for row in data[1:] if row]
    rows[0] = [line.strip() for line in rows[0]]
    rows[1] = [line.strip() for line in rows[1]]

    df = pd.DataFrame(rows, columns=header)

    columns_to_convert = ['AWND', 'PRCP', 'SNOW', 'TAVG']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df.fillna(0,inplace=True)


    return df

def weather_info(origin,dest,date):

    url = 'https://www.ncei.noaa.gov/access/services/data/v1'

    params = {
        'dataset': 'daily-summaries',
        'stations': f'{origin}, {dest}',
        'dataTypes': 'AWND,PRCP,SNOW,TAVG',
        'startDate': f'{date}',
        'endDate': f'{date}'
        }
    

    response = requests.get(url, params=params)

    if response.status_code == 200:

        df = processResponse(response)
        awnd_o, prcp_o, tavg_o, awnd_d, prcp_d, tavg_d,snow_o, snow_d = df['AWND'][0], df['PRCP'][0], df['TAVG'][0], df['AWND'][1], df['PRCP'][1], df['TAVG'][1], df['SNOW'][0], df['SNOW'][1]
        return awnd_o, prcp_o, tavg_o, awnd_d, prcp_d, tavg_d,snow_o, snow_d
    
    return 0,0,0,0,0,0,0,0


def preprocess_input(date, operating_airline, origin, dest, dep_time, distance):
    quarter = (date.month - 1) // 3 + 1
    month = date.month
    day_of_month = date.day
    day_of_week = date.weekday() + 1

    processed_time = dep_time.hour * 100 + dep_time.minute
    dep_hour_of_day = int(processed_time) // 100

    awnd_o, prcp_o, tavg_o, awnd_d, prcp_d, tavg_d,snow_o, snow_d = weather_info(airport_codes[origin],airport_codes[dest],date)

    format = {
    "Distance": False, "DepHourofDay": False, "AWND_O": False, "PRCP_O": False, "TAVG_O": False, "AWND_D": False, 
    "PRCP_D": False, "TAVG_D": False, "SNOW_O": False, "SNOW_D": False, "Quarter_1": False, "Quarter_2": False, 
    "Quarter_3": False, "Quarter_4": False, "Month_1": False, "Month_2": False, "Month_3": False, "Month_4": False, 
    "Month_5": False, "Month_6": False, "Month_7": False, "Month_8": False, "Month_9": False, "Month_10": False, 
    "Month_11": False, "Month_12": False, "DayofMonth_1": False, "DayofMonth_2": False, "DayofMonth_3": False, 
    "DayofMonth_4": False, "DayofMonth_5": False, "DayofMonth_6": False, "DayofMonth_7": False, "DayofMonth_8": False, 
    "DayofMonth_9": False, "DayofMonth_10": False, "DayofMonth_11": False, "DayofMonth_12": False, "DayofMonth_13": False, 
    "DayofMonth_14": False, "DayofMonth_15": False, "DayofMonth_16": False, "DayofMonth_17": False, "DayofMonth_18": False, 
    "DayofMonth_19": False, "DayofMonth_20": False, "DayofMonth_21": False, "DayofMonth_22": False, "DayofMonth_23": False, 
    "DayofMonth_24": False, "DayofMonth_25": False, "DayofMonth_26": False, "DayofMonth_27": False, "DayofMonth_28": False, 
    "DayofMonth_29": False, "DayofMonth_30": False, "DayofMonth_31": False, "DayOfWeek_1": False, "DayOfWeek_2": False, 
    "DayOfWeek_3": False, "DayOfWeek_4": False, "DayOfWeek_5": False, "DayOfWeek_6": False, "DayOfWeek_7": False, 
    "Operating_Airline _AA": False, "Operating_Airline _DL": False, "Operating_Airline _OO": False, "Operating_Airline _UA": False, 
    "Operating_Airline _WN": False, "Origin_ATL": False, "Origin_CLT": False, "Origin_DEN": False, "Origin_DFW": False, 
    "Origin_IAH": False, "Origin_LAX": False, "Origin_ORD": False, "Origin_PHX": False, "Origin_SFO": False, 
    "Dest_ATL": False, "Dest_CLT": False, "Dest_DEN": False, "Dest_DFW": False, "Dest_IAH": False, "Dest_LAX": False, 
    "Dest_ORD": False, "Dest_PHX": False, "Dest_SFO": False}

    format["Distance"] = distance
    format["DepHourofDay"] = dep_hour_of_day
    format["AWND_O"] = awnd_o
    format["PRCP_O"] = prcp_o
    format["TAVG_O"] = tavg_o
    format["AWND_D"] = awnd_d
    format["PRCP_D"] = prcp_d
    format["TAVG_D"] = tavg_d
    format["SNOW_O"] = snow_o
    format["SNOW_D"] = snow_d
    format[f"Quarter_{quarter}"] = True
    format[f"Month_{month}"] = True
    format[f"DayofMonth_{day_of_month}"] = True
    format[f"DayOfWeek_{day_of_week}"] = True
    format[f"Operating_Airline _{operating_airline}"] = True
    format[f"Origin_{origin}"] = True
    format[f"Dest_{dest}"] = True

    return pd.DataFrame(format, index=[0])


def predict(data):
    pred = model.predict(data.iloc[:, :])

    return pred[0]

# Streamlit Code

st.title("Flight Delay Prediction")

input1 = st.selectbox("Please Select Your Airline", Operating_Airline)
input2 = st.selectbox("Please Select your Origin Airport", Origin)
input3 = st.selectbox("Please Select your Destination Airport", Dest)
date = st.date_input("Please Pick Date of your Journey")
time = st.time_input("Please Select Scheduled Departure Time")

input1 = airlines[f"{input1}"]
input2 = airports[f"{input2}"]
input3 = airports[f"{input3}"]

if st.button("Predict"):

    df = preprocess_input(date,input1,input2,input3,time, pivot_table[input2][input3])

    prediction = predict(df)

    if prediction == 1:
        st.error("Your Flight is Most Likely to be delayed more than 15 minutes")
    elif prediction == 2:
        st.error("Your Flight is Most Likely to be delayed more than 30 minutes")
    elif prediction == 3:
        st.error("Your Flight is Most Likely to be delayed more than 45 minutes")
    elif prediction == 4:
        st.error("Your Flight is Most Likely to be delayed more than 1 hour")
    else:
        st.success("Your flight is likely to be on time")