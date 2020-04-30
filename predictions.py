import pandas as pd
from fbprophet import Prophet
from flask import Flask
from flask import jsonify
from timeloop import Timeloop
from datetime import timedelta
import threading
import datetime


app = Flask(__name__)
covid_data = pd.read_csv('http://coronavairus.herokuapp.com/brazil/csv') 


def normalizer_date(date):
    result = ''
    for i in range(10):
        result += date[i]
    return result

def normalizer_results(num):
    result = int(num)
    return result

def to_obj(cases,deaths):

    data_res = []

    for i in range(len(cases['date'].values)):
        data_res.append({ 
        
        'date': str(cases['date'].values[i])[:10], 
        'cases': {
        'lower_prediction': int(cases['lower_prediction'].values[i]), 
        'mean_prediction': int(cases['mean_prediction'].values[i]),
        'high_prediction': int(cases['high_prediction'].values[i]) },
        'deaths': {
        'lower_prediction': int(deaths['lower_prediction'].values[i]), 
        'mean_prediction': int(deaths['mean_prediction'].values[i]),
        'high_prediction': int(deaths['high_prediction'].values[i]) }
        }
        
        )
    
    return data_res


def get_predict(covid_data):
  covid_data['date'] = covid_data.date.apply(normalizer_date)
  covid_data['date'] = pd.to_datetime(covid_data['date'])

  covid_new_deaths = covid_data[['date','newDeaths']]
  covid_new_deaths.columns = ['ds','y']

  covid_new_cases = covid_data[['date','newCases']]
  covid_new_cases.columns = ['ds','y']

  m_deaths = Prophet(interval_width= 0.90)
  m_deaths.fit(covid_new_deaths)

  m_cases = Prophet(interval_width= 0.90)
  m_cases.fit(covid_new_cases)

  predict_deaths = m_deaths.make_future_dataframe(periods=7)
  preview_deaths = m_deaths.predict(predict_deaths)

  predict_cases = m_cases.make_future_dataframe(periods=7)
  preview_cases= m_cases.predict(predict_cases)

  data_deaths = preview_deaths[['ds','yhat_lower','yhat','yhat_upper']].tail(7)
  data_deaths.columns = ['date','lower_prediction','mean_prediction','high_prediction']

  data_cases = preview_cases[['ds','yhat_lower','yhat','yhat_upper']].tail(7)
  data_cases.columns = ['date','lower_prediction','mean_prediction','high_prediction']


  data_deaths['lower_prediction'] = data_deaths.lower_prediction.apply(normalizer_results)
  data_deaths['mean_prediction'] = data_deaths.mean_prediction.apply(normalizer_results)
  data_deaths['high_prediction'] = data_deaths.high_prediction.apply(normalizer_results)

  data_cases['lower_prediction'] = data_cases.lower_prediction.apply(normalizer_results)
  data_cases['mean_prediction'] = data_cases.mean_prediction.apply(normalizer_results)
  data_cases['high_prediction'] = data_cases.high_prediction.apply(normalizer_results)
  
  return {"cases": data_cases, "deaths": data_deaths}


predict_data = get_predict(covid_data)
last_fetch = datetime.datetime.now()

tl = Timeloop()

@tl.job(interval=timedelta(minutes=30))
def update_data():
    global last_fetch
    global predict_data
    global covid_data
    covid_data = pd.read_csv('http://coronavairus.herokuapp.com/brazil/csv') 
    predict_data = get_predict(covid_data)
    last_fetch = datetime.datetime.now()

@app.route("/")
def res():
    global last_data
    global predict_data

    return {'data': to_obj(predict_data['cases'],predict_data['deaths']), 'last_fetch': last_fetch}

@tl.job(interval=timedelta(seconds=0))
def run_server():
    app.run(debug=False)


if __name__ == "__main__":
    tl.start(block=True)
