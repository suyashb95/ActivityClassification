'''
GUI to make real time predictions 
and calculate calory consumption
'''

from kivy.support import install_twisted_reactor
install_twisted_reactor()

from twisted.internet import reactor
from twisted.internet import protocol
from kivy.app import App
from kivy.uix.label import Label
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget 
from kivy.clock import Clock
import pandas as pd
import os, pickle, socket
import numpy as np
import scipy.stats as stats
from statsmodels.robust.scale import mad as mediad
from sklearn.naive_bayes import GaussianNB
from collections import deque


activities = {
    0: 'Stationary',
    1: 'Walking',
    2: 'Running',
}

def get_parameters(data):
    '''
    perform aggregates on 
    rolling windows of 410
    after scaling and normalizing
    This function seems to be faster,
    no unnecessary calculations
    '''
    func_dict = {
        'min': np.min,
        'max': np.max,
        'diff': lambda x: np.max(x) - np.min(x),
        #'mean': np.mean,
        'std': np.std,
        'iqr': stats.iqr,
        'rms': lambda x: np.sqrt(np.mean(np.square(x))),
        #'integral': np.trapz,
        'mad': lambda x: x.mad(),
        'mediad': mediad
    }
    aggregations = {
        'X': func_dict,
        'Y': func_dict,
        'Z': func_dict
    }
    #data[['X', 'Y', 'Z']] = preprocessing.scale(preprocessing.MinMaxScaler().fit_transform(data[['X', 'Y', 'Z']]))
    #data[['X', 'Y', 'Z']] = preprocessing.scale(data[['X', 'Y', 'Z']])
    data['temp'] = 10
    data = data.groupby('temp', as_index=False)
    stats_data = data.agg(aggregations)
    # correlations = data[['X', 'Y', 'Z']].corr().unstack()
    # correlations.columns = correlations.columns.droplevel()
    # correlations.columns = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    # correlations = correlations.drop(['XX', 'YY', 'ZZ', 'YX', 'XZ', 'ZY'], axis=1)
    stats_data.columns = [''.join(col).strip() for col in stats_data.columns.values]
    #tats_data = pd.concat([stats_data], axis=1)
    del stats_data['temp']
    return stats_data

class EchoProtocol(protocol.DatagramProtocol):

    def __init__(self, app):
        self.app = app

    def datagramReceived(self, data, addr):
        self.app.handle_message(data)

class TestWidget(Widget):
    x_reading = ObjectProperty(None)
    y_reading = ObjectProperty(None)
    z_reading = ObjectProperty(None)
    sample_rate = ObjectProperty(None)
    activity_label = ObjectProperty(None)
    calorie_label = ObjectProperty(None)
    calories = 0
    calorie_rate = 0
    X = deque(maxlen=300)
    Y = deque(maxlen=300)
    Z = deque(maxlen=300)
    NBModel = pickle.load(open('NBModel2.pkl', 'rb'))
    count = 0

    def show_sample_rate(self, dt):
        self.sample_rate.text = "{0:.2f}".format((self.count + float(self.sample_rate.text))/2)
        if self.count != 0:
            data_frame = pd.DataFrame({
                        'X': self.X,
                        'Y': self.Y,
                        'Z': self.Z,
                    }
                )
            activity = self.NBModel.predict(
                    get_parameters(data_frame)
                )[0]
            if activity == 1:
                self.calories += 0.064
            if activity == 2:
                self.calories += 0.176
            self.activity_label.text = activities[activity]
        self.count = 0
        self.calorie_rate += 1
        if self.calorie_rate == 5:
            self.calorie_label.text = "{0:.2f}".format((self.calories))
            self.calorie_rate = 0

    def handle_message(self, msg):
        x, y, z = str(msg, 'utf-8').strip().split(',')
        self.x_reading.text = x
        self.y_reading.text = y
        self.z_reading.text = z
        self.X.append(float(x))
        self.Y.append(float(y))
        self.Z.append(float(z))
        self.count += 1

class TwistedServerApp(App):

    def build(self):
        ui = TestWidget()
        reactor.listenUDP(5000, EchoProtocol(ui))
        Clock.schedule_interval(ui.show_sample_rate, 1.0)
        return ui


if __name__ == '__main__':
    TwistedServerApp().run()