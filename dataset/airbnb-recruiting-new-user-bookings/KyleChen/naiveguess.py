

import numpy as np
import pandas as pd

class Booking(object):
    def __init__(self):
        self.df_pred = None
        self.read_data()

    def read_data(self):
        self.df_train = pd.read_csv('../input/train_users.csv')
        self.df_test = pd.read_csv('../input/test_users.csv')
        # self.df_session = pd.read_csv('../input/sessions.csv')
        # self.df_age_gender = pd.read_csv('../input/age_gender_bkts.csv')
        # self.df_countries = pd.read_csv('../input/countries.csv')
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def save_sub(self, filename='../submission/pred.csv'):
        expand_id = [ x for x in self.df_pred.id.tolist() for i in range(5) ]
        expand_country = [ x for country in self.df_pred.country_pred for x in country ]
        df = pd.DataFrame(list(zip(expand_id, expand_country)), columns=['id', 'country'])
        df.to_csv(filename,index=False)

    def run(self, save_filename='pred.csv'):
        self.fit()
        self.predict()
        self.save_sub(save_filename)

class SimpleStat(Booking):
    # 0.85359
    def __init__(self, *args, **kargs):
        super(SimpleStat, self).__init__(*args, **kargs)

    def fit(self):
        country_stat = self.df_train.groupby('country_destination').count().id
        self.pred_country = country_stat.sort_values(ascending=False)[0:5].index.tolist()
        pass

    def predict(self):
        self.df_pred = self.df_test
        self.df_pred['country_pred'] = self.df_pred.id.map(lambda x: self.pred_country)
        pass

def run(booking_model):
    model = booking_model()
    model.run()

def test():
    pass

def main():

    run(SimpleStat)
    # test()

if __name__ == '__main__':
    main()