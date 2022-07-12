# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import pandas as pd, numpy as np, scipy as sp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
populations = pd.read_csv('/kaggle/input/covid19-population-data/population_data.csv')
populations = populations.drop(columns=['Type']).set_index('Name').transpose()
populations = populations.to_dict()


train.columns = ['Id', 'State', 'Country', 'Lat', 'Long', 'Date', 'ConfirmedCases', 'Fatalities']
test.columns = ['ForecastId'] + list(train.columns)[1:-2]






def dS_dt(S, I, alpha1_t, alpha2_t):
    return -alpha1_t*S*I -alpha2_t*S*I

def dE_dt(S, I, E, alpha1_t, alpha2_t, beta):
    return alpha1_t*S*I + alpha2_t*S*I - beta*E

def dI_dt(E, I, beta, gamma, psi):
    return beta*E - gamma*I - psi*I

def dR_dt(I, gamma):
    return gamma*I

def dD_dt(I, psi):
    return psi*I


def ODE_model(t, y, alpha1t, alpha2t, beta, gamma, psi):

    alpha1_t = alpha1t(t)
    alpha2_t = alpha2t(t)
    
    S, E, I, R, D = y
    St = dS_dt(S, I, alpha1_t, alpha2_t)
    Et = dE_dt(S, I, E, alpha1_t, alpha2_t, beta)
    It = dI_dt(E, I, beta, gamma, psi)
    Rt = dR_dt(I, gamma)
    Dt = dD_dt(I, psi)
    return [St, Et, It, Rt, Dt]


def loss(theta, data, population, nforecast=0, error=True):
    alpha1_0, alpha2_0, beta, gamma, psi = theta
    k,L=2.5,25
    Infected_0 = data.ConfirmedCases.iloc[0]
    ndays = nforecast
    ntrain = data.shape[0]
    y0 = [(population-Infected_0)/population, 0, Infected_0/population, 0, 0]
    t_span = [0, ndays] # dayspan to evaluate
    t_eval = np.arange(ndays) # days to evaluate
    
    def a1_t(t):
        return alpha1_0 / (1 + (t/L)**k)

    def a2_t(t):
        return alpha2_0 / (1 + (t/L)**k)

    sol = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 
                                 args = (a1_t, a2_t, beta, gamma, psi))
    
    pred_all = np.maximum(sol.y, 0)
    ccases_pred = np.diff((pred_all[2] + pred_all[3] + pred_all[4])*population, n = 1, prepend = Infected_0).cumsum()
    deaths_pred = pred_all[4]*population
    ccases_act = data.ConfirmedCases.values
    deaths_act = data.Fatalities.values
    
    if ccases_act[-1]<ccases_act[-2]:
        ccases_act[-1]=ccases_act[-2]
    if deaths_act[-1]<deaths_act[-2]:
        deaths_act[-1]=deaths_act[-2]
    
    weights =  np.exp(np.arange(data.shape[0])/10)/np.exp((data.shape[0]-1)/10) 

    ccases_rmse = np.sqrt(mean_squared_error(ccases_act, ccases_pred[0:ntrain], sample_weight=weights))
    ccases_penalty = 0.0*np.abs(ccases_act[-1] - ccases_pred[0:ntrain][-1])
    deaths_rmse = np.sqrt(mean_squared_error(deaths_act, deaths_pred[0:ntrain], sample_weight=weights))
    deaths_penalty = 0.0*np.abs(deaths_act[-1] - deaths_pred[0:ntrain][-1])

    ccases_loss = ccases_rmse + ccases_penalty
    deaths_loss = deaths_rmse + deaths_penalty
    
    loss = np.mean((ccases_loss, deaths_loss))
    
    if error == True:
        return loss
    else:
        return loss, ccases_pred, deaths_pred



train['location'] = train['State'].fillna(train['Country'])
test['location'] = test['State'].fillna(test['Country'])
locations=list(train['location'].drop_duplicates())

valid = train[train['Date'] >= test['Date'].min()]
train = train[train['Date'] < test['Date'].min()]

train.set_index(['location', 'Date'], inplace=True)
valid.set_index(['location', 'Date'], inplace=True)
test.set_index(['location', 'Date'], inplace=True)

submission['ConfirmedCases'] = 0
submission['Fatalities'] = 0

parms0 = [1.5, 1.5, 0.5, 0.05, 0.001]
bnds = ((0.001, None), (0.001, None), (0, 10), (0, 10), (0, 10))

def fit_ODE_model(location, plot_fit = False, update_submission=False):
        
    train_data = train.loc[location].query('ConfirmedCases > 0')
    valid_data = valid.loc[location]
    dat = train_data.append(valid_data)
    
    test_data = test.loc[location]
    
    nforecast = len(train_data)+len(test_data)
    
    population = populations[location]['Population']

    n_infected = train_data['ConfirmedCases'].iloc[0]
        
    res = sp.optimize.minimize(fun = loss, x0 = parms0, 
                               args = (dat, population, nforecast),
                               method='L-BFGS-B', bounds=bnds)
    
    dates_all = train_data.index.append(test_data.index)
    dates_val = train_data.index.append(valid_data.index)
    
    err, ccases_pred, deaths_pred = loss(theta = res.x, data = dat, population = population, nforecast=nforecast, error=False)
    
    predictions = pd.DataFrame({'ConfirmedCases': ccases_pred,
                                'Fatalities': deaths_pred}, index=dates_all)
    
    y_pred_test = predictions.iloc[len(train_data):]
    
    if plot_fit:
        train_true = dat[['ConfirmedCases',  'Fatalities']]
        predictions.columns = ['ConfirmedCases_pred',  'Fatalities_pred']

        plot_df = pd.merge(predictions,train_true,how='left', left_index=True, right_index=True)
        
        plt.plot(plot_df.ConfirmedCases_pred.values, color='green',linestyle='--', linewidth=0.5)
        plt.plot(plot_df.Fatalities_pred.values, color='blue',linestyle='--', linewidth=0.5)
        plt.plot(plot_df.Fatalities.values, color='red')
        plt.plot(plot_df.ConfirmedCases.values, color='orange')

        plt.show()        
    
    if update_submission:
        forecast_ids = test_data['ForecastId']
        submission.loc[forecast_ids, ['ConfirmedCases', 'Fatalities']] = y_pred_test.values
        
        

for location in locations:
    if location != 'Nepal':
        try:
            fit_ODE_model(location, plot_fit = False, update_submission=True)
            print(location+' complete.')
        except:
            print(location+ ' error.')
    else:
        print('Nepal error.')

submission.ConfirmedCases = submission.ConfirmedCases.astype(np.float32)
submission.loc[submission.ConfirmedCases<0,'ConfirmedCases']=0
submission.Fatalities = submission.Fatalities.astype(np.float32)
submission.loc[submission.Fatalities<0,'Fatalities']=0


submission.to_csv('submission.csv',index=False)










