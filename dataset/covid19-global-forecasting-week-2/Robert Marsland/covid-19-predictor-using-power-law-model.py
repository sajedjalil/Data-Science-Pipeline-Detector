import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
        
#Define global parameters
start_cutoff=4 #Initial estimate of t0 is when the number of cases exceeds this number
p0=5e2 #For confirmed cases model, only include times after the number of cases crosses this point
Delta0 = 5 #Initial estimate of time lag between t0 and first fatality
r0 = 0.1 #Initial estimate of death rate
region_exceptions = ['Japan','Holy See','Diamond Princess','Greenland','Korea, South','China','South Africa','Ecuador','Syria'] #Regions that have saturated or have other issues
subregion_exceptions = ['Missouri','Wisconsin','North Carolina','Quebec','New South Wales'] #Subregions that have saturated or have other issues
highz_regions = ['Australia','Japan','Germany','Canada','United Kingdom','France','Iceland'] #Regions with big exponents, need different initial conditions
z0low = 4.5 #This is a good initial exponent estimate for most countries
f0low = 50 #This is a good fatality cutoff to get rid of the initial noise
z0high = 9 #A higher initial estimate is required for the regions in the "highz_regions" list
f0high = 10 #For these regions, there have not yet been many fatalities, so we reduce the cutoff

#Load the data
data_global = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
data_global.columns = ['Id','subregion','state','date','positive','death']
data_global['date'] = pd.to_datetime(data_global['date'],format='%Y-%m-%d')
tref = data_global['date'].iloc[0]
data_global['elapsed'] = (data_global['date'] - tref)/timedelta(days=1)
data_global = data_global.fillna(value='NaN')

#Make lookup table for prediction ID's
id_list = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv',index_col=0)
id_list['Date'] = pd.to_datetime(id_list['Date'],format='%Y-%m-%d')
id_list['elapsed'] = (id_list['Date'] - tref)/timedelta(days=1)
id_list = id_list.fillna(value='NaN')
id_list = id_list.reset_index().set_index(['Country_Region','Province_State','elapsed'])
id_list = id_list.sort_index()

#Load file to hold output
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv',index_col=0)

#Define cost functions and Jacobian
def cost_p(params,data):
    t0,C,z = params
    prediction = np.log(C)+z*np.log(data.index.values-t0)
    
    return 0.5*((np.log(data.values)-prediction)**2).sum()
def cost_f(params,data,p_params):
    t0,C,z = p_params
    Delta,r = params
    prediction = np.log(r*C)+z*np.log(data.index.values-t0-Delta)
    
    return 0.5*((np.log(data.values)-prediction)**2).sum()
def jac_p(params,data):
    t0,C,z = params
    prediction = np.log(C)+z*np.log(data.index.values-t0)
    
    return np.asarray([((z/(data.index.values-t0))*(np.log(data.values)-prediction)).sum(),
                       -((1/C)*(np.log(data.values)-prediction)).sum(),
                      -(np.log(data.index.values-t0)*(np.log(data.values)-prediction)).sum()])
def jac_f(params,data,p_params):
    t0,C,z = p_params
    Delta,r = params
    prediction = np.log(r*C)+z*np.log(data.index.values-t0-Delta)
    
    return np.asarray([((z/(data.index.values-t0-Delta))*(np.log(data.values)-prediction)).sum(),
                       -((1/r)*(np.log(data.values)-prediction)).sum()])

#########Train the model and get predictions#######

#Set up the variables
p_valid = True
f_valid = True
p = data_global.pivot_table(index='elapsed',values='positive',columns=['state','subregion'],aggfunc=np.sum)
f = data_global.pivot_table(index='elapsed',values='death',columns=['state','subregion'],aggfunc=np.sum)
params_table = pd.DataFrame(columns=['Country_Region','State_Province','t0','t0_abs','C','z','Delta t','r'])
param_id = 0

#Loop through regions
for region in set(id_list.reset_index()['Country_Region']):
    #These regions need different initial conditions for optimizer to converge nicely
    if region in highz_regions:
        z0 = z0high
        f0 = f0high
    #These initial conditions work well everywhere else
    else:
        z0 = z0low
        f0 = f0low
    p_region = p.T.loc[region].T
    f_region = f.T.loc[region].T
    
    #Now loop through the "subregions" (states or provinces)
    for subregion in p_region.keys():
        p_train = p_region[subregion]
        f_train = f_region[subregion]
        
        #Only use places where the number of cases eventually exceeds twice the minimum threshold, and where there are at least three data points
        #South Korea and China have already saturated, so I'm going to keep the estimate at the current number of cases
        #South Africa and Ecuador aren't working, and I haven't tracked down the problem yet.
        if np.max(p_train)>2*p0 and np.sum(p_train>p0)>3 and region not in region_exceptions and subregion not in subregion_exceptions:
            t00 = p_train.loc[p_train>=start_cutoff].index.values[0]
            C0 = p_train.max()/(np.max(p_train.index.values)-t00)**z0
            p_train = p_train.loc[p_train>p0]
            out = minimize(cost_p,[t00,C0,z0],args=(p_train,),jac=jac_p,bounds=((None,int(p_train.index.values[0])-1),(1e-6,None),(0,10)))
            t0,C,z = out.x
            p_valid = True #out.success
        else:
            p_valid = False

        #If the spreading model was successfully trained, now try to learn the fatality rate and time delay
        if np.max(f_train)>2*f0 and p_valid and np.sum(f_train>f0)>2:
            f_train = f_train.loc[f_train>f0]
            Delta_max = np.min([12,np.min(f_train.index.values)-t0-.5])
            out = minimize(cost_f,[Delta0,r0],args=(f_train,[t0,C,z]),jac=jac_f,bounds=((0,Delta_max),(1e-6,1)))
            Delta,r = out.x
            f_valid = True #out.success
        else:
            f_valid = False

        #Record the data
        if p_valid:
            params_table.loc[param_id,'Country_Region']=region
            params_table.loc[param_id,'State_Province']=subregion
            params_table.loc[param_id,'t0'] = t0
            params_table.loc[param_id,'t0_abs'] = (timedelta(days=t0)+tref).isoformat()[:10]
            params_table.loc[param_id,'C'] = C
            params_table.loc[param_id,'z'] = z
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]-1
                submission.loc[pred_id,'ConfirmedCases'] = C*((t-t0)**z)
        else:
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]-1
                if t in p_train.index.values:
                    submission.loc[pred_id,'ConfirmedCases'] = p_train.loc[t]
                else:
                    submission.loc[pred_id,'ConfirmedCases'] = np.max(p_train)
                
        if f_valid:
            params_table.loc[param_id,'Delta t'] = Delta
            params_table.loc[param_id,'r'] = r
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]-1
                submission.loc[pred_id,'Fatalities'] = r*C*((t-t0-Delta)**z)
        else:
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]-1
                if t in f_train.index.values:
                    submission.loc[pred_id,'Fatalities'] = f_train.loc[t]
                else:
                    submission.loc[pred_id,'Fatalities'] = np.max(f_train)
                
        param_id += 1
        
#Save the data and parameters
#submission.to_csv('submission.csv')

###########Make predictions for regions with insufficient data, based on global averages#########
params_table = params_table.set_index(['Country_Region','State_Province']).sort_index()
zlist = params_table['z']
rlist = params_table['r']
Dellist = params_table['Delta t']
z = zlist.mean()
r0 = rlist.mean()
Delta = Dellist.mean()
for region in set(id_list.reset_index()['Country_Region']):
    p_region = p.T.loc[region].T
    f_region = f.T.loc[region].T
    if region in highz_regions:
        z0 = z0high
        f0 = f0high
    else:
        z0 = z0low
        f0 = f0low
    
    #Now loop through the "subregions" (states or provinces)
    for subregion in p_region.keys():
        p_train = p_region[subregion]
        f_train = f_region[subregion]
        z = zlist.mean()
        
        #Find all the regions that did not meet our criteria before (and that are not on the excluded list)
        if not(np.max(p_train)>2*p0 and np.sum(p_train>p0)>3) and region not in region_exceptions and subregion not in subregion_exceptions:
            #Estimate start time if possible
            if (p_train>start_cutoff).sum()>1:
                t0 = p_train.loc[p_train>=start_cutoff].index.values[0]
                C = p_train.max()/(np.max(p_train.index.values)-t0)**z
            #If not enough cases to estimate start time, assume infection is contained
            else:
                t0 = -80
                z = 0
                C = p_train.max()
            p_valid = False 
        else:
            p_valid = True

        #Find all the regions that did not meet our criteria before (and that are not on the excluded list)
        if not(np.max(f_train)>2*f0 and p_valid and np.sum(f_train>f0)>2) and region not in region_exceptions and subregion not in subregion_exceptions:
            #If there is spread and there are fatalities, estimate fatality rate from data
            if (region, subregion) in params_table.index.tolist():
                z = params_table['z'].loc[region,subregion]
                t0 = params_table['t0'].loc[region,subregion]
                C = params_table['C'].loc[region,subregion]
                t0_abs = (timedelta(days=t0)+tref).isoformat()[:10]
            if f_train.max() > 1 and z>0 and np.max(f_train.index.values)-t0-Delta > 0:
                r = f_train.max()/(C*(np.max(f_train.index.values)-t0-Delta)**z)
            else:
                r = r0
            f_valid = False 
        else:
            f_valid = True

        #Save the data
        if not p_valid:
            t0_abs = (timedelta(days=t0)+tref).isoformat()[:10]
            new_params = pd.DataFrame(np.asarray([t0,t0_abs,C,z,np.nan,np.nan])[np.newaxis,:],index=pd.MultiIndex.from_tuples([(region,subregion)]),columns=['t0','t0_abs','C','z','Delta t','r'])
            params_table = params_table.append(new_params)
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]-1
                if t > t0:
                    submission.loc[pred_id,'ConfirmedCases'] = C*((t-t0)**z)
                else:
                    submission.loc[pred_id,'ConfirmedCases'] = 0
                
        if not f_valid:
            params_table.loc[region,subregion] = np.asarray([t0,t0_abs,C,z,Delta,r])
            for t in id_list.loc[region].reset_index()['elapsed']:
                pred_id = id_list['ForecastId'].loc[region,subregion,t]-1
                if t-t0-Delta > 0:
                    submission.loc[pred_id,'Fatalities'] = r*C*((t-t0-Delta)**z)
                else:
                    submission.loc[pred_id,'Fatalities'] = 0
                
#Save the data and parameters
submission = submission.fillna(value=0)
submission.to_csv('submission.csv')
params_table.to_csv('params.csv')