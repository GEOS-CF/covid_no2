#!/bin/python

# requirements
import argparse
import numpy as np
import datetime as dt
import os
import sys
import glob
import pandas as pd
import logging
import difflib
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Parameter
DROPVARS = ['location','original_station_name','lat','lon','conc_unit','year']
EMISSCAL = 1.0e6*3600.0
CONCSCAL = 1.0e9

def main(args):
    '''
    Calculate NO2 anomalies (relative to business as usual) for year 2020 based on
    historical model-observation comparisons.  
    '''
    # set up logger
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
#---Read data
    # Read OpenAQ NO2 observations 
    allobs = _read_obs(args)
    # Read model output. This contains model-simulated NO2 plus a suite of other weather and chemistry parameter 
    allmod = _read_model(args)
#---Compute NO2 anomalies for each location separately
    anomalies = [] 
    for s in allobs.location.unique():
        # subset data
        obs = allobs.loc[allobs['location']==s].copy() 
        model = allmod.loc[allmod['location']==s].copy() 
#-------Train bias-correction model using XGBoost
        # Prepare training data 
        Xall, Yall, _ = _prepare_training_data(args,obs,model)
        Xtrain, Xvalid, Ytrain, Yvalid =  train_test_split( Xall, Yall, test_size=0.5 )
        # Train model
        bst = _train(args,Xtrain,Ytrain) 
        # Validate 
        if args.validate==1:
            _valid(args,s,bst,Xvalid,Yvalid) 
#-------Apply bias correction to model output to obtain 'business-as-usual' estimate and compare this value against observations
        pred = _apply_bias(args,bst,obs,model)
        pred['station'] = [s for i in range(pred.shape[0])]
        anomalies.append(pred)
#---Plot results agreggated over all sites
    no2diff = pd.concat(anomalies)
    _make_timeseries(args,no2diff,title=args.location)
    return


def _read_obs(args):
    '''read OpenAQ observations'''
    log = logging.getLogger(__name__)
    log.info('Reading {}'.format(args.obsfile))
    obsdat = pd.read_csv(args.obsfile,parse_dates=['ISO8601'],date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    allobs = obsdat.loc[(obsdat['obstype']=='no2')&(~np.isnan(obsdat['conc_obs']))]
    return allobs


def _read_model(args):
    '''read model output'''
    log = logging.getLogger(__name__)
    SKIPVARS = ['ISO8601','location','lat','lon','CLDTT','Q10M','T10M','TS','U10M','V10M','ZPBL','year','month','day','hour']
    log.info('Reading {}'.format(args.modfile))
    model = pd.read_csv(args.modfile,parse_dates=['ISO8601'],date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    # scale concentrations to ppbv (from mol/mol) and emissions to mg/m2/h (from kg/m2/s)
    for v in model:
        if v in SKIPVARS:
            continue 
        if v == 'TPREC':
            scal = 1.0e6
        elif v == 'PS': 
            scal = 0.01 
        elif 'EMIS_' in v:
            scal = EMISSCAL 
        else:
            scal = CONCSCAL
        model[v] = model[v].values * scal 
    return model 


def _prepare_training_data(args,obs,model,mindate=None,maxdate=dt.datetime(2020,1,1),trendday=dt.datetime(2018,1,1)):
    '''prepare training and validation data'''
    log = logging.getLogger(__name__)
    obsl = obs.copy()
    _ = [obsl.pop(var) for var in ['location','lat','lon','original_station_name','obstype']]
    Xall = obsl.merge(model,how='inner',on='ISO8601')
    if mindate is not None:
        Xall = Xall.loc[Xall['ISO8601']>mindate]
    if maxdate is not None:
        Xall = Xall.loc[Xall['ISO8601']<maxdate]
    Xall['weekday'] = [i.weekday() for i in Xall['ISO8601']]
    if trendday is not None:
        Xall['trendday'] = [(i-trendday).days for i in Xall['ISO8601']]
    # target is observation - model difference
    Yall = Xall['conc_obs'] - Xall['NO2']
    # drop values not needed
    _ = [Xall.pop(var) for var in DROPVARS if var in Xall]
    conc_obs = Xall.pop('conc_obs')
    return Xall,Yall,conc_obs


def _valid(args,station,bst,Xvalid,Yvalid):
    '''make prediction using XGboost model'''
    log = logging.getLogger(__name__)
    bias,conc,dates = _predict(args,bst,Xvalid)
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    axs[0] = _plot_scatter(axs[0],bias,Yvalid,-60.,60.,'Predicted bias [ppbv]','True bias [ppbv]','Bias')
    axs[1] = _plot_scatter(axs[1],Xvalid['NO2'],Xvalid['NO2'].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Original')
    axs[2] = _plot_scatter(axs[2],conc,Xvalid['NO2'].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Adjusted (XGBoost)')
    _ = fig.suptitle(station,y=0.98)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    ofile = 'validation_'+station+'.png' 
    fig.savefig(ofile)
    log.info('Scatter plot written to {}'.format(ofile))
    plt.close()
    return


def _predict(args,bst,Xpredict):
    '''make prediction using XGBoost model and return predicted bias and bias-corrected concentration'''
    log = logging.getLogger(__name__)
    Xp = Xpredict.copy()
    dates = Xp.pop('ISO8601')
    predict = xgb.DMatrix(Xp)
    predicted_bias = bst.predict(predict)
    predicted_conc = Xpredict['NO2'].values + predicted_bias    
    return predicted_bias, predicted_conc, dates


def _train(args,Xtrain,Ytrain):
    '''train XGBoost model'''
    log = logging.getLogger(__name__)
    Xt = Xtrain.copy()
    Xt.pop('ISO8601')
    train = xgb.DMatrix(Xt,np.array(Ytrain))
    params = {'booster':'gbtree'}
    log.info('Training XGBoost model...')
    bst = xgb.train(params,train)
    return bst


def _apply_bias(args,bst,obs,model):
    '''apply bias-correction to model and compare against observations'''
    log = logging.getLogger(__name__)
    Xall, Yall, conc_obs = _prepare_training_data(args,obs,model,mindate=dt.datetime(2018,1,1),maxdate=None)
    bias_pred, conc_pred, dates = _predict(args,bst,Xall)
    anomaly = conc_obs - conc_pred
    return pd.DataFrame({'ISO8601':dates,'predicted':conc_pred,'observed':conc_obs,'anomaly':anomaly})


def _make_timeseries(args,anomalies,rolling=21,mindate=dt.datetime(2019,1,1),title='unknown'):
    '''plot time series of NO2 anomalies'''
    log = logging.getLogger(__name__)
    dat = anomalies.set_index('ISO8601').resample('1D').mean().rolling(window=rolling,min_periods=1).mean().reset_index()
    dat['percent_deviation'] = 100.0*(dat['observed']-dat['predicted'])/dat['predicted'] 
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(1,1,1)
    plt.axhline(0.0,color='darkgrey',linestyle='dashed')
    ax.plot(dat.ISO8601,dat['percent_deviation'],color='teal',linewidth=3,label='NO$_{2}$')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax.set_xlim([mindate,dt.datetime(2021,1,1)])
    ax.set_ylabel('NO$_{2}$ anomaly from baseline')
    plt.title(title)
    plt.xticks(rotation=45)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    ofile = 'no2_anomalies_'+title.replace(' ','')+'.png'
    plt.savefig(ofile)
    plt.close()
    log.info('Figure written to {}'.format(ofile))
    return


def _plot_scatter(ax,x,y,minval,maxval,xlab,ylab,title):
    '''make scatter plot of XGBoost prediction vs. true values'''
    log = logging.getLogger(__name__)
    r,p = stats.pearsonr(x,y)
    nrmse = np.sqrt(mean_squared_error(x,y))/np.std(x)
    mb   = np.sum(y-x)/np.sum(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    ax.hexbin(x,y,cmap=plt.cm.gist_earth_r,bins='log')
    ax.set_xlim(minval,maxval)
    ax.set_ylim(minval,maxval)
    ax.plot((0.95*minval,1.05*maxval),(0.95*minval,1.05*maxval),color='grey',linestyle='dashed')
    # regression line
    ax.plot((0.95*minval,1.05*maxval),(intercept+(0.95*minval*slope),intercept+(1.05*maxval*slope)),color='blue',linestyle='dashed')
    ax.set_xlabel(xlab)
    if ylab != '-':
        ax.set_ylabel(ylab)
    istr = 'N = {:,}'.format(y.shape[0])
    _ = ax.text(0.05,0.95,istr,transform=ax.transAxes)
    istr = '{0:.2f}'.format(r**2)
    istr = 'R$^{2}$ = '+istr
    _ = ax.text(0.05,0.90,istr,transform=ax.transAxes)
    istr = 'NRMSE [%] = {0:.2f}'.format(nrmse*100)
    _ = ax.text(0.05,0.85,istr,transform=ax.transAxes)
    _ = ax.set_title(title)
    return ax


def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-o','--obsfile',type=str,help='observation file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/obs_NewYork.csv')
    p.add_argument('-m','--modfile',type=str,help='model file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/model_NewYork.csv')
    p.add_argument('-l','--location',type=str,help='location name',default='New York')
    p.add_argument('-v','--validate',type=int,help='make validation figures?',default=1)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

