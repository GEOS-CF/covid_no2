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
import time
import cudf
import cupy as cp

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

############
# Parameter
############

# Data columns to be excluded from the machine learning
DROPVARS = ['location','original_station_name','lat','lon','unit','year']

# Scale factors used to scale the model emissions and concentrations, respectively.
EMISSCAL = 1.0e6*3600.0
CONCSCAL = 1.0e9

# Conversion factor used to go from ug/m3 to ppbv. Assumes a pressure of 1 atm and a temperature of 15 degC.
# It's ok if those assumptions are only approximate since the bias-correction should be able to pick up the 
# conversion error (which depends on temperature and pressure, both provided as input featues). 
UG2PPBV  = 0.514 


def main(args):
    '''
    Calculate NO2 anomalies (relative to business as usual) for year 2020 based on historical model-observation comparisons.  
    '''
    # set up logger
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)
    ta = time.perf_counter()
    # Read NO2 observations from ascii file 
    allobs = _read_obs(args)
    # Read model output. This contains model-simulated NO2 plus a suite of other weather and chemistry parameter 
    allmod = _read_model(args)
    tb = time.perf_counter()
    # For every city, train bias-corrector for each available observation site and calculate bias-corrected model
    # predictions (to be compared against actual observations)
    runtimes = []
    dttrain_total = 0.0
    for c in args.cities:
        t0 = time.perf_counter()
        nlocations, dttrain = _train_and_predict(args,allobs,allmod,c)
        t1 = time.perf_counter()
        runtimes.append(pd.DataFrame({'city':[c],'locations':[nlocations],'runtime':[t1-t0]}))
        dttrain_total += dttrain
    # show runtimes
    rtimes = pd.concat(runtimes)
    rtimes['time_per_loc'] = rtimes['runtime']/rtimes['locations']
    log.info('Run times ({}):'.format('GPU' if args.gpu==1 else 'CPU'))
    log.info('Loading data time: {:.2f}s'.format(tb-ta))
    log.info('Run time per city (train, predict, plot):')
    for c in range(rtimes.shape[0]):
        if rtimes['locations'].values[c]>0:
            log.info('- {:20s}: {:.2f} seconds ({:.2f}s / location)'.format(rtimes['city'].values[c],rtimes['runtime'].values[c],rtimes['time_per_loc'].values[c]))   
    if rtimes.shape[0]>0:
        log.info('Average ML training time: {:.2f}s / location'.format(dttrain_total/rtimes['locations'].sum()))
    return


def _train_and_predict(args,allobs,allmod,cityname):
    '''train model and make prediction for a given location'''
    log = logging.getLogger(__name__)
#---Latitude and longitude boundaries of model data
    lats = np.arange(-90.,90.01,0.25)
    lons = np.arange(-180,180.,0.25)
    # sub-select observations that fall within specified lat/lon boundaries
    log.info('Working on city {}:'.format(cityname))
    subobs = allobs.loc[(allobs['CityKey']==cityname)]
    locations = list(subobs['stationID'].to_pandas().unique()) if args.gpu==1 else list(subobs['stationID'].unique())
    log.info('Working on locations {}:'.format(locations))
    if len(locations)==0:
        cities = list(allobs['CityKey'].to_pandas().unique()) if args.gpu==1 else list(allobs['CityKey'].unique())
        log.error('No observations found for city {}. Cities available in dataset: {}'.format(cityname,cities))
        return 0
    log.info('Will calculate bias-correction model for {} observation sites'.format(len(locations)))
    # longname is the name to be used for the figure title
    longname = cityname+', '+subobs['country'].iloc[0]
#---Compute NO2 anomalies for each location
    shap_list = []
    anomalies = []
    nlocations = 0
    dttrain = 0.0
    for l in locations:
        log.info('Building model for location {}'.format(l))
        # subset observation data
        obs = subobs.loc[subobs['stationID']==l].copy()
        # get model data closest to observation location
        ilat = lats[np.abs(lats-obs['lat'].mean()).argmin()]
        ilon = lons[np.abs(lons-obs['lon'].mean()).argmin()]
        model = allmod.loc[(allmod['lat']==ilat)&(allmod['lon']==ilon)].copy()
        if model.shape[0]==0:
            log.warning('No model data found for site {} ({:.1f}N,{:.1f}E) - skip'.format(l,ilat,ilon))
            continue 
#-------Train bias-correction model using XGBoost and apply bias correction to model output to obtain a business-as-usual estimate
#       that can be compare against the actual observations
        log.info('Prepare features and target')
        Xall, Yall, _ = _prepare_data(args,obs,model)
        if Yall.shape[0]<args.minnobs:
            log.warning('Not enough observations found ({}<{}) - skip {}'.format(Yall.shape[0],args.minnobs,l))
            continue 
        for n in range(args.nsplit):
            log.info('Building model {} of {}'.format(n+1,args.nsplit))
            # Split into training and validation. To do so split full dataset into n segments, and set one aside for validation.
            # The remaining segments form the training data. 
            if args.gpu==1:
                Xsplit = np.array_split(Xall.to_pandas(),args.nsplit)
                Ysplit = np.array_split(Yall.to_pandas(),args.nsplit)
            else:
                Xsplit = np.array_split(Xall,args.nsplit)
                Ysplit = np.array_split(Yall,args.nsplit)
            Xvalid = Xsplit.pop(n)
            Yvalid = Ysplit.pop(n)
            Xtrain = pd.concat(Xsplit)
            Ytrain = np.concatenate(Ysplit)
            if args.gpu==1:
                Xvalid = cudf.from_pandas(Xvalid)
                Yvalid = cudf.from_pandas(Yvalid)
                Xtrain = cudf.from_pandas(Xtrain)
                Ytrain = cp.asarray(Ytrain)
            bst, idttrain = _train(args,Xtrain,Ytrain)
            dttrain += idttrain
            # Validate 
            if args.validate==1:
                _valid(args,l,bst,Xvalid,Yvalid,n) 
            # Apply bias correction to model output to obtain 'business-as-usual' estimate and compare this value against observations
            pred, shap_values = _apply_bias(args,bst,obs,model)
            pred['stationID'] = [l for i in range(pred.shape[0])]
            pred['instance'] = [n for i in range(pred.shape[0])]
            anomalies.append(pred)
            shap_list.append(shap_values)
        # increase location counter
        nlocations += 1
#---Plot results agreggated over all sites
    no2diff = cudf.concat(anomalies).to_pandas() if args.gpu==1 else pd.concat(anomalies)
    _make_timeseries(args,no2diff,title=longname)
#---Plot shap values
    if args.shap==1:
        _plot_shap_values(args,shap_list,title=longname) 
    return nlocations, dttrain


def _read_obs(args):
    '''read observations from file'''
    log = logging.getLogger(__name__)
    log.info('Reading observations: {}'.format(args.obsfile))
    if args.gpu==1:
        obsdat = cudf.read_csv(args.obsfile,parse_dates=['ISO8601'],date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    else:
        obsdat = pd.read_csv(args.obsfile,parse_dates=['ISO8601'],date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    allobs = obsdat.loc[(obsdat['obstype']=='no2')]
    return allobs


def _read_model(args):
    '''read model output from file'''
    log = logging.getLogger(__name__)
    SKIPVARS = ['ISO8601','weekday','trendday','location','lat','lon','CLDTT','Q10M','T10M','TS','U10M','V10M','ZPBL','year','month','day','hour']
    log.info('Reading model: {}'.format(args.modfile))
    if args.gpu==1:
        model = cudf.read_csv(args.modfile,parse_dates=['ISO8601'],date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    else:
        model = pd.read_csv(args.modfile,parse_dates=['ISO8601'],date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
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
        model[v] = model[v].fillna(0.0)
        model[v] = model[v].values * scal 
    return model 


def _prepare_data(args,obs,model,mindate=None,maxdate=dt.datetime(2020,1,1),trendday=dt.datetime(2018,1,1)):
    '''prepare data for training and prediction'''
    log = logging.getLogger(__name__)
    # subsample observation data to columns of interest
    obsl = obs.copy()
    obsl.loc[obsl.unit=='ugm-3','value']=obsl.loc[obsl.unit=='ugm-3','value']*UG2PPBV
    _ = [obsl.pop(var) for var in ['stationID','unit','lat','lon','original_station_name','obstype','CityKey','country'] if var in obsl]
    # eventually convert ug/m3 to ppbv using standard conversion factor 
    # merge observations and model data and restrict to specified time window
    Xall = obsl.merge(model,how='inner',on='ISO8601')
    if mindate is not None:
        Xall = Xall.loc[Xall['ISO8601']>mindate]
    if maxdate is not None:
        Xall = Xall.loc[Xall['ISO8601']<maxdate]
    # ML target is observation - model difference
    Yall = Xall['value'] - Xall['NO2']
    # drop values not needed
    _ = [Xall.pop(var) for var in DROPVARS if var in Xall]
    values = Xall.pop('value')
    return Xall,Yall,values


def _valid(args,location,bst,Xvalid,Yvalid,instance):
    '''make prediction using XGboost model'''
    log = logging.getLogger(__name__)
    bias,conc,dates,shap = _predict(args,bst,Xvalid)
    if args.gpu==1:
        bias = bias.get()
        conc = conc.get()
        Xvalid = Xvalid.to_pandas()
        Yvalid = Yvalid.to_pandas()
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    axs[0] = _plot_scatter(axs[0],bias,Yvalid,-60.,60.,'Predicted bias [ppbv]','True bias [ppbv]','Bias')
    axs[1] = _plot_scatter(axs[1],Xvalid['NO2'],Xvalid['NO2'].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Original')
    axs[2] = _plot_scatter(axs[2],conc,Xvalid['NO2'].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Adjusted (XGBoost)')
    _ = fig.suptitle(location,y=0.98)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    ofile = 'validation_{}_{:02d}.png'.format(location.replace(' ',''),instance)
    fig.savefig(ofile)
    log.info('Scatter plot written to {}'.format(ofile))
    plt.close()
    return


def _predict(args,bst,Xpredict):
    '''make prediction using XGBoost model and return predicted bias and bias-corrected concentration'''
    log = logging.getLogger(__name__)
    dates = Xpredict.pop('ISO8601')
    predict = xgb.DMatrix(Xpredict)
    predicted_bias = bst.predict(predict)
    if args.gpu==1:
        predicted_bias = cp.asarray(predicted_bias)
    predicted_conc = Xpredict['NO2'].values + predicted_bias    
    shap_values = _get_shap_values(args,bst,predict) if args.shap==1 else None
    return predicted_bias, predicted_conc, dates, shap_values


def _get_shap_values(args,bst,X):
    '''Get SHAP values for given xgboost object bst and set of input features X'''
    log = logging.getLogger(__name__)
    predictor = "gpu_predictor" if args.gpu==1 else "cpu_predictor"
    bst.set_param({"predictor": predictor}) 
    shap_array = np.abs(bst.predict(X,pred_contribs=True))
    # shap_array has n+1 values, last column is the bias
    shap_values = pd.DataFrame(data=shap_array[:,:-1],columns=list(bst.feature_names))
    return shap_values 


def _train(args,Xtrain,Ytrain):
    '''train XGBoost model'''
    log = logging.getLogger(__name__)
    Xtrain.pop('ISO8601')
    train = xgb.DMatrix(data=Xtrain,label=Ytrain)
    tree_method = 'gpu_hist' if args.gpu==1 else 'hist'
    params = {'booster':'gbtree','tree_method':tree_method}
    #log.info('Training XGBoost model...')
    t0 = time.perf_counter()
    bst = xgb.train(params,train)
    return bst, time.perf_counter()-t0


def _apply_bias(args,bst,obs,model):
    '''apply bias-correction to model and compare against observations'''
    log = logging.getLogger(__name__)
    Xall, Yall, conc_obs = _prepare_data(args,obs,model,mindate=dt.datetime(2018,1,1),maxdate=None)
    bias_pred, conc_pred, dates, shap_values = _predict(args,bst,Xall)
    if args.gpu==1:
        anomaly = conc_obs - cudf.Series(conc_pred)
        odat = cudf.DataFrame({'ISO8601':dates,'predicted':conc_pred,'observed':conc_obs,'anomaly':anomaly}), shap_values
    else:
        anomaly = conc_obs - conc_pred
        odat = pd.DataFrame({'ISO8601':dates,'predicted':conc_pred,'observed':conc_obs,'anomaly':anomaly}), shap_values
    return odat 


def _make_timeseries(args,anomalies,rolling=21,mindate=dt.datetime(2019,1,1),title='unknown'):
    '''plot time series of NO2 anomalies'''
    log = logging.getLogger(__name__)
    # determine 'large' prediction uncertainty, defined as the standard deviation of the model-prediction error on the 
    # original (unaggregated) data
    sigma_large = np.std(anomalies.loc[anomalies.ISO8601<dt.datetime(2020,1,1),'anomaly'])
    # first group data by date and location, i.e. aggregate across all n-fold predictions
    dat = anomalies.groupby(['ISO8601','stationID']).mean().reset_index()
    # now agreggate data across locations
    dat = dat.set_index('ISO8601').resample('1D').mean().rolling(window=rolling,min_periods=1).mean().reset_index()
    # determine 'small' prediction uncertainty, defined as the standard deviation of the model-prediction error on the 
    # aggregate data
    sigma_small = np.std(dat.loc[dat.ISO8601<dt.datetime(2020,1,1),'anomaly'])
    # percent deviation is the percent difference between observed and model-predicted concentrations 
    dat['percent_deviation'] = 100.0*(dat['observed']-dat['predicted'])/dat['predicted'] 
    dat['yerrl1'] = 100.0*(dat['observed']-sigma_large-dat['predicted'])/dat['predicted'] 
    dat['yerrl2'] = 100.0*(dat['observed']+sigma_large-dat['predicted'])/dat['predicted'] 
    dat['yerrs1'] = 100.0*(dat['observed']-sigma_small-dat['predicted'])/dat['predicted'] 
    dat['yerrs2'] = 100.0*(dat['observed']+sigma_small-dat['predicted'])/dat['predicted'] 
    # plot time series of the percent deviation 
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(1,1,1)
    plt.axhline(0.0,color='darkgrey',linestyle='dashed')
    _ = ax.fill_between(dat.ISO8601.values,dat['yerrl1'].values,dat['yerrl2'].values,alpha=0.1,facecolor='teal',lw=0)
    _ = ax.fill_between(dat.ISO8601.values,dat['yerrs1'].values,dat['yerrs2'].values,alpha=0.4,facecolor='teal',lw=0)
    ax.plot(dat.ISO8601.values,dat['percent_deviation'].values,color='teal',linewidth=3,label='NO$_{2}$')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax.set_xlim([mindate,dt.datetime(2021,1,1)])
    ax.set_ylabel('NO$_{2}$ anomaly from baseline')
    ax.text(0.02,0.02,'N={}'.format(len(anomalies['stationID'].unique())),horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)    
    plt.title(title)
    plt.xticks(rotation=45)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    ofile = 'no2_anomalies_'+title.replace(' ','').replace(',','')+'.png'
    plt.savefig(ofile)
    plt.close()
    log.info('Figure written to {}'.format(ofile))
    return


def _plot_shap_values(args,shap_list,title):
    '''Make boxplot of SHAP values for the given location'''
    log = logging.getLogger(__name__)
    NFEATURES = 20 # number of features to plot
    # get dataframe with all shap values, reduce data to first NFEATURES number of features, and sort by median
    shaps = pd.concat(shap_list)
    medians = shaps.median()
    medians = pd.DataFrame(medians).sort_values(by=0,ascending=False)
    features = list(medians.index[:NFEATURES])
    shaps = shaps[list(medians.index[:NFEATURES])[::-1]]
    # make boxplot
    fig = plt.figure(figsize=(6,4))
    box = plt.boxplot(shaps,patch_artist=True,showfliers=False,vert=False,labels=shaps.columns)
    colors = list(plt.cm.jet(1.*np.arange(NFEATURES)/float(NFEATURES))) 
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.xlabel('absolute SHAP value')
    plt.xscale("log")
    plt.title(title)
    plt.tight_layout()
    ofile = 'shap_values_'+title.replace(' ','').replace(',','')+'.png'
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
    p.add_argument('-o','--obsfile',type=str,help='observation file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/obs.csv')
    p.add_argument('-m','--modfile',type=str,help='model file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/model.csv')
    p.add_argument('-c','--cities',type=str,nargs="+",help='city names',default='NewYork')
    p.add_argument('-n','--nsplit',type=int,help='number of cross-fold validations',default=8)
    p.add_argument('-v','--validate',type=int,help='make validation figures (1=yes; 0=no)?',default=0)
    p.add_argument('-s','--shap',type=int,help='plot shap values for each city (1=yes; 0=no)?',default=0)
    p.add_argument('-g','--gpu',type=int,help='run on gpu?',default=0)
    p.add_argument('-mn','--minnobs',type=int,help='minimum number of required observations (for training)',default=8760)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

