## Imports for Flask

from __future__ import division, print_function
from random import randint
import os
from time import strftime
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import random

################ import cluster py  ##################
import image_clustering
from image_clustering import *


############################# imports for obspy ################

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time, datetime
from obspy.io.segy.core import _read_segy
import obspy
from numpy.lib.stride_tricks import as_strided
import forgeUtils as utils
from joblib import Parallel, delayed, load, dump
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
# from das_utility_latest import *
import obspy
import scipy
import scipy.interpolate as interp
import math,sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import mlab
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from obspy.signal import freqattributes as fq
import pywt

##################################################################

DEBUG=True
#METHOD=1, complicated method
#METHOD=2, simple method
METHOD=2

###################################################################

def moving_avg(a, halfwindow, mask=None):
    """
    Performs a fast n-point moving average of (the last
    dimension of) array *a*, by using stride tricks to roll
    a window on *a*.
    Note that *halfwindow* gives the nb of points on each side,
    so that n = 2*halfwindow + 1.
    If *mask* is provided, values of *a* where mask = False are
    skipped.
    Returns an array of same size as *a* (which means that near
    the edges, the averaging window is actually < *npt*).
    """


    if mask is None:
        mask = np.ones_like(a, dtype='bool')

    zeros = np.zeros(a.shape[:-1] + (halfwindow,))
    falses = zeros.astype('bool')

    a_padded = np.concatenate((zeros, np.where(mask, a, 0), zeros), axis=-1)
    mask_padded = np.concatenate((falses, mask, falses), axis=-1)


    npt = 2 * halfwindow + 1  # total size of the averaging window
    rolling_a = as_strided(a_padded,
                           shape=a.shape + (npt,),
                           strides=a_padded.strides + (a.strides[-1],))
    rolling_mask = as_strided(mask_padded,
                              shape=mask.shape + (npt,),
                              strides=mask_padded.strides + (mask.strides[-1],))

    # moving average
    n = rolling_mask.sum(axis=-1)
    return np.where(n > 0, rolling_a.sum(axis=-1).astype('float') / n, np.nan)


def filterSingleTrace(tr, *args):
    """
    Performing aggregation or filtering on a single trace
    Warning: this will overwrite original trace content, so pass a copy of trace
    """
    
    #do bandpass filtering
    fmin = args[0]
    fmax = args[1]
    dn_rate = args[2]
    onebit_norm = args[3]
    
    corners= 4
    zerophase=True
    window_time = 15.0
    window_freq = 30.0

    tr.filter('bandpass', freqmin=fmin, freqmax=fmax, 
              corners=corners, zerophase=zerophase)
    #resample if necessary
    if dn_rate < tr.stats.sampling_rate:
        dnfactor = tr.stats.sampling_rate/dn_rate
        if abs(dnfactor - int(dnfactor))>1e-6:
            raise Exception('down sampling must equal integer multiple')
        #note: after decimation, the stat will be changed
        tr.decimate(int(dnfactor), no_filter=True)

    # ==================
    # Time normalization
    # ==================
    if onebit_norm:
        # one-bit normalization
        tr.data = np.sign(tr.data)
    else:
        # normalization of the signal by the running mean
        # in the earthquake frequency band
        # Time-normalization weights from smoothed abs(data)
        # Note that trace's data can be a masked array
        halfwindow = int(round(window_time * tr.stats.sampling_rate / 2))
        mask = ~tr.data.mask if np.ma.isMA(tr.data) else None
        tnorm_w = moving_avg(np.abs(tr.data),
                                     halfwindow=halfwindow,
                                     mask=mask)
        if np.ma.isMA(tr.data):
            # turning time-normalization weights into a masked array
            tnorm_w = np.ma.masked_array(tnorm_w, tr.data.mask)

        if np.any((tnorm_w == 0.0) | np.isnan(tnorm_w)):
            # illegal normalizing value -> skipping trace
            raise Exception("Zero or NaN normalization weight")

        # time-normalization
        tr.data /= tnorm_w

        # ==================
        # Spectral whitening
        # ==================
        fft = rfft(tr.data)  # real FFT
        deltaf = tr.stats.sampling_rate / tr.stats.npts  # frequency step
        
        # smoothing amplitude spectrum
        halfwindow = int(round(window_freq / deltaf / 2.0))
        weight = moving_avg(abs(fft), halfwindow=halfwindow)
        # normalizing spectrum and back to time domain
        tr.data = irfft(fft / weight, n=len(tr.data))
        # re bandpass to avoid low/high freq noise
        tr.filter(type="bandpass",
                     freqmin=fmin,
                     freqmax=fmax,
                     corners=corners,
                     zerophase=zerophase)

    # Verifying that we don't have nan in trace data
    if np.any(np.isnan(tr.data)):
        raise Exception("Got NaN in trace data")

def getSingleTrace(tr, dnRate, isIntegrate=False):
    """
    this is used to process a single trace
    """
    if isIntegrate:
        tr.integrate()
        
    #hardcoding bandpass filter window here
    filterSingleTrace(tr, 10.0, 200.0, dnRate, False)
                     
    return tr

############## Main Class for file processing #####################

class Forge():
    """
    Main class for loading and processing Forge data
    Forge has 1088 channels, sampling Rate 2000
    From Forge training,
    gaugelength     = 10.0 
    dx_in_m         = 1.02
    das_units       = 'n$\epsilon$/s'
    geophone_units  = 'm/s^2'
    geophone_fac    = 2.333e-7
    fo_start_ch     = 197    
    """
    def __init__(self, segyfile, 
                 channelRange, 
                 frameWidth, 
                 downsampleFactor=1,
                 skipInterval = 1,
                 isIntegrate=False, traces=[]):    
        """
        @param segyfile, name of the seg-y file
        @param channelRange, list [min_channelNo, max_channelNo]        
        @param frameWidth, width of each frame for outputting
        @param downsampleFactor, downsampling factor, subset interval on time dimension
        @param skipInterval, subset interval on channel dimension
        @param isIntegrate, true to integrate the trace
        """    
        iloc = segyfile.find('iDAS')
        self.filename = segyfile[iloc:-4]
        #
        startime=time.time()
        self.load(segyfile)
        print ('loading seg-y took', time.time()-startime)

        self.gather = None
        self.frameWidth = frameWidth
        self.dsfactor = downsampleFactor
        self.skipInt = skipInterval
        self.channelRange = np.arange(channelRange[0],channelRange[1])
        self.isIntegrate = isIntegrate
        self._getGather()
        
        
    def load(self, segyfile):
        """Load seg-y and add hard coded header information to trace
        """
        gaugelength     = 10.0 
        dx_in_m         = 1.02
        das_units       = 'n$\epsilon$/s'
        fo_start_ch     = 197
        #end channel 1280
        stream = obspy.Stream()
        dd = _read_segy(segyfile, unpack_trace_headers=True)
        stream += utils.populate_das_segy_trace_headers(dd,
                                                        dx_in_m=dx_in_m,
                                                        fo_start_ch=fo_start_ch,
                                                        units=das_units)
        self.st = stream
        
    def _getGather(self):
        """
        This is the main function that gathers all traces and form a station
        """
        if self.gather is None:
            print ('loading traces')
            if DEBUG:
                start_time = time.time()

            nChannels = len(self.channelRange)
            print(self.channelRange)
            traceList = [None]*nChannels
            #
            if METHOD==1:
                #demean all traces
                self.st.detrend('constant')
                #detrend
                self.st.detrend('linear')
                #
                #taper all traces on both sides
                #self.st.taper(max_percentage=0.05, type='cosine')
                print ('original sample rate is', self.st[0].stats.sampling_rate)
                self.sampRate = self.st[0].stats.sampling_rate /self.dsfactor
                print ('new sample rate is ', self.sampRate)
                #self.st.decimate(self.dsfactor)
                #process traces in parallel
    
                with Parallel(n_jobs=12) as parallelPool:
                    traceList = parallelPool(delayed(getSingleTrace)
                                            (self.st[channelNo],                                                     
                                             self.sampRate,
                                             self.isIntegrate)
                                 for channelNo in self.channelRange)

                self.traceList = traceList
                self.st = obspy.Stream(traceList)  
            elif METHOD==2:
                #do simple filtering as in Ariel Lellouch paper
                #self.st = utils.medianSubtract(self.st)
                self.st.detrend('constant')
                self.st.detrend('linear')
                self.st.filter('bandpass',freqmin=10,freqmax=150)
                if self.dsfactor>1:
                    self.sampRate = self.st[0].stats.sampling_rate /self.dsfactor
                    self.st.decimate(self.dsfactor, no_filter=True)
                print(self.channelRange)
                self.traceList=[self.st[channelNo] for channelNo in self.channelRange]
                
            if DEBUG:
                print ('processing time is ', time.time()-start_time)


############################# Function for Scalogram Plot #########################################

def getChannelScalogram(traceList, channelNo, channelStart, outfile, imagefolder='static/Obspy_Plots/diff_plots'):
    tr = traceList[channelNo]
    dt = tr.stats.delta
    f_min = 10
    f_max = 150
    npts = tr.stats.npts

    t = np.linspace(0, dt * npts, npts)
    
    scalogram = cwt(tr.data, dt, 8, f_min, f_max)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    x, y = np.meshgrid(
        t,
        np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))   
    #ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
    ax.pcolormesh(x, y, np.abs(scalogram), cmap='jet')
    ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_yscale('log')
    ax.set_ylim(f_min, f_max)
    image_name = '{0}_scalogram_channel{1}.png'.format(outfile, channelNo+channelStart)
    print(image_name)
    plt.savefig('{0}/{1}_scalogram_channel{2}.png'.format(imagefolder, outfile, channelNo+channelStart))
    plt.close()

    return image_name



################# Function for Spectrogram Plot ####################
def getChannelSpecgram(datatype, traceList, outfile, channelStart, channelStep=10):
    assert(datatype in ['mat', 'segy'])
    if datatype=='segy':
        st = obspy.Stream(traceList)            
        nTraces = len(st)
    else:
        raise Exception('not implemented')
    sampleRate = traceList[0].stats.sampling_rate
    print ('in spectrogram sampleRate=', sampleRate)
    window = 256
    nfft = np.min([256, len(traceList[0].data)])
    frac_overlap = 0.1
    img_list=[]

    for itr in range(0,nTraces,channelStep):
        F,T,SXX = signal.spectrogram(st[itr].data, fs=sampleRate, window='hann')
        S1 = np.log10(np.abs(SXX/np.max(SXX)))
        if DEBUG:
            plt.figure()
            plt.pcolormesh(T, F, S1)
            print (channelStart+itr)
            image_name = 'tracespectrogram_{0}_ch{1}.png'.format(outfile, channelStart+itr)
            print(image_name)
            img_list.append(image_name)
            plt.savefig('static/Obspy_Plots/diff_plots/tracespectrogram_{0}_ch{1}.png'.format(outfile, channelStart+itr))
            plt.close()

    return img_list[0]


####################### Fucntion for Gather Plot ###############################

def getGatherPlot(datatype, traceList, sampleRate, outfile, channelStart, channelEnd, 
                  winlen=1000, clim=None, outimagefolder='static/Obspy_Plots/diff_plots/gather_plots'):        
        nTraces = len(traceList)
        #data length in the das file
        nperlen = len(traceList[0].data)

        gatherArr = np.zeros((nTraces,nperlen),dtype=np.float64)     
        for itr in range(nTraces):
            gatherArr[itr,:] = traceList[itr].data
        gatherArr = np.flipud(gatherArr.T)
        vmin = np.nanmin(gatherArr)
        vmax = np.nanmax(gatherArr)
        print ('vmin', vmin, 'vmax', vmax)
        #if True scale to [-1,1]
        isScale = False
        if isScale:
            gatherArr =  (gatherArr-gatherArr.min())/(gatherArr.max()-gatherArr.min())

        if winlen>=nperlen:
            nFrames=1
        else:
            nFrames = int(nperlen/winlen)

        img_list=[]
        for iframe in range(nFrames):
            if DEBUG:
                vmin = np.nanmin(gatherArr)
                vmax = np.nanmax(gatherArr)
                t_ = (traceList[0].stats.endtime-traceList[0].stats.starttime)/nFrames
                dx_ = traceList[1].stats.distance - traceList[0].stats.distance
                extent = [0,len(traceList)*dx_/1e3,0,t_]
                xlabel = 'Linear Fiber Length [km]'                

                plt.figure(figsize=(10,10))
                if clim is None:
                    plt.imshow(gatherArr[iframe*winlen:(iframe+1)*winlen,:],  
                               origin='lower', vmin=vmin/10.0, vmax=vmax/10.0, 
                               extent=extent,
                               aspect='auto')   
                else:
                    plt.imshow(gatherArr[iframe*winlen:(iframe+1)*winlen,:],  
                               origin='lower', vmin=clim[0], vmax=clim[1], 
                               extent=extent,
                               aspect='auto')   
                    
                ax = plt.gca()
                ax.axis('off')
                img_name = 'gatherplot{0}_ch{1}_{2}_{3}o.png'.format(outfile, channelStart, channelEnd, iframe)  
                img_list.append(img_name)                 
                filename = '{0}/gatherplot{1}_ch{2}_{3}_{4}o.png'.format(outimagefolder, 
                                                                         outfile,
                                                                         channelStart,
                                                                         channelEnd, iframe)
                plt.savefig(filename, transparent=True)
                plt.close()   
        return img_list[0]     


####################### Fucntion to get Trace Plot #################################

def gen_trace_plot(traceList, channelNo, outfile):
    tr = traceList[channelNo]
    #print(outfile)
    img_name = 'trace_plot{0}_ch{1}.png'.format(outfile,channelNo)
    filename = 'static/Obspy_Plots/diff_plots/trace_plot{0}_ch{1}.png'.format(outfile,channelNo)
    tr.plot(outfile = filename)

    return img_name


##################### Fucntion to get  trace details for the uploaded segy file ############################

def get_segy_details(filename):
    st = _read_segy(filename, unpack_trace_headers=True)
    trace_cnt = len(st)
    tr=st[0]
    sampling_rate = tr.stats.sampling_rate
    npts = tr.stats.npts
    
    return trace_cnt,sampling_rate,npts

def get_obspy_plots(minchannelrange,maxchannelrange,framelen,dsfactor,filename):
    skipInterval=1
    channelRange=[minchannelrange,maxchannelrange]
    forge = Forge(filename, 
                   channelRange, 
                   framelen, 
                   downsampleFactor = dsfactor,
                   skipInterval=skipInterval,
                   isIntegrate=False, 
                   )
    gather_image = getGatherPlot('segy', forge.traceList, forge.sampRate, forge.filename, channelRange[0], channelRange[1],
                   winlen=framelen) 
    
    scalogram_img = getChannelScalogram(forge.traceList, int(channelRange[0]),channelRange[0], forge.filename)

    trace_Plt = gen_trace_plot(forge.traceList, int(channelRange[0]), forge.filename)
    specgram = getChannelSpecgram('segy', forge.traceList, forge.filename, channelRange[0], 100)
    obspy_plot = {'gather':gather_image,'scalogram':scalogram_img,'trace':trace_Plt,'specgram':specgram}
    
    return obspy_plot
    
def getMultipleGatherPlots(minchannelrange,maxchannelrange,framelen,dsfactor,filename):
    skipInterval=1
    for i in range(2):
        # chmin = random.sample(range(minchannelrange, minchannelrange + 100 ), 1)
        # print(chmin)
     
        chmin = random.randrange(minchannelrange,500,10)
        chmax = chmin + 400
        channelRange=[chmin,chmax]
        print(channelRange)
        forge = Forge(filename, 
                   channelRange, 
                   framelen, 
                   downsampleFactor = dsfactor,
                   skipInterval=skipInterval,
                   isIntegrate=False, 
                   )
        print("##### start generating plots")
        getGatherPlot('segy', forge.traceList, forge.sampRate, forge.filename, channelRange[0], 
                      channelRange[1], winlen=framelen)  
    

################################# app functionality ###############################################

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'
SPEC_FOLDER = os.path.join('static', 'Obspy_Plots','diff_plots')
# GATHER_FOLDER = os.path.join('static','Obspy_Plots','gather_plots')
print(SPEC_FOLDER)
app.config['UPLOAD_FOLDER'] = SPEC_FOLDER
minchannelrange=""
maxchannelrange=""
framelen = ""
dsfactor = ""
pathh5 = ""
pathjson = ""
pklpath = ""
gather_full_filename = ""

##################################### Index Page ###########################################

@app.route("/", methods=['GET', 'POST'])
def segydata():
    form1 = request.form
    #segy_files = ['PoroTomo_iDAS16043_160321000521.sgy', 'PoroTomo_iDAS16043_160321000721.sgy', 'PoroTomo_iDAS16043_160321000921.sgy']
    #return render_template('DASindex.html',form=form1,files=segy_files)
    return render_template('DASindex.html',form=form1)


@app.route("/process", methods=['GET', 'POST'])
def processdata():
    global filename
    form = request.form
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        path = str(filename)
        f.save(path)
        # filter_type=['Low Pass','High Pass','Band Pass']
        trace_cnt, sampling_rate,npts = get_segy_details(filename)
        file_data={'tr_cnt':trace_cnt,'samp_rate':sampling_rate,'number_sample':npts}

    return render_template('DAS_Process.html',form=form,data=file_data)


@app.route("/model", methods=['GET', 'POST'])
def display_plots():
    form = request.form
    print('###########testgen######')
    print(request.method)
    global gather_full_filename
    global minchannelrange
    global maxchannelrange
    global framelen
    global dsfactor
    if request.method == 'POST':
        print("----------process")
        minchannelrange=request.form['minchannelrange']
        maxchannelrange=request.form['maxchannelrange']
        framelen=request.form['framelen']
        dsfactor=request.form['dsfactor']
        #### generate gather_plots
        plot_details = get_obspy_plots(int(minchannelrange),int(maxchannelrange),int(framelen),int(dsfactor),str(filename))
        spec_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], plot_details['specgram'])
        CWT_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], plot_details['scalogram'])
        TF_full_filename = os.path.join(app.config['UPLOAD_FOLDER'], plot_details['trace'])
        gather_full_filename = os.path.join(app.config['UPLOAD_FOLDER'],'gather_plots\\' + plot_details['gather'] )
        print("full_filename is " ,gather_full_filename)

        test_data={'spec':spec_full_filename,'cwt':CWT_full_filename,'tf':TF_full_filename,'gather':gather_full_filename}
        #get_gather_plots(int(minchannelrange),int(maxchannelrange),int(framelen),int(dsfactor),str(filename))
    return render_template('plots.html', form=form, data = test_data)


@app.route("/gatherplots", methods=['GET', 'POST'])
def get_multiple_gatherplots():
    form = request.form
    form_data = {'minchannel':minchannelrange,'maxchannel':maxchannelrange}
    return render_template('multiple_gather_plots.html',form=form,data=form_data)

@app.route("/plots", methods=['GET', 'POST'])
def model_upload():
    form = request.form
    getMultipleGatherPlots(int(minchannelrange),int(maxchannelrange),int(framelen),int(dsfactor),str(filename))
    return render_template('model_upload_UI.html',form=form)


@app.route("/predict",methods=['GET', 'POST'])
def gen_image_clusters():
    global pathh5
    global pathjson
    global pklpath

    if request.method == 'POST':

        h5file = request.files['h5file']
        h5filename = h5file.filename
        pathh5 = 'model_uploads/' + str(h5filename)
        print(pathh5)
        h5file.save(pathh5)

        jsonfile = request.files['jsonfile']
        jsonfilename = jsonfile.filename
        pathjson = 'model_uploads/' + str(jsonfilename)
        print(pathjson)
        jsonfile.save(pathjson)

        pklfile = request.files['pklfile']
        pklfilename = pklfile.filename
        pklpath = 'model_uploads/' + str(pklfilename)
        print(pklpath)
        pklfile.save(pklpath)


    return render_template('image_cluster.html')
        
       
@app.route("/getcluster", methods=['GET', 'POST'])
def predict_cluster():
    input_gather = 'static/Obspy_Plots/diff_plots/predict'
    kmeans_model = pathh5
    densenet_json = pathjson
    densenet_h5 =  pathh5

    # result = predicting_cluster(input_gather, str(kmeans_model), str(densenet_json),str(densenet_h5))
    #result = 9
    return render_template('predict_cluster.html')




if __name__ == "__main__":
    app.run(use_reloader=False)


