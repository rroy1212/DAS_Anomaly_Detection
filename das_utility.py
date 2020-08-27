#Author: Alex Sun
#Date: 06/24/2020
#purpose: move all common utility functions here
#==============================================================================
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

DEBUG = True

def getChannelSpecgram(datatype, traceList, outfile, channelStart, channelStep=10):
    """Generate spectrogram for a single channel
    one picture for each channel
    @param datatype, 'mat' or 'segy'
    @param a list of traces for segy or numpy array for mat
    @param sampleRate, rate of sampling
    @param outfile, name of the output
    """
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

    for itr in range(0,nTraces,channelStep):
        F,T,SXX = signal.spectrogram(st[itr].data, fs=sampleRate, window='hann')
        S1 = np.log10(np.abs(SXX/np.max(SXX)))
        if DEBUG:
            plt.figure()
            plt.pcolormesh(T, F, S1)
            print (channelStart+itr)
            plt.savefig('tracespectrogram_{0}_ch{1}.png'.format(outfile, channelStart+itr))
            plt.close()
        datafile = 'tracespectrogram_{0}_ch{1}.npy'.format(outfile,channelStart+itr)
        np.save(datafile,S1)
    """
    #this uses obspy spectogram
    st[50].spectrogram(samp_rate=sample_rate, 
                              per_lap = frac_overlap,
                              wlen = window,
                              dbscale=True, 
                              log=True)
    for itr in range(0,100,10):
        st[itr].spectrogram(log=True, cmap='copper')
        plt.savefig('tracespectrogram_{0}_{1}.png'.format(self.filename, itr))
    """

def _nearest_pow_2(x):
    """
    Find power of two nearest to x
    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0
    :type x: float
    :param x: Number
    :rtype: Int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b 

def getSpectralEnergyFrame(datatype, traceList, outfile, channelStart, channelEnd, winlen=1000):
    """Get spectral energy for all channels
    one picture for all channels
    @param datatype, 'mat' or 'segy'
    @param a list of traces for segy or numpy array for mat
    @param sampleRate, rate of sampling
    @param outfile, name of the output    
    """
    assert(datatype in ['mat', 'segy'])    
    if datatype=='segy':
        st = obspy.Stream(traceList)            
    else:
        raise Exception('not implemented')
    sampleRate = traceList[0].stats.sampling_rate

    wlen = 256
    nfft = int(_nearest_pow_2(wlen))
    npts = len(st[0].data)
    per_lap = 0.9
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))
    nlap = int(nfft * float(per_lap))

    nTraces = len(traceList)
    nperlen = len(traceList[0].data)
    if winlen>=nperlen:
        nFrames=1
    else:
        nFrames = int(nperlen/winlen)

    print ('sample rate is ', sampleRate, 'nfft=', nfft, 'noverlap', nlap)

    for iframe in range(nFrames):    
        Emat = None
        for itr in range(0,nTraces):
            F,T,SXX = signal.spectrogram(np.array(st[itr].data[iframe*winlen:(iframe+1)*winlen]), fs=sampleRate,  
                               window='hann', nfft=nfft)
            #sum along frequency axis        
            energy = np.sum((SXX[1:,:]/np.max(SXX[1:,:])),axis=0)
            #energy = np.abs(np.log10(np.abs(energy/np.max(energy)))*10.0)
            #energy = np.log10(energy)*10.0
            if Emat is None:
                Emat = np.zeros((nTraces, len(T)))
            Emat[itr,:]=energy
        
        #datafile = 'spectralenergy_{0}_ch{1}_{2}.npy'.format(outfile,channelStart,channelEnd)
        #np.save(datafile,Emat)
        #scale to 0 255
        print (Emat.max())
        Emat = (255.0 / Emat.max() * (Emat - Emat.min())).astype(np.uint8)
        im = Image.fromarray(Emat, 'L')
        imgfile = 'spectralenergy_{0}_ch{1}_{2}_{3}.png'.format(outfile,channelStart,channelEnd,iframe)            
        im.save(imgfile)
        histogram = im.histogram()
        imgfile = 'spectralhist_{0}_ch{1}_{2}_{3}.png'.format(outfile,channelStart,channelEnd,iframe)            
        plt.figure()
        plt.plot(histogram)
        plt.savefig(imgfile)



       
def getSpectralEnergy(datatype, traceList, outfile, channelStart, channelEnd):
    """Get spectral energy for all channels
    one picture for all channels
    @param datatype, 'mat' or 'segy'
    @param a list of traces for segy or numpy array for mat
    @param sampleRate, rate of sampling
    @param outfile, name of the output    
    """
    assert(datatype in ['mat', 'segy'])    
    if datatype=='segy':
        st = obspy.Stream(traceList)            
    else:
        raise Exception('not implemented')
    sampleRate = traceList[0].stats.sampling_rate
    #for decimated data,sampleRate should be reflected
    #set wlen to 0.25 sec, high pass is 250
    wlen = 0.5*sampleRate
    nfft = int(_nearest_pow_2(wlen))
    npts = len(st[0].data)
    per_lap = 0.9
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))
    nlap = int(nfft * float(per_lap))

    nTraces = len(traceList)
    Emat = None
    print ('sample rate is ', sampleRate, 'nfft=', nfft, 'noverlap', nlap)
    
    t_ = (traceList[0].stats.endtime-traceList[0].stats.starttime)
    dx_ = traceList[1].stats.distance - traceList[0].stats.distance
    extent = [0,len(traceList)*dx_/1e3,0,t_/100.0]

    for itr in range(0,nTraces):
        #F,T,SXX = signal.spectrogram(np.array(st[itr].data), fs=sampleRate,  
        #                   window='hann', nfft=nfft, mode='magnitude')
        F,T,SXX = signal.spectrogram(np.array(st[itr].data), fs=sampleRate,  
                           window='hann', nfft=nfft)
        #sum along frequency axis        
        #energy = np.sum((SXX[1:,:]/np.max(SXX[1:,:])),axis=0)
        energy = np.sum(SXX[1:,:],axis=0)
        #energy = np.log10(np.abs(energy/np.max(energy)))*10.0
        energy = np.log10(energy)*10.0
        if Emat is None:
            Emat = np.zeros((nTraces, len(T)))
        Emat[itr,:]=energy
    if DEBUG:
        plt.figure()
        im = plt.imshow(Emat,extent=extent)
        plt.colorbar(im)
        plt.savefig('spectralenergy{0}_ch{1}_{2}.png'.format(outfile,channelStart,channelEnd))
        plt.close()
    
def getGatherPlot(datatype, traceList, sampleRate, outfile, channelStart, channelEnd, 
                  winlen=1000, clim=None, outimagefolder='images'):        
        """Generate gather Plot
        @param traceList, list of obspy traces
        @param sampleRate, rate of das sampling
        @param outfile, name of the output
        @param channelStart, start no of channel
        @param channelEnd, end no of channel
        @param winlen, length of the window (in no. of samples)
        """
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
                ax.set_xticks(np.arange(0,channelEnd-channelStart, int((channelEnd-channelStart)/4)))
                ax.set_xticklabels(np.arange(channelStart,channelEnd, int((channelEnd-channelStart)/4)))
                ax.set_xlabel(xlabel)    
                ax.set_ylabel('Time (s)')
                plt.colorbar()
                filename = '{0}/gatherplot{1}_ch{2}_{3}_{4}o.png'.format(outimagefolder, 
                                                                         outfile,
                                                                         channelStart,
                                                                         channelEnd, iframe)
                plt.savefig(filename, transparent=True)
                plt.close()          
                
                #PC = PhaseCoherence(50.0, gatherArr[:,iframe*winlen:(iframe+1)*winlen], FS=sampleRate)
                #print (filename, 'PC=',PC)
            
            #npyfile = 'npyfiles/gather_{0}_ch{1}_{2}_{3}.npy'.format(outfile,channelStart,channelEnd,iframe)            
            #np.save(npyfile, gatherArr[:,iframe*winlen:(iframe+1)*winlen], allow_pickle=True)
            
            #uncomment to generate histogram
            """
            histogram = im.histogram()
            imgfile = 'hist_{0}_ch{1}_{2}_{3}.png'.format(outfile,channelStart,channelEnd,iframe)            
            plt.figure()
            plt.plot(histogram)
            plt.savefig(imgfile)
            """
            
def getStackPlot(datatype, stackArr, sampleRate, outfile, channelStart, channelEnd):        
        nTraces = stackArr.shape[0]
        nperlen = stackArr.shape[1]
        #window width
        winlen = 1000 
        
        plt.figure(figsize=(10,4),dpi=400)
        plt.imshow(stackArr, cmap='bwr', origin='lower')   
        ax = plt.gca()
        ax.set_aspect(0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')    
        plt.savefig('fullimages/stackplot{0}_ch{1}_{2}.png'.format(outfile,channelStart,channelEnd))
        plt.close()          
        return
        nFrames = int(nperlen/winlen)
        for iframe in range(nFrames-2):
            if DEBUG:
                plt.figure(figsize=(16,10))
                plt.imshow(stackArr[:,iframe*winlen:(iframe+1)*winlen], cmap='bwr', origin='lower')   
                ax = plt.gca()
                ax.set_xlabel('Time')
                ax.set_ylabel('Channel')    
                plt.savefig('images/stackplot{0}_ch{1}_{2}_{3}.png'.format(outfile,channelStart,channelEnd, iframe))
                plt.close()          

def getSingleChannelSpectrum(datatype, traceList, outfile, channelStart, channelStep=10):
    print ('channel samp rate is ', traceList[0].stats.sampling_rate)
    sampRate = traceList[0].stats.sampling_rate
    nTraces = len(traceList)
    wlen = 128
    ndata = len(traceList[0].data)
    print (ndata)
    nfft = int(_nearest_pow_2(wlen))

    for itr in range(0,nTraces,channelStep):        
        sp = fq.spectrum(traceList[itr].data,nfft=nfft,win=wlen)
        if DEBUG:
            plt.figure()
            plt.plot(sp[:int(wlen/2)])
            plt.savefig('spectrum{0}_ch{1}.png'.format(outfile,channelStart+itr))
            plt.close()

def getChannelSpecgram2(traceList, outfile, wlen, channelStart, channelStep=10, 
                        smooth=True, interpolate=False):
    sampRate = traceList[0].stats.sampling_rate
    nTraces = len(traceList)
    ndata = len(traceList[0].data)
    #
    nfft = int(_nearest_pow_2(wlen))
    print (nfft)
    
    for itr in range(0,nTraces,channelStep):        
        #sp = fq.spectrum(traceList[itr].data,nfft=nfft,win=wlen)       
        # finally...calculate spectrogram of current cont. segment
        spectrogram, freqs, time = mlab.specgram(traceList[itr].data, 
                                                 nfft, 
                                                 sampRate, 
                                                 mode="psd")     
        print (time, freqs)
        # smooth spectrogram
        if smooth:
            spectrogram = scipy.ndimage.gaussian_filter(spectrogram, sigma=(30, 0.75))

        # "downsample" spectrogram
        if interpolate:
            freqs, spectrogram = _interpolate_ppsd(freqs, spectrogram, fmin=10.0, fmax=100.0)

        if DEBUG:
            plt.figure()
            plt.pcolormesh(time, freqs, spectrogram)
            plt.savefig('spectrum{0}.png'.format(outfile,channelStart))
            plt.close()
        break
def _interpolate_ppsd(freqs, spectrogram, fmin, fmax):
    """
    Function to downsample spectrogram by interpolation.
    :param freqs: Frequency vector returned by mlab.spectrogram
    :param spectrogram: Spectrogram returned by mlab.spectrogram (potentially postprocessed)
    :param fmin: Minimum frequency of interest.
    :param fmax: Maximum frequency of interest.
    :return: New frequency vector and associated downsampled (interpolated) spectrogram.
    """
    # frequencies at which ppsd is evaluated
    f_new = np.logspace(np.log10(fmin), np.log10(fmax), 7500)

    # interpolate ppsds (colums of spectrogram) at the new frequencies
    wins = spectrogram.shape[1]
    spec_new = np.zeros((f_new.size, wins))
    for i in range(wins):
        f = interp.interp1d(freqs, spectrogram[:,i], kind="cubic")
        spec_new[:,i] = f(f_new)
    return f_new, spec_new

def getChannelScalogram(traceList, channelNo, outfile, imagefolder='scalogram'):
    tr = traceList[channelNo]
    dt = tr.stats.delta
    f_min = 10
    f_max = 200
    npts = tr.stats.npts

    t = np.linspace(0, dt * npts, npts)
    
    scalogram = cwt(tr.data, dt, 8, f_min, f_max)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x, y = np.meshgrid(
        t,
        np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    
    ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
    ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_yscale('log')
    ax.set_ylim(f_min, f_max)
    plt.savefig('{0}/{1}_scalogram_channel.png'.format(imagefolder, outfile, channelNo))


######################### trace plot ############################

def gen_trace_plot(traceList, channelNo, outfile):
    tr = traceList[channelNo]
    #print(outfile)
    tr.plot(outfile='test.png')

        
