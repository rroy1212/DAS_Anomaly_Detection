#1. Download and Read Data Tools

def get_paths(style,times):
    '''
    converts a UTCDateTime range times=[t0,t1] to a url list for FORGE.
    '''
    def get_das_paths(times):
        '''
        converts a UTCDateTime range times=[t0,t1] to a url list of FORGE DAS files
        using pandas. file urls will span the interval from 15-seconds before t0
        through the file containing t1. if t0=t1, return a single url.
        '''
        import pandas as pd
        from datetime import datetime,timedelta
        from obspy import UTCDateTime
        struct = pd.read_csv('get_all_silixa.sh',skiprows=1,header=None)
        struct[1] = [x[-16:-4] for x in struct[0]]
        struct[2] = [UTCDateTime(datetime.strptime(x,"%y%m%d%H%M%S")) for x in struct[1]]
        struct = struct[struct[2]>=times[0]-timedelta(seconds=15)]
        struct = struct[struct[2]<=times[1]]
        struct = struct.sort_values(2)
        struct.columns=['url','timestr','UTCTimeDate']
        struct = struct.reset_index()
        url = struct['url'].tolist()
        return url

    def get_geophone_paths(times):
        '''
        converts a UTCDateTime range times=[t0,t1] to a url list of FORGE GEOPHONE
        files using pandas. file urls will span the interval from 15-seconds before
        t0 through the file containing t1. if t0=t1, return a single url.
        '''
        import pandas as pd
        from datetime import datetime,timedelta
        from obspy import UTCDateTime
        struct = pd.read_csv('get_all_slb.sh',skiprows=1,header=None)
        struct[1] = [x[-21:-5] for x in struct[0]]
        struct[2] = [UTCDateTime(datetime.strptime(x,"%y%m%d%H%M%S.%f")) for x in struct[1]]
        struct = struct[struct[2]>=times[0]-timedelta(seconds=15)]
        struct = struct[struct[2]<=times[1]]
        struct = struct.sort_values(2)
        struct.columns=['url','timestr','UTCTimeDate']
        struct = struct.reset_index()
        url = struct['url'].tolist()
        return url

    if style=='das':
        url = get_das_paths(times)
    elif style=='geophone':
        url = get_geophone_paths(times)
    else:
        print('ERROR: style arg must be either "das" or "geophone"')
        return

    return url

def download(url,switch='off'):
    '''
    downloads segy DAS or GEOPHONE files for url list
    from https://constantine.seis.utah.edu/datasets.html to forgeData
    return filenames list
    '''
    import os

    if switch=='off':
        filenames = [u.split('/')[4] for u in url]
        try:
            os.system('mkdir forgeData') # first time, create dir
        except:
            None # directory already created
        for i,f in enumerate(filenames):
            if os.path.exists('forgeData/'+f):
                print(f+' alrady downloaded!')
            else:
                print('Downloading '+f+'...')
                os.system('wget --no-check-certificate '+url[i])
                os.system('mv '+f+' forgeData/')

    elif switch=='workshop':
        filenames = [u.split('/')[4] for u in url]

    return ['forgeData/'+f for f in filenames]

def populate_das_segy_trace_headers(Stream, dx_in_m=1.02, network='FORGE', fo_start_ch=0, units='None'):
    import pandas as pd
    for Trace in Stream:
        df = pd.DataFrame(dict(Trace.stats.segy.trace_header),index=[0]).transpose()[0]
        Trace.stats.network = network
        Trace.stats.station = '%05d' % (df.trace_sequence_number_within_line-1)
        Trace.stats.location = '0'
        Trace.stats.distance = (df.trace_sequence_number_within_line-1-fo_start_ch)  * dx_in_m
        Trace.stats.channel = 'Z'
        if units!='None':
            Trace.stats.units = units
    return Stream

def populate_geophone_segy_trace_headers(Stream, network='FORGE'):
    import numpy as np
    geophone_locations = np.linspace(645.28,980.56,12)
    for i,Trace in enumerate(Stream):
        if i%3 == 0:
            Trace.stats.channel = '2'
        elif i%3 == 1:
            Trace.stats.channel = '1'
        elif i%3 == 2:
            Trace.stats.channel = 'Z'
        Trace.stats.network = network
        Trace.stats.station = '%05d' % (int(np.floor(i/3)))
        Trace.stats.location = '0'
        Trace.stats.distance = geophone_locations[int(np.floor(i/3))]
        Trace.stats.starttime = Trace.stats.starttime+.35
    return Stream

def read(filelist,timerange=[0],dx_in_m=1.02,fo_start_ch=0,fac='None',units='None'):

    def read_das_segy_files(filelist,dx_in_m=1.02,fo_start_ch=fo_start_ch,units=units):
        import obspy.io.segy.core
        stream = obspy.Stream()
        for f in filelist:
            print('Reading '+f)
            d = obspy.io.segy.core._read_segy(f,format='segy',unpack_trace_headers=True)
            stream += populate_das_segy_trace_headers(d,dx_in_m=dx_in_m,fo_start_ch=fo_start_ch,units=units)
        return stream

    def read_geophone_segy_files(filelist):
        import obspy.io.segy.core
        stream = obspy.Stream()
        for f in filelist:
            print('Reading '+f)
            d = obspy.io.segy.core._read_segy(f,format='segy',unpack_trace_headers=True)
            stream += populate_geophone_segy_trace_headers(d)
        return stream

    if 'DAS' in filelist[0]:
        style='das'
        Stream = read_das_segy_files(filelist,dx_in_m=1.02,fo_start_ch=fo_start_ch,units=units)
    else:
        style='geo'
        Stream = read_geophone_segy_files(filelist)

    Stream.merge(fill_value=0)
    if len(timerange)>1:
        Stream.trim(starttime=timerange[0],endtime=timerange[1])

    return Stream


# Plotting

def wiggle(Stream,style=1,skip=10,scale=1.0,fig=None,color='k',zorder=-1):
    '''
    simple 2 styles for wiggle plot
    #skip=10 is default to skip every 10 ch for speed
    #style=2 is a red/blue plot,
    #scale=1.0 is default scaling
    '''
    import matplotlib.pyplot as plt
    if fig==None:
        fig = plt.figure(figsize=(10,10));
    if style==1: # simple lines
        fig = Stream[::skip].plot(fig=fig, type='section',lw=1,
                           alpha=1.0,scale=scale,color=color,
                           offset_min=Stream[0].stats.distance-10,
                           offset_max=Stream[-1].stats.distance+10,handle=True,
                           zorder=zorder);
    if style==2: #red/plot
        fig = Stream[::skip].plot(fig=fig, type='section',
                           alpha=0.5,linewidth=0,fillcolors=('w','k'),
                           scale=scale,offset_min=Stream[0].stats.distance-10,
                           offset_max=Stream[-1].stats.distance+10,handle=True,
                           zorder=zorder);
    plt.gca().set_title(str(Stream[0].stats.starttime.datetime)+' - '+str(Stream[0].stats.endtime.datetime.strftime('%H:%M:%S'))+' (UTC)');
    plt.ylabel('Time [s]');
    plt.xlabel('Linear Fiber Length [km]');
    return fig

def image(Stream,style=1,skip=10,clim=[0],physicalFiberLocations=False,fig=None):
    '''
    simple image plot of das Stream
    #skip=10 is default to skip every 10 ch for speed
    #style=1 is a raw plot, or 2 is a trace normalized plot
    #clim=[min,max] will clip the colormap to [min,max], deactivated by default
    '''
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(10,10))

    if style==1:
        img = stream2array(Stream[::skip]) # raw amplitudes
        clabel = Stream[0].stats.units
    if style==2:
        img = stream2array(Stream[::skip].copy().normalize()) # trace normalize
        clabel = Stream[0].stats.units+' (trace normalized)'

    t_ = Stream[0].stats.endtime-Stream[0].stats.starttime
    if physicalFiberLocations==True:
        extent = [Stream[0].stats.distance/1e3,Stream[-1].stats.distance/1e3,0,t_]
        xlabel = 'Distance relative to wellhead [km]'
    else:
        dx_ = Stream[1].stats.distance - Stream[0].stats.distance
        extent = [0,len(Stream)*dx_/1e3,0,t_]
        xlabel = 'Linear Fiber Length [km]'
    if len(clim)>1:
        plt.imshow(img.T,aspect='auto',interpolation='None',alpha=0.7,
                   origin='lower left',extent=extent,vmin=clim[0],vmax=clim[1]);
    else:
        plt.imshow(img.T,aspect='auto',interpolation='None',alpha=0.7,
                   origin='lower left',extent=extent);
    h=plt.colorbar(pad=0.01);
    h.set_label(clabel)
    plt.ylabel('Time [s]');
    plt.xlabel(xlabel);

    plt.gca().set_title(str(Stream[0].stats.starttime.datetime)+' - '+str(Stream[0].stats.endtime.datetime.strftime('%H:%M:%S'))+' (UTC)');

    plt.tight_layout();

    return fig

def spectra(stream,units='strain-rate',kind='psd',trace=99999):

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import cumtrapz

    if units=='strain':
        # stream.detrend().taper(0.05).integrate()
        oldstream = stream.copy()
        stream = array2stream(cumtrapz(stream2array(stream)),oldstream)

    if trace==99999:
        A = stream2array(stream)*0
        nx,nt = np.shape(A)
        nyq = nt//2
        dt = stream[0].stats.delta
        A = A[:,slice(1,nyq)]
        for i,tr in enumerate(stream.copy()):
            fr = np.fft.fftfreq(nt,d=dt)[slice(1,nyq)]
            A[i,:] = np.fft.fft(tr.data/1e9)[slice(1,nyq)] # convert from ne/s or ne to e/s or e
        if kind=='as':
            sp = np.abs(A.T)
            sp = 10*np.log10(sp)
        if kind=='ps':
            sp = 2*(np.abs(A.T)**2) / (nt**2)
            sp = 10*np.log10(sp)
        if kind=='psd':
            sp = 2*(np.abs(A.T)**2) / (nt**2) * dt * nt
            sp = 10*np.log10(sp)

    else:
        nt = len(stream[trace].data)
        nyq = nt//2
        dt = stream[trace].stats.delta
        fr = np.fft.fftfreq(nt,d=dt)[slice(1,nyq)]
        A = np.fft.fft(stream[trace].data/1e9)[slice(1,nyq)]
        if kind=='as':
            sp = np.abs(A)
            sp = 10*np.log10(sp)
        if kind=='ps':
            sp = 2*(np.abs(A)**2) / (nt**2)
            sp = 10*np.log10(sp)
        if kind=='psd':
            sp = 2*(np.abs(A)**2) / (nt**2) * dt * nt
            sp = 10*np.log10(sp)

    return fr,sp

def plotSpectra(stream,fr,sp,kind='psd',units='strain-rate',cmap='magma',clim=[-70,10],fig=0,trace=99999):

    import matplotlib.pyplot as plt
    import numpy as np

    # manage labeling
    if units=='strain':
        unit_plot = '$\epsilon$'
    else:
        unit_plot = '$\epsilon$/s'
    if kind=='as':
        units_plot = 'Amplitude Spectrum [dB] (rel. 1 '+unit_plot+')'
    if kind=='ps':
        units_plot = 'Power Spectrum [dB rel. 1 ('+unit_plot+')$^2$)'
    if kind=='psd':
        units_plot = 'Power Spectral Density [dB] (rel. 1 ('+unit_plot+')$^2$/Hz)'

    #quick check on shape of spectra computed using utils.spectra
    try:
        ntr,nc = np.shape(sp) # if this works, sp is a 2d numpy.array, so plot an image
        style='image'
    except:
        ntr = len(sp) # otherwise sp is just a vector, so plot the line
        nc = 1
        style='line'

    #plot
    if style=='image':
        if fig==0:
            fig = plt.figure(figsize=(10,6))
        extent=[stream[0].stats.distance/1e3,stream[-1].stats.distance/1e3,min(fr),max(fr)]
        plt.imshow(sp,extent=extent,aspect='auto',origin='lower left',cmap=cmap,vmin=clim[0],vmax=clim[1])
        h=plt.colorbar(label=units_plot,pad=0.01)
        plt.yscale('log')
        plt.xlabel('Distance [km]')
        plt.ylabel('Frequency [Hz]')

    elif style=='line':
        if fig==0:
            fig = plt.figure(figsize=(10,6))
        plt.plot(fr,sp,label=stream[trace].stats.distance)
        plt.xscale('log')
        plt.ylabel(units_plot)
        plt.xlabel('Frequency [Hz]')
        plt.legend()

    plt.title(str(stream[0].stats.starttime.datetime)+' - '+str(stream[0].stats.endtime.datetime.strftime('%H:%M:%S'))+' (UTC)')
    plt.tight_layout()
    # plt.show()
    return fig


# 2. QA/QC Data Tools

def setup(start,end):
    gaugelength     = 10.0
    dx_in_m         = 1.02
    das_units       = 'ne/s'
    geophone_units  = 'm/s^2'
    geophone_fac    = 2.333e-7
    fo_distance_in_well = 996.51 #m
    fo_start_ch     = 197
    files_das = download(get_paths('das',[start,end]))
    files_geo = download(get_paths('geophone',[start,end]))
    das   = read(files_das,dx_in_m=dx_in_m,timerange=[start,end],units=das_units,fo_start_ch=fo_start_ch)
    geo   = read(files_geo,timerange=[start,end],fac=geophone_fac,units=geophone_units)
    return das,geo,gaugelength

def medianSubtract(stream):
    import numpy as np
    arr = stream2array(stream)
    med = np.median(arr,axis=0)
    arr = arr-med
    return array2stream(arr,stream.copy())

def stream2array(stream):
    '''
    populates a 2D np.array that is the traces as rows by the samples as cols
    '''
    import numpy as np
    import obspy
    array=np.empty((len(stream),len(stream[0].data)),dtype=float) # initialize
    for index,trace in enumerate(stream):
        array[index,:]=trace.data
    return array

def array2stream(array,oldStream):
    import numpy as np
    import obspy
    stream = obspy.Stream()
    for i,row in enumerate(array):
        trace = obspy.Trace(data=row)
        trace.stats = oldStream[i].stats
        stream += trace
    return stream

def mask(stream,ch1,ch2):
    stream2 = stream.copy()
    for tr in stream2[ch1:ch2]:
        tr.data *= 0
    return stream2