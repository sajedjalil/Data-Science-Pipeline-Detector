import pandas as pd
import glob
import os
import numpy as np
import gc
import logging
import mne

logger = logging.getLogger('__SignalManager__')
  
"""Utility that acts as a wrapper around the Python Pandas package, which is useful for handling time series data efficiently.
   In particular, the management tool helps to sanity check, align and utilise experimentally relevant epochs in multi channel data"""

__author__ =  'Andrew O\Harney'

class SignalManager:
    """This class manages signals in edf,fif,or hd5 format (note all files are converted to hd5)
    It makes extensive use of pandas to represent time series and events logs
    
    Requirements - pandas,numpy,matplotlib
                 - mne (utility functions)
                 - matplotlib (utility functions/plotting)"""
    
    
    ############Member Variables################
    __base_file_name = None
    __signals = None #Pandas Data frame columns=chans, index = times
    __log_file = None #Events log file
    __wd = None #Working data
    __wc = None #Working Channels
    __eventskeys = None #Codes for events in log
    __currentMeanCalcChans = None #Channels used to calculate current mean
    __currentMeanApplyChans = None #Channels means were applied to
    
    ###############Private Methods##########################
    
    
    def __init__(self,base_file_name=None,log_file=None,offsets=None,new_log_out=False,eventsKey=None):
        """Initialiser takes the path of the signal data
        #Can also set the event matrix or generate it from a log path
        
        Keyword arguments:
        base_file_name=None -- The path and name of the signal data with no file extension
        log_file=None -- Path to log file of events
        offsets=None -- If a log file has been provided, startimes is a path to a file containing appropriate offets for each block of events (if required - often useful for alignment)
        new_log_out=None -- If offsets is specified then new_log_out is a Boolean value if the corrected file is to be output
        eventsKey=None -- If special event codes are required (i.e not from the csv file)"""
        
        self.__base_file_name = base_file_name
        logger.debug('Using file : %s'%(base_file_name))
        self.__load_data__()      
        
        if eventsKey is None:
            eventsKey = {'_':0,'blockStart':1}
        self.set_eventsKey(eventsKey)
        
        #Check for log file to create event matrix
        if log_file:
            self.set_log_file(log_file,offsets,new_log_out)
        else:
            logger.info( 'No log specified -- assuming event matrix is in the data')
    

    def __load_data__(self):
        """Attempts to load data from .hd5
        If .hd5 file does not exist, it will try to convert to it"""

        if self.base_file_name() is None:
            raise Exception('Data was not specified')
        elif self.__check_for_files__('hd5'): #HD5 is the basis for pytables
            logger.info( 'Found .hd5 -- opening')
        elif self.__check_for_files__('fif'):
            logger.info( 'Could not find .hd5 -- converting  .fif->.hd5')
            self.__fif_2_hdf5__()
        elif self.__check_for_files__('edf'):
            logger.info( 'Could not find .hd5 -- converting .edf->.fif->.hd5')
            self.__edf_2_fif__()
            self.__fif_2_hdf5__()
        else:
            logger.info( "Could not find any appropriate files. Valid files are *.[edf, fif, hd5]. Assuming data will be supplied later")
        self.__open_hdf5__()
        #except Exception as e: raise e

    def __check_for_files__(self, ftype):
        """Returns a list of files in the local Directory containing the file type
        Keyword Arguments:
        ftype -- Type of file to check for"""
        return glob.glob(self.__base_file_name + '.' + ftype)
    
    def __open_hdf5__(self):
        """Tries to open hd5 file"""
        try:
            self.__signals = pd.HDFStore(self.__base_file_name+'.hd5')
        except:
            logger.warning('Could not open hd5 file')
            raise Exception('Could not open hd5 file')

    def __edf_2_fif__(self):
        """"Tries to convert edf to fif"""
        
        sysString = 'mne_edf2fiff --edf '+self.base_file_name()+'.edf --fif ' + self.base_file_name()+'.fif'
        logger.info('System Command : %s'%(sysString))
        try:
            os.system(sysString)
            logger.info( 'Conversion edf->fif complete')
        except:
            logger.warning('Could not find mne on system path')
            raise Exception('Could not find mne on system path -- cannot convert from .edf')
    
    def removeNonImportant(self,importantEventCodes):
        #NEED TO THINK ABOUT THIS ONE
        pass
        #Will remove parts of the signal not important to the experiment
        em = self.event_matrix()
        em = em[em['event.code'].isin(importantEventCodes)]
        exptimes = self.eventsTimes(em)
        self.__signals['Data'] = self.__signals['Data'].ix[exptimes]
        self.__signals.flush()
    
       
    def __fif_2_hdf5__(self):
        """Tries to convert .fif file to .hd5 format"""
        
        #Get data from the raw .fif file
        try:
            raw = mne.fiff.Raw(self.__base_file_name+'.fif')
        except:
            logger.warning('Could not open fif file')
            raise Exception("Could not open fif file")
        
        logger.debug( 'Extracting data from .fif')
        data,time_stamps = raw[1:,:raw.last_samp]
        ch_names = raw.ch_names[1:]
        logger.debug('Found channels : %s'%(str(ch_names)))
        fs = raw.info['sfreq']
        logger.debug('Found frequency : %f'%(fs))
        raw.close()
        self.save_hdf(data,time_stamps,ch_names,fs,self.base_file_name())
        self.__open_hdf5__()

    def __create_events_matrix__(self):
        """Creates a Dataframe with index=data timestamps times, columns=signal channels"""
        logger.info( "Generating event matrix")
        events = pd.read_csv(self.__log_file,delimiter=r'\t|,')
        logger.debug( 'Found columns:'+str(events.columns))
        self.__signals['event_matrix'] = events
        self.__flushSignals__()
        logger.info( "Saving event matrix")
        self.__find_blocks__()
          

    def __find_blocks__(self):
        """Finds the on and off times of blocks
        Note: Blocks are defined as starting at event types blockStart(event id 1)"""
        
        logger.info( "Finding blocks")
        logger.debug( '\tCalculating block indices')
        em = self.event_matrix()
        blockStartIndices = em[em['event.code'] == self.__eventskey['blockStart']].index #Start of each block
        logger.debug(blockStartIndices)
    
        blockEndIndices = blockStartIndices
        blockEndIndices = blockEndIndices[1:].values - 2 #Remove the first pulse and shift to become last pulse in each preceding block
        blockEndIndices = np.append(blockEndIndices, len(em) - 1) #Add final pulse in file
        #Define the times of each block
        logger.debug( '\tCalculating start and end times of each block')
        
        startTimes = em.ix[blockStartIndices]['pulse.on'].values
        endTimes = em.ix[blockEndIndices]['pulse.off'].values
        logger.debug('Start times '+str(startTimes))
        logger.debug('End times '+str(endTimes))
        blocks = pd.DataFrame([startTimes,endTimes])
        blocks = blocks.T
        blocks.columns=['pulse.on','pulse.off']
        logger.info( "Saving blocks")
        self.__signals['blocks'] = blocks
        self.__flushSignals__()
    
    def __flushSignals__(self):
        """"Forces a write to the hdf file"""
        self.__signals.flush()
           
    ###############Public Methods#################################

    def set_eventsKey(self,eventsKey):
        """Set a dictionary containing event code descriptions
        Keyword Arguments:
        eventsKey -- Dictionary containing event codes and label names"""
        self.__eventskey = eventsKey
    
    def eventsKey(self):
        """Get events key"""
        return self.__eventskey    
    

    @staticmethod
    def save_hdf(data,times,cnames,fs,base_file_name):
        """Takes raw data and saves to HD5
        Keyword arguments:
        #data - raw signal data
        #cnames - channel names
        #base_file_name - base file name
        #fs - sample rate of data"""
        
        (x,y)= data.shape
        #Store in hd5(pytables) format
        logger.info( "Converting to pytables")
        #signals = pd.HDFStore(base_file_name+'.hd5','w',complevel=9)
        signals = pd.HDFStore(base_file_name+'.hd5','w')
        #
        logger.debug( '\tSaving timing info')
        signals['times'] = pd.Series(times,dtype='float64')
        #
        logger.debug( '\tSaving data')
        signals['data']=pd.DataFrame(data.T,columns=cnames,index=times) #Ideally this would be tables=True
        #        
        logger.debug( "\tSaving meta data")
        signals['channels'] = pd.Series(cnames)
        signals['fs'] = pd.Series(fs)
        #signals['data_dimensions'] = pd.Series(['channels', 'samples'])
        signals.close()
        logger.info( 'Conversion complete')
    

    def add_channel(self,sig,name):
        """Adds a channel to the hd5 file
        
        Keyword Arguments:
        sig -- the raw signal to be added
        name -- name of the new channel"""
        
        if name not in self.channels():
            newData = self.data()
            newData[name] = pd.Series(sig,name=[name],index=self.data().index)
            self.__signals['data']  = newData
            self.__signals['channels'] = self.channels().append(pd.Series(name,index=[len(self.channels())]))
            self.__signals.flush()
        else:
            logger.info( 'Channel with that name already exists')
             
    def remove_channel(self,chan):
        """Removes channel chan from the persistent .hd5 file
        Keyword Arguments:
        chan -- Channel to be removed"""
        
        #Try and remove the specified channel
        try:
            self.__signals['data'] = self.data().drop(chan,axis=1)
            #Remove from channel record
            self.__signals['channels'] = self.channels()[self.channels()!=chan]
            self.__flushSignals__()
        
            #If the channel was in the current working set
            currentChan = self.wc()
            if chan in currentChan:
                currentChan.remove(chan)
            
                if self.__currentMeanCalcChans is not None and chan in self.__currentMeanCalcChans:
                    self.set_wd(currentChan,meanCalcChans=[mc for mc in self.__currentMeanCalcChans if mc != chan],meanApplyChans=[mc for mc in self.__currentMeanApplyChans if mc != chan])
                else:   
                    self.set_wd(currentChan)
        except:
            logger.info( 'No channel called '+chan)
        
 
    #################Public Methods#########################   
    def blocks(self):
        """Return the on and off times of blocks"""
        return self.__signals['blocks']
    
    
    def base_file_name(self):
        """Return the base file path"""
        return self.__base_file_name
    
    
    def log_file_name(self):
        """"Returns the log path of the psychopy file in use (i.e the file the event_matrix was generated from)"""
        return self.__log_file
    
    def event_matrix(self,types=None):
        """Returns the event matrix
        
        Keyword Arguments:
        types -- Event types (codes to return) - Still to be implemented
        """
        return self.__signals['event_matrix']

        
    def calc_mean(self,channels):
        """Return the mean channel (i.e the mean power across all channels for a given time point)
        channels -- Channels to calculate the mean over"""
    
        return pd.Series(self.data(channels=channels).mean(axis=1),index=self.times())
    
    
    def set_mean(self,meanCalcChans=None,meanApplyChans=None):
        """Remove a mean value from channels
        meanCalcChans=None : channels to calculate the mean from - mean will be applied to these channels unless meanApplyChans is specified
        meanApplyChans=None : channels to apply the mean to  (default is meanCalcChans)"""
        
        if meanCalcChans is not None:
            logger.info( 'Calculating mean')
            m = self.calc_mean(meanCalcChans)
            permMeanChans = []
            for chan in meanApplyChans if (meanApplyChans is not None) else meanCalcChans:
                logger.debug('Calculating mean for channel '+chan)
                self.__wd[chan] -= m #Cannot use .sub() - blows up!
                permMeanChans.append(chan)
            self.__currentMeanCalcChans = permMeanChans
            return m
        

    def data(self,channels=None):
        """Efficiently get data chunks from disk by supplying a column list (Note:data must be in table format)
        Keyword Arguments: 
        channels=None -- The channels to pull from the data (This should be done efficiently but needs to be reviewed)"""
        if channels is not None:
            try:
                #Efficient read from disk (no need to load all data in memory) if data is in table format
                d = self.__signals.select('data', [pd.Term('columns','=',channels)])
            except:
                #If in pytables format then we need to load all data into memory and clip
                d= self.__signals['data'][channels]
        else:
            d= self.__signals['data']
        
        gc.collect() #Do garbage collection to free up wasted memory
        return d
    
    def times(self):
        """Returns the timestamps of samples"""
        
        return self.__signals['times']
    
    def channels(self):
        """Returns all channel names"""
        
        return self.__signals['channels']

    def wd(self,channels=None):
        """Return the current working data
        Keyword arguments:
        channels=None -- Channels to pull from working data"""
        if self.__wd is not None:
            return self.__wd[self.wc() if channels is None else channels]
        else:
            logger.debug( "No working data was set")
       
    def fs(self):
        """Return the sample rate of the signal"""
        
        return self.__signals['fs'][0]
    

    def correct_event_times(self,offsets,new_log_out=False):
            """Correct the event matrix to include the appropriate block offsets
            #Keyword Arguments
            Offsets -- A Pandas Dataframe or Series with ['time'] offsets for each block
            new_log_out=None -- Boolean value if the corrected file is to be output"""
    
            logger.info( 'Correcting times in log file')
            offsets = pd.read_csv(offsets,sep='\t')
            blocks = self.blocks()
            startTimes = blocks['pulse.on']
            
            offsets = pd.Series(offsets['time']-startTimes,index=range(len(offsets))) #Remove the psychopy start time from the offset
            #Correct block times by the offsets
            logger.debug( "\tCorrecting blocks data")
            blocks['pulse.on']+=offsets
            blocks['pulse.off']+=offsets
            self.__signals['blocks'] = blocks
            logger.debug(blocks)
                
            logger.debug( '\tCorrecting event times')
            em = self.event_matrix()
            for i,block in enumerate(em['Block'].unique()):
                em.ix[em['Block']==block,'pulse.on']+= offsets.ix[i]
                em.ix[em['Block']==block,'pulse.off']+= offsets.ix[i]
            
            self.__signals['event_matrix'] = em
            self.__flushSignals__()
        
            if new_log_out:
                logger.info( "Saving corrected log file")
                self.__signals['event_matrix'].to_csv(self.__base_file_name+'_corrected_log.csv')
                self.__log_file = self.__base_file_name+'_corrected_log.csv'
            
    def set_log_file(self,log,offsets=None,new_log_out=False):
        """Sets the psychopy log file
        
        Keyword Arguments: 
        log -- path to log file
        offsets=None -- path to file contains offsets of block times
        new_log_out=False -- Boolean value if the corrected file is to be output"""
        
        logger.info( 'Saving log file')
        self.__log_file = log
        self.__create_events_matrix__()
        if offsets:
            self.correct_event_times(offsets,new_log_out)

    def set_wd(self,channels=None,meanCalcChans=None,meanApplyChans=None):
        """Sets the working data to the selected channels (selects all channels by default)
        Keyword Arguments:
        channels=None -- list of channels to use
        meanChans=None -- The channels to calculate the mean from
        meanApplyChans=None - The channels to apply the mean to (default = meanCalcChans)"""
    
        logger.info( "Loading working data")
        self.__wd = self.data(channels=channels if channels else self.channels())
        self.__wc = channels if channels else self.channels()

        if meanCalcChans is not None:
            self.set_mean(meanCalcChans, meanApplyChans)
    
    def wc(self):
        """Returns a list of channel names for the working data"""
        return list(self.__wc)
    
    def set_fs(self,fs):
        """Set the frequency that the data was sampled at
        Keyword Arguments:
        fs -- sample frequency"""
        self.__signals['fs']=fs
        self.__flushSignals__() 

    def splice(self,data=None,times=None,indices=None):
        """Returns the signal specified between two time points
        data=None -- The data to splice (default is the whole data set)
        times=None -- The start and end times to splice between
        indices=None -- The start and end indices"""
        
        if data is None:
            data = self.wd()
        
        if times:
            return data.ix[self.snap_time(min(times)):self.snap_time(max(times))].values[:-1]
        elif indices:
            return data.iloc[indices]
    
    def eventsTimes(self,events,limit=None):
        """Get combined time indices of each event period
        events - Events to index
        limit=None - Specify cut-off for each event in seconds"""
        
        limit = None if limit is None else int(limit*self.fs())
        allTimes = np.array([])
        for i in range(len(events)):
            x = self.event_times(event=events.iloc[i])[:limit]
            allTimes = np.hstack([allTimes,x])    
        return allTimes
    
    #Timing functions
    def snap_time(self,t):
        """Finds the nearest time
        Keyword Arguments:
        t - Time to snap to
        """
        
        return self.time_to_index(t)/float(self.fs())
        #return self.times()[self.time_to_index(t)]

    def index_to_time(self,ix):
        """Returns the time of a given index
        Keyword Arguments:
        ix - Index of time"""
        return self.times().iloc[ix]

    def time_to_index(self,t):
        """Returns the index of a given time point
        Keyword Arguments:
        t -- Time point"""
        return int(np.floor(t*float(self.fs())))
    
    def event_times(self,event=None,times=None):
        """Returns a full list of event sample times
        Keyword Arguments:
        event=None -- Event to retrieve time points of
        times=None -- Start and stop times to find times between
        """
        if event is not None:
            [start,stop] = event[['pulse.on','pulse.off']].values
        elif times is not None:
            start = np.min(times)
            stop = np.max(times)
        else:
            logger.debug('No event or time was supplied')
            return None
        
        return self.times()[self.time_to_index(start):self.time_to_index(stop)]
    
    def event_data(self,event,chans=None):
        """Returns the data for a given event
        Keyword Arguments:
        event -- Event to get data from
        chans=None -- Channels to get from"""
        if chans is None:
            chans = self.wc()
    
    def num_points(self,event=None,times=None):
        """Returns the number of (inclusive) samples between two data points
        Keyword Arguments:
        event -- Event to get number of points of
        times -- start and stop time to get number of points between"""
        #If fs is specified then use that, otherwise will need to snip a section and check the length
        if event is not None:
            return self.time_to_index(event['pulse.off'])-self.time_to_index(event['pulse.on'])
        elif times is not None:
            return self.time_to_index(max(times))-self.time_to_index(min(times))
        else:
            logger.debug('No event or times were supplied ')
            return None
    