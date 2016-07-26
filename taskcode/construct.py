'''
Module to construct full feature grid (as a dataframe or 2D array)

Will handle various attempts at dimensionality reduction that we come up with by hand.
'''

import os
import glob
import pickle
import itertools
import sys
import pandas as pd
import numpy as np
import scipy.signal as signal

from IPython import embed

def gps_transform(func):
    '''
    Decorator to mark function for use in gps dim reduction.

    These functions take a dataframe of gps data matched to a task
    and return a _list_ of _pd.Series_ which are the features for each generated row
    '''
    def feature_func(df, **kwargs):
        # These are all the same so max just gets a single value
        default_features = pd.Series(dict(label=df.task_label.max(), name=df.name.max(),
                                          start_time=df.start_time.max(), end_time=df.end_time.max(),
                                          skill=df.skill.max(), room=df.room.max(), last_task=df.last_task.max(),
                                          ntask_1_completed=df.ntask_1_completed.max(),ntask_2_completed=df.ntask_2_completed.max(),
                                          ntask_3_completed=df.ntask_3_completed.max(),ntask_4_completed=df.ntask_4_completed.max(),
                                          ntask_5_completed=df.ntask_5_completed.max(),ntask_6_completed=df.ntask_6_completed.max(),
                                          ntask_7_completed=df.ntask_7_completed.max(),ntask_8_completed=df.ntask_8_completed.max(),
                                          ntask_9_completed=df.ntask_9_completed.max(),ntask_10_completed=df.ntask_10_completed.max(),
                                          ntask_11_completed=df.ntask_11_completed.max()))
        rows = []
        for row in func(df, **kwargs):
            rows.append(pd.concat((default_features, row)))
        return rows
    gps_transform.funcs[func.__name__] = feature_func
    return feature_func

gps_transform.funcs = {}


def cached(func):
    '''
    Decorator to optionally cache a DataFrame to file .

    This decoration relies on func taking only simple (hashable) kwargs
    Using pickle for now. If performance gets slow consider msgpack or hdf5
    '''
    def cached_func(*args, **kwargs):
        c = kwargs.pop('cache', False)
        context = "data/{}.pkl".format(hash(frozenset(kwargs.items())))
        if c:
            # Attempt to retrieve from file
            if os.path.exists(context):
                return pd.read_pickle(context)
        df = func(*args, **kwargs)
        if c:
            # Load to file if cache specified
            df.to_pickle(context)
        return df
    return cached_func


@gps_transform
def padded(df):
    '''
    Retain all of the x,y,z as features, padded into an 8 hour window.
    '''
    # Get times as seconds since start and use as index
    df.index = ((df.position_update_timestamp - df.position_update_timestamp.min())/pd.Timedelta('1s')).astype('int') 
    # Drop everything but x,y,z
    df = df['position_x','position_y','position_z']

    # Reindex to pad out to a length of 8h
    df.reindex(np.arange(8*3600, fill_value=0))

    # Now build a feature vector from it. It'll be big :(
    return [pd.concat(df.position_x, df.position_y, df.position_z)]
    

def distance(*args):
    '''
    N-dim euclidean distance
    '''
    return np.sqrt(np.sum(a*a for a in args))

@gps_transform
def chunked(df, **kwargs):
    '''
    Extract a few statistical values from time data chunked over an interval (1m).
    Keep an hour's worth.
    '''
    # First calcluate any new values
    # Velocity is distance between consecutive points per sec
    df['velocity'] = distance(df.position_x.diff(), df.position_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_x'] = distance(df.position_x.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_y'] = distance(df.position_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['acceleration'] = distance(df.velocity_x.diff(), df.velocity_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    # List of columns to form features
    # cols = ['position_x','position_y','position_z','velocity']
    cols = ['position_x', 'position_y','velocity', 'acceleration']
    # Apparently sometimes the gps data is not consecutive in seconds
    # so we need to focus on timestamps and not indices
    interval = pd.Timedelta(kwargs.pop('interval','10m')) # Length of interval for each output row
    sub_interval = pd.Timedelta(kwargs.pop('subinterval','2m')) # Sub interval in which to sample derived quantities
    dens = float(kwargs.pop('dens','1.0'))
    n_sub = int(interval/(dens*sub_interval))

    rows = []

    # Create a chunk containing all timestamps within one interval
    lower = df.position_update_timestamp.min() 
    upper = lower + interval
    chunk = df[(df.position_update_timestamp > lower) & (df.position_update_timestamp < upper)].copy()

    while len(chunk):
        # Calculate values in the sub intervals for this chunk.
        means = []
        moveon=False
        for col in cols:
            mean = chunk[col].groupby(((chunk.position_update_timestamp - chunk.position_update_timestamp.min())/sub_interval).astype(int)).mean()
            if (len(mean) < n_sub) or (mean.var()==0.0):
                moveon=True
            mean = mean.reindex(range(int(interval/sub_interval)), method='nearest')
            means.append(mean)
        stds = []
        # print moveon
        for col in cols:
            std = chunk[col].groupby(((chunk.position_update_timestamp - chunk.position_update_timestamp.min())/sub_interval).astype(int)).std()
            std = std.reindex(range(int(interval/sub_interval)), method='nearest')
            stds.append(std)
        features = pd.concat((pd.concat(means), pd.concat(stds)))
        features.index = range(len(features))
        # Get the next chunk
        lower,upper = lower+interval, upper+interval
        chunk = df[(df.position_update_timestamp > lower) & (df.position_update_timestamp < upper)].copy()
        if not moveon: rows.append(features)
    #if len(rows): embed()
    return rows



def periodicity(df, col):
    x = df[col]
    t = (df.position_update_timestamp - df.position_update_timestamp.min())/pd.Timedelta('1s')
    
    # We sample at 1 hz best, so we can't measure any components above 0.5 hz
    # We can in principle see very low frequency all the way down to dc
    # but we can't tell the difference between dc and an 8h frequency
    
    # If we chunk down to 30m or 1h, 1e-4 is a better lower bound
    f = np.logspace(-4, np.log10(0.5), 10000)# in hertz
    try:
        return f, signal.lombscargle(t.values, x.values.astype(float), f)
    except ZeroDivisionError:
        return [], []
        


def peak_frequencies(f, p, n=5):
    if len(f) == 0:
        return np.zeros(n), np.zeros(n)
    p/=p.max()
    arm = signal.argrelmax(p)
    fm, pm = f[arm], p[arm]
    sort = np.argsort(pm)[::-1]
    maxes, periods = pm[sort], 1.0/fm[sort]
    # Get rid of any above 95% of max, that is almost definitely part of the DC with 
    # some noise or float errors causing it to show up as a relative maximum
    maxes, periods = maxes[maxes < 0.95], periods[maxes < 0.95]

    # If we don't have any peaks at all, just give zeros
    if len(maxes) == 0:
        maxes = np.zeros(n)
        periods = np.zeros(n)
    # In case we have less than n, repeate the last one
    else:
        maxes = maxes[:n] if len(maxes) >= n else np.append(maxes, maxes[-1]*np.ones(n-len(maxes)))
        periods = periods[:n] if len(periods) >= n else np.append(periods, periods[-1]*np.ones(n-len(periods)))
    return maxes, periods


@gps_transform
def derived(df, **kwargs):
    '''
    Extract many statistical values from time data
    '''
    
    # Require a minimum number of gps points
    print len(df)
    if len(df) < 10:
        return []

    # Variables defined at each time point
    # Distance from center
    df['distance'] = distance(df.position_x - df.position_x.mean(), df.position_y - df.position_y.mean())
    df['distance_x'] = distance(df.position_x - df.position_x.mean())
    df['distance_y'] = distance(df.position_y - df.position_y.mean())
    
    # First derivative in various styles
    df['velocity'] = distance(df.position_x.diff(), df.position_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_x_rms'] = distance(df.position_x.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_y_rms'] = distance(df.position_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_x'] = df.position_x.diff()/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['velocity_y'] = df.position_y.diff()/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))

    # Second derivative in various styles
    df['acceleration'] = distance(df.velocity_x.diff(), df.velocity_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['acceleration_x_rms'] = distance(df.velocity_x.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['acceleration_y_rms'] = distance(df.velocity_y.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['acceleration_x'] = df.velocity_x.diff()/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))
    df['acceleration_y'] = df.velocity_y.diff()/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))    

    # More ideas?
    # We could potentially find rooms and get centers or edges or similar

    cols = ['position_x', 'position_y', 'distance', 'distance_x', 'distance_y',
            'velocity', 'velocity_x_rms', 'velocity_y_rms', 'velocity_x', 'velocity_y', 
            'acceleration', 'acceleration_x_rms', 'acceleration_y_rms', 'acceleration_x', 'acceleration_y']

    interval = pd.Timedelta(kwargs.pop('interval','10m')) # Length of interval for each output row

    rows = []

    # Create a chunk containing all timestamps within one interval
    lower = df.position_update_timestamp.min() 
    upper = lower + interval
    chunk = df[(df.position_update_timestamp > lower) & (df.position_update_timestamp < upper)].copy()

    while len(chunk):
        # Calculate values in the sub intervals for this chunk.

        # Stat variables from some columns
        means = pd.Series(chunk[col].mean() for col in cols)
        stds = pd.Series(chunk[col].std() for col in cols)

        # FT variables
        fft_max, fft_period = [], []
        for col in ['position_x', 'position_y']:
            maxes, periods = peak_frequencies(*periodicity(chunk, col))
            fft_max.extend(maxes)
            fft_period.extend(periods)
        fft_max, fft_period = pd.Series(fft_max), pd.Series(fft_period)
        
        features = pd.concat((fft_max, fft_period, means, stds))
        features.index = range(len(features))
        rows.append(features)

        # Get the next chunk
        lower,upper = lower+interval, upper+interval
        chunk = df[(df.position_update_timestamp > lower) & (df.position_update_timestamp < upper)].copy()
        
    return rows


@gps_transform
def consecutives():
    '''
    Map gps to a list of consecutive stationary points (x,y,z,t,v)
    where t is total time at that location and v is velocity to next
    '''
    raise KeyError("consecutives not yet implemented")
    
def create_gps_pickles():
    '''
    Create a series of files that store the subset of gps data for each distinct task.
    '''
    
    # First we load the timestamps DF (just 100 for testing)
    # df = pd.read_pickle('data/TaskCodeTimestamps.pkl')[:100]
    df = pd.read_pickle('data/TaskCodeTimestamps.pkl')
    df['duration'] = (df.end_time - df.start_time) / pd.Timedelta('1h')    
    df = df[df.duration <= 8]
    sizes = df.groupby(df.task).size()
    common = sizes[sizes > 10].index
    df = df[df.task.isin(common)]
    
    # Grab gps data for each task, processing if specified
    gps = pd.read_pickle('data/LocationData.pkl')
    with open('data/NameToNode.pkl','r') as infile:
        nodes = pickle.load(infile)
    nodes = dict((int(key), val) for key,val in nodes.iteritems())

    df['name'] = df.first_name + ' ' + df.last_name
    gps['name'] = gps.node_id.map(nodes)

    # Build an extension to the df by creating feature vectors from (transformed) x,y,z data
    for index, name, start, end, task, room in itertools.izip(df.index, df.name, df.start_time, df.end_time, df.task, df.room):
        sys.stdout.write('Processing task number {0} out of {1}\r'.format(index,df.index[-1]))
        sys.stdout.flush()
        sub_gps = gps[(gps.name == name) & (gps.position_update_timestamp > start) & (gps.position_update_timestamp < end)].copy()
        if not len(sub_gps):
            continue
        # Get completed tasks information
        fin_bool = (df.loc[df.room==room].end_time < start)
        finished = df.loc[df.room==room].loc[fin_bool]
        finished_task_counts = finished.groupby('task').size()
        finished_task_counts = finished_task_counts.reindex(range(1,12),fill_value=0)
        try:
            last_task = finished.sort_values('end_time').iloc[-1].task
        except:
            last_task = 0
        sub_gps['start_time'] = start
        sub_gps['end_time'] = end
        sub_gps['task_id'] = index
        sub_gps['task_label'] = task
        sub_gps['room'] = room
        sub_gps['skill'] = name.split(" ")[0]
        sub_gps['last_task'] = last_task
        for i in finished_task_counts.index:
            sub_gps['ntask_'+str(i)+'_completed'] = finished_task_counts[i]
        sub_gps.to_pickle('data/gps_{:06d}.pkl'.format(index))
    print


@cached
def load_tasks(gps_reduce='chunked', accel_reduce=None, interval='60m', subinterval='1m', dens='1.0', n=None, categories=True):
    '''
    Return a pandas data frame that stores the features and labels for each task.

    For now only gps data is supported.

    Kwargs:
    gps_reduce -- name of the transform of gps three vectors into reduced vectors (any number of outputs)!
                  Accepts the name of any function marked @gps
    '''

    # First determine the indices from the pickle files
    fnames = glob.glob('data/gps_*.pkl')
    if n is not None:
        fnames = fnames[:n]
    
    # Because each of the input files can generate multiple rows depending on 
    # the choice of transform, store all as a list first
    reduced_gps = list()
    for i,fname in enumerate(fnames):
        sys.stdout.write('Processing file number {0} out of {1}\r'.format(i,len(fnames)))
        sys.stdout.flush()
        reduced_gps.append(gps_transform.funcs[gps_reduce](pd.read_pickle(fname), interval=interval, subinterval=subinterval, dens=dens))
    rows = list(itertools.chain.from_iterable(reduced_gps))
    df = pd.DataFrame(index=range(len(rows)), columns=rows[0].index)
    for i,row in enumerate(rows):
        df.iloc[i] = row
    # Some final processing! We want to add features to encode the categorical variables as dummies.
    if categories:
        df = pd.concat((df, pd.get_dummies(df.skill)), axis=1)
        df = pd.concat((df, pd.get_dummies(df.room).rename(columns=lambda x: 'room_'+str(x))), axis=1)

    return df


def to_array(df):
    df = df.drop(['end_time','name','skill','start_time','room'],axis=1).astype(float)
    df_sr = df.iloc[:,253:]
    df_no_se = df.iloc[:,0:133]
    #df=df_no_se
    df=pd.concat((df_no_se,df_sr),axis=1)
    df[df.isnull()] = 0.0
    # Short term, use only tasks with more than min_count examples
    min_count = 30
    counts = df.label.groupby(df.label).count()
    print counts
    labels_above_min = counts > min_count
    df = df[df.label.isin(labels_above_min[labels_above_min].index)]
    #df = df[df.label.isin([2,3,4])]
    X = df.iloc[:,1:].astype(float)
    y = df.label.values.astype(int)
    return X, y

   
 
def main():
    '''
    Make the gps pickles which are used by other methods.
    '''
    create_gps_pickles()
    #df=load_tasks(cache=False)
                       
if __name__ == "__main__":
    main()
