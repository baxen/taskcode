'''
Module to construct full feature grid (as a dataframe or 2D array)

Will handle various attempts at dimensionality reduction that we come up with by hand.
'''

import os
import glob
import pickle
import itertools

import pandas as pd
import numpy as np

def gps_transform(func):
    '''
    Decorator to mark function for use in gps dim reduction.

    These functions take a dataframe of gps data matched to a task
    and return a _list_ of _pd.Series_ which are the features for each generated row
    '''
    def feature_func(df):
        default_features = pd.Series(dict(label=df.task_label.max(), name=df.name.max(), start_time=df.start_time.max(), end_time=df.end_time.max())) # These are all the same so max just gets a single value
        rows = []
        for row in func(df):
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
        context = "{}.pkl".format(hash(frozenset(kwargs.items())))
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
def chunked(df):
    '''
    Extract a few statistical values from time data chunked over an interval (1m).
    Keep an hour's worth.
    '''
    # First calcluate any new values
    # Velocity is distance between consecutive points per sec
    df['velocity'] = distance(df.position_x.diff(), df.position_y.diff(), df.position_z.diff())/(df.position_update_timestamp.diff()/pd.Timedelta('1s'))

    # List of columns to form features
    cols = ['position_x','position_y','position_z','velocity']

    # Apparently sometimes the gps data is not consecutive in seconds
    # so we need to focus on timestamps and not indices
    interval = pd.Timedelta('20m') # Length of interval for each output row
    sub_interval = pd.Timedelta('1m') # Sub interval in which to sample derived quantities

    rows = []


    # Create a chunk containing all timestamps within one interval
    lower = df.position_update_timestamp.min() 
    upper = lower + interval
    chunk = df[(df.position_update_timestamp > lower) & (df.position_update_timestamp < upper)].copy()

    while len(chunk):
        # Calculate values in the sub intervals for this chunk.
        mean = pd.concat(df[col].groupby(((df.position_update_timestamp - df.position_update_timestamp.min())/pd.Timedelta('1m')).astype(int)).mean() for col in cols)
        std = pd.concat(df[col].groupby(((df.position_update_timestamp - df.position_update_timestamp.min())/pd.Timedelta('1m')).astype(int)).std() for col in cols)
        features = pd.concat((mean, std))
        features[features.isnull()] = 0
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
    df = pd.read_pickle('TaskCodeTimestamps.pkl')[:100]

    # Grab gps data for each task, processing if specified
    gps = pd.read_pickle('LocationData.pkl')
    with open('NameToNode.pkl','r') as infile:
        nodes = pickle.load(infile)
    nodes = dict((int(key), val) for key,val in nodes.iteritems())

    df['name'] = df.first_name + ' ' + df.last_name
    gps['name'] = gps.node_id.map(nodes)

    # Build an extension to the df by creating feature vectors from (transformed) x,y,z data
    for index, name, start, end, task in itertools.izip(df.index, df.name, df.start_time, df.end_time, df.task):
        sub_gps = gps[(gps.name == name) & (gps.position_update_timestamp > start) & (gps.position_update_timestamp < end)].copy()
        if not len(sub_gps):
            continue
        sub_gps['start_time'] = start
        sub_gps['end_time'] = end
        sub_gps['task_id'] = index
        sub_gps['task_label'] = task
        sub_gps.to_pickle('gps_{:06d}.pkl'.format(index))

@cached
def load_tasks(gps_reduce='chunked', accel_reduce=None, interval=None, n=None):
    '''
    Return a pandas data frame that stores the features and labels for each task.

    For now only gps data is supported.

    Kwargs:
    gps_reduce -- name of the transform of gps three vectors into reduced vectors (any number of outputs)!
                  Accepts the name of any function marked @gps
    '''

    # First determine the indices from the pickle files
    fnames = glob.glob('gps_*.pkl')
    if n is not None:
        fnames = fnames[:n]
    
    # Because each of the input files can generate multiple rows depending on 
    # the choice of transform, store all as a list first
    rows = list(itertools.chain.from_iterable(gps_transform.funcs[gps_reduce](pd.read_pickle(fname)) for fname in fnames))
    df = pd.DataFrame(index=range(len(rows)), columns=rows[0].index)
    for i,row in enumerate(rows):
        df.iloc[i] = row
    return df
    
    
def main():
    '''
    Make the gps pickles which are used by other methods.
    '''
    create_gps_pickles()


if __name__ == "__main__":
    main()
