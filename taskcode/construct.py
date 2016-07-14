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

def gps_transform(func):
    '''
    Decorator to mark function for use in gps dim reduction.

    These functions take a dataframe of gps data matched to a task
    and return a series which are the features for that task
    '''
    def feature_func(df): # Accepts just a single dataframe!
        features = pd.Series(dict(label=df.task_label.max(), name=df.name.max(), start_time=df.start_time.max(), end_time=df.end_time.max())) # These are all the same so max just gets a single value
        features = pd.concat((features, func(df)))
        return features
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
    return pd.concat(df.position_x, df.position_y, df.position_z)
    

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
    # Clean up index to be just integers
    # Keep only first on hour and pad out to one hour if necessary
    df.index = range(len(df.index))
    df = df.reindex(xrange(3600))

    # Velocity is distance between consecutive points (/s)
    df['velocity'] = distance(df.position_x.diff(), df.position_y.diff(), df.position_z.diff())
    
    # Get mean, std for now
    mean = pd.concat(df[col].groupby(df.index/60).mean() for col in ['position_x', 'position_y', 'position_z','velocity'])
    std  = pd.concat(df[col].groupby(df.index/60).std()  for col in ['position_x', 'position_y', 'position_z','velocity'])
    features = pd.concat((mean, std))
    features[features.isnull()] = 0
    return features


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

    # Grab gps data for each task, processing if specified
    gps = pd.read_pickle('data/LocationData.pkl')
    with open('data/NameToNode.pkl','r') as infile:
        nodes = pickle.load(infile)
    nodes = dict((int(key), val) for key,val in nodes.iteritems())

    df['name'] = df.first_name + ' ' + df.last_name
    gps['name'] = gps.node_id.map(nodes)

    # Build an extension to the df by creating feature vectors from (transformed) x,y,z data
    for index, name, start, end, task in itertools.izip(df.index, df.name, df.start_time, df.end_time, df.task):
        sys.stdout.write('Processing task number {0} out of {1}\r'.format(index,len(df.index)))
        sys.stdout.flush()
        sub_gps = gps[(gps.name == name) & (gps.position_update_timestamp > start) & (gps.position_update_timestamp < end)].copy()
        if not len(sub_gps):
            continue
        sub_gps['start_time'] = start
        sub_gps['end_time'] = end
        sub_gps['task_id'] = index
        sub_gps['task_label'] = task
        sub_gps.to_pickle('data/gps_{:06d}.pkl'.format(index))
    print
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
    fnames = glob.glob('data/gps_*.pkl')
    if n is not None:
        fnames = fnames[:n]
    index = pd.Series(int(fname.split("_")[1].split(".")[0]) for fname in fnames)
    
    df = None
    for ix, fname in itertools.izip(index,fnames):
        # Grab a dataframe which represents a single task
        gps_task = pd.read_pickle(fname)
        features = gps_transform.funcs[gps_reduce](gps_task)
        if df is None:
            df = pd.DataFrame(columns=features.index, index=index)
        df.loc[ix] = features
    return df
    
    
def main():
    '''
    Make the gps pickles which are used by other methods.
    '''
    create_gps_pickles()


if __name__ == "__main__":
    main()
