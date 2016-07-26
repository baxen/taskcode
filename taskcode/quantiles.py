import pickle
import pandas as pd
import numpy as np

def quantile_definer(dimension, df):
    '''
    Returns quantile grid boundaries based on LocationData for x and y
    As an example, dimension 3 would return two lists:
        a) One list of length 2 for x-coordinate cut-offs
        b) One list of length 2 for y-coordinate cut-offs
    This will establish 9 quantiles overall
    '''
    t = len(df)
    #Start by sorting all x- and y-coodrinates in ascending order
    df_x = df.position_x.sort_values(ascending = True).tolist()
    df_y = df.position_y.sort_values(ascending = True).tolist()
    index_boundaries = []
    #Define the index boundaries for the particular quantile dimension
    for i in range(dimension-1):
        index_boundaries.append(t*(i+1)/dimension)
    #Now define the x,y boundaries using df_x/y and the index boundaries
    x_boundaries = []
    y_boundaries = []
    for index in index_boundaries:
        x_boundaries.append(df_x[index])
        y_boundaries.append(df_y[index])
    #return the boundaries for each coordinate as a list of lists
    return [x_boundaries, y_boundaries]

def quantile_id(xpos, ypos):
    '''
    IDs the quantile that a positional vector belongs to given
    the boundaries identified using quantile definer
    '''
    X = boundaries[0]
    Y = boundaries[1]
    #A list of quantile x,y coords that will be converted to an integer at the end
    quantile_coord = []
    i = 0; j = 0
    while i < len(X):
        if xpos < X[i]:
            quantile_coord.append(i)
            break
        else:
            i += 1
    if len(quantile_coord) != 1:
        quantile_coord.append(i)
    while j < len(Y):
        if ypos < Y[j]:
            quantile_coord.append(j)
            break
        else:
            j += 1
    if len(quantile_coord) != 2:
        quantile_coord.append(j)
    dimension = len(X) + 1
    #Convert the [x,y] quantile bounds to an integer value in range(0, dimension^2)
    return int(dimension*quantile_coord[0] + quantile_coord[1])

def quantiler(df, quant_bounds):
    '''
    Defines useful parameters for pickles for feature vector derivation later on
    quant_bounds are the bounds established by quantile_definer above
    '''
    df['quantile'] = df.apply(lambda row: quantile_id(row['position_x'], row['position_y']), axis = 1)
    #For each data point, determine the time spent in the quantile defined above
    df['time_delta'] = df.position_update_timestamp.diff()/pd.Timedelta('1s')
    df['time_delta'] = df['time_delta'].shift(-1)
    df['time_delta'] = df['time_delta'].fillna(0)
    #Determine if there is a quantile switch and record it as new quantile
    df['quantile_diff'] = df['quantile'].diff()
    df['quantile_diff'] = df['quantile_diff'].fillna(0)
    df['new_quantile'] = df['quantile'] - df['quantile_diff'].astype(int)
    #Return df with the new quantile id features
    return df

def quantile_dwell(df, dimension):
    '''
    This function determines the dwell time in each quantile for each task
    Based on dimension
    Also determines the fraction of time spent in each quantile
    '''
    df = df.sort_values(by = 'quantile', ascending = True)
    quant_rep = df['quantile'].unique().tolist()
    quant_times = df['time_delta'].groupby(df['quantile']).sum().tolist()
    total_time = sum(quant_times)
    temp_dict = {}
    keys = list(range(pow(dimension, 2)))
    for i,item in enumerate(quant_rep):
        temp_dict[item] = quant_times[i]
    quant_dict = {}
    #Calculate the fraction of time in each quantile
    for key in keys:
        if key in temp_dict:
            quant_dict['Quant_' + str(key) + '_Frac'] = temp_dict[key]/total_time
        else:
            quant_dict['Quant_' + str(key) + '_Frac'] = 0
    #Return the updated dictionary
    return quant_dict

def quant_switcher(df, dimension):
    '''
    This function determines the counts for quantile switching referenced to boundary
    '''
    keys = []
    '''
    We will assume that any two quantiles can be directly connected between consecutive timepoints.
    This is not true if the GPS data are very dense (e.g., in some cases you would have to cross
    multiple quantile edges to move from quantile M to quantile N), but in cases where there
    are significant gaps between consecutive timepoints, you could envision "teleporting"
    between non-adjacent quantiles without any information about the true continuous path that was taken.
    We will also for now ignore directionality of crossing (e.g. 1->2 and 2->1 are equivalent), since
    the number of forward crossings should be linearly dependent on the number of reverse crossings.
    This may not be the case if you have a lot of teleportation, but I imagine that
    those types of events will be rare relative to continuous crossings.
    '''
    for i in range(pow(dimension, 2)):
        for j in range(i + 1, pow(dimension, 2)):
            keys.append(str(i) + '_' + str(j) + '_cross')
    #Initialize the switcher dictionary, first with zero counts for each crossing
    switcher_dict = {}
    for key in keys:
        switcher_dict[key] = 0
    #Remove entries that do not have quantile crossings
    df2 = df[df['quantile_diff'] != 0]
    df2.index = range(len(df2))
    #Define a helper function to convert quantile and new_quantile into a dictionary key
    def key_maker(x, y):
        if int(x) < int(y):
            return str(x) + '_' + str(y) + '_cross'
        else:
            return str(y) + '_' + str(x) + '_cross'
    if len(df2) == 0:
        switcher_dict['total_crossings'] = 0
        return switcher_dict
    else:
        df2['cross_keys'] = df2.apply(lambda row: key_maker(row['quantile'], row['new_quantile']), axis = 1)
        cross_keys = df2['cross_keys'].tolist()
        for key in cross_keys:
            switcher_dict[key] += 1
        #also compute the total number of crossings
        switcher_dict['total_crossings'] = len(df2)
        return switcher_dict
