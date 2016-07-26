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
    This function determines the fraction of time in each quantile for each task
    as well as the total number of quantile edge crossings
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
    #Determine the total number of quantile crossings
    df2 = df[df['quantile_diff'] != 0]
    quant_dict['total_crossings'] = len(df2)
    #Return the updated dictionary with fractions and total number of crossings
    return quant_dict
