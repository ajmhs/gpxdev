import numpy as np
import gpxpy.gpx
import pandas as pd
import haversine as hs
import matplotlib.pyplot as plt
import argparse
import os
import pdb

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    distance = hs.haversine(
        point1=(lat1, lon1),
        point2=(lat2, lon2),
        unit=hs.Unit.METERS
    )
    return np.round(distance, 2)


def gpx_to_df(source):
    route_info = []
    with open(source, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)  
        #for track in gpx.tracks:
            #for segment in track.segments:
                #for point in segment.points:
        for route in gpx.routes:
            for point in route.points:
                route_info.append({'latitude': point.latitude, 'longitude': point.longitude,'elevation': point.elevation})
                
                
    # Convert to a pandas dataframe and return
    return pd.DataFrame(route_info)

def calc_distances(df):
    # Calculate and add the distances between points
    distances = [np.nan]
    for i in range(len(df)):
        if i == 0:
            continue
        
        distances.append(haversine_distance(
            lat1=df.iloc[i - 1]['latitude'],
            lon1=df.iloc[i - 1]['longitude'],
            lat2=df.iloc[i]['latitude'],
            lon2=df.iloc[i]['longitude']))
            
    df['distance'] = distances
    return df

def calc_gradients(df):
    # Calculate and add the gradients between points
    gradients = [np.nan]
    df = df.reset_index() # make sure indexes pair with number of rows
    for i, row in df.iterrows(): 
        if i == 0:
            continue
       
        grade = (row['elevation_diff'] / row['distance']) * 100
        if np.abs(grade) > 30: # Filter out silly gradients
            gradients.append(np.nan)
            print('Ignoring calculated gradient of {}'.format(grade))
        else:
            gradients.append(np.round(grade, 1))

    df['gradient'] = gradients

    # Apply interpolation to the entire gradients dataset, fixing skipped gradients
    df['gradient'] = df['gradient'].interpolate().fillna(0)
    return df


def plot_elevation(out_dir, df):   
    # Create the elevation plot
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.plot(df['cum_distance'], df['cum_elevation'], color='#101010', lw=3)
    plt.title('Route elevation profile', size=20)
    plt.xlabel('Distance in meters', size=14)
    plt.ylabel('Elevation in meters', size=14)
    plt.savefig(os.path.join(out_dir, 'elevation_profile.png'))


def plot_gradient(out_dir, df):
    # Create the gradient plot
    plt.clf()
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.title('Route Terrain gradient', size=20)
    plt.xlabel('Data point', size=14)
    plt.ylabel('Gradient (%)', size=14)
    plt.plot(np.arange(len(df)), df['gradient'], lw=2, color='#101010');
    plt.savefig(os.path.join(out_dir, 'gradient_profile.png'))


def filter_gradients(df, gradient_len, gradient_pc):
    # Isolate each slope and if within parameters, add to filtered gain/loss statistics
    gain = 0
    loss = 0
    last_idx = 0
    is_climbing = False

    for i, row in df.iterrows(): 
        if i == 0:
            continue
        elif i == 1:
            is_climbing = row['elevation_diff'] > 0
        else:
            if is_climbing and row['elevation_diff'] >= 0 and i+1 < df.shape[0]:
                continue
            elif not is_climbing and row['elevation_diff'] < 0 and i+1 < df.shape[0]:
                continue
            else:
                rows = range(last_idx, i)

                slope_distance = df['distance'].iloc[rows].sum()
                slope_gradient = df['gradient'].iloc[rows].mean()
                slope_notes = '{}hill slope of {:.2f}m length with a mean gradient of {:.2f}'.format(("up" if is_climbing else "down"), slope_distance, slope_gradient)

                if slope_distance >= gradient_len and np.abs(slope_gradient) >= gradient_pc:
                    print('Including {}'.format(slope_notes))

                    slope_elevation = df['elevation_diff'].iloc[rows].sum()
                    if is_climbing:
                        gain += slope_elevation
                    else:
                        loss += slope_elevation
                else:
                    print('Excluding {}'.format(slope_notes))

                is_climbing = row['elevation_diff'] > 0
                last_idx = i
    return gain, loss


def process_file(source, gradient_len, gradient_pc):
    #pdb.set_trace()

    route_df = gpx_to_df(source)

    # Remove duplicate points
    route_df = route_df.drop_duplicates(subset=['latitude', 'longitude', 'elevation'], keep='last')
    
    # Calculate and add the elevation differences between point
    route_df['elevation_diff'] = route_df['elevation'].diff()

    # Calculate and add the distances between points
    route_df = calc_distances(route_df)

    # Calculate and add the gradients between points
    route_df = calc_gradients(route_df)

    # Calculate and add cumulative elevation
    route_df['cum_elevation'] = route_df['elevation_diff'].cumsum()
    
    # Calculate and add cumulative distance
    route_df['cum_distance'] = route_df['distance'].cumsum()

    # Calculate some stats, should be printed out to stdout
    elevation_gain = route_df[route_df['elevation_diff'] >= 0]['elevation_diff'].sum()
    elevation_loss = route_df[route_df['elevation_diff'] < 0]['elevation_diff'].sum()
    distance = route_df['distance'].sum()

    # Clean up any NaN values
    route_df = route_df.fillna(0)

    # Create an output directory with the same name as the gpx file to hold output
    out_dir, ext = os.path.splitext(source)
    os.makedirs(out_dir, exist_ok=True)

    # Export to file
    route_df.to_csv(os.path.join(out_dir, 'data.csv'), index=False)

    # Create the elevation plot
    plot_elevation(out_dir, route_df)
    
    # Create the gradient plot
    plt.clf()
    plot_gradient(out_dir, route_df)

    filter_gain, filter_loss = filter_gradients(route_df, gradient_len, gradient_pc)

    print('Elevation Gain: {:.2f}({:.2f})m, Elevation Loss: {:.2f}({:.2f})m, Distance: {:.2f}m'.format(filter_gain, elevation_gain, np.abs(filter_loss), np.abs(elevation_loss), distance))


def main():
    parser = argparse.ArgumentParser(description='GPX Track Analysis tool.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    req_named = parser.add_argument_group('required named arguments')
    req_named.add_argument('-s', '--source', required=True, help='GPX file to examine')
    parser.add_argument('-d', '--distance', required=False, default=20, help='Minimum distance (m) to consider, default is 20')
    parser.add_argument('-g', '--gradient', required=False, default=3.5, help='Minimum gradient (%) to consider, default is 3.5')

    # Read the arguments
    pargs = parser.parse_args()
    source = pargs.source
    min_dist = pargs.distance
    min_grad = pargs.gradient

    try:
        process_file(source, min_dist, min_grad)
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    main()
