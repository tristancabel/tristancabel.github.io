import os
import io
import sys
import codecs
import string
import operator

import pandas  as pd
import numpy   as np
import seaborn as sns
import matplotlib.pyplot as plt



# Plotting Options
sns.set_style("whitegrid")
sns.despine()


def plot_bar(df, column, title, fname, hue):
    col_freq       = df.groupby(column).size()
    col_freq.sort_values(ascending=False, inplace=True)
    color = sns.color_palette(hue, len(col_freq))
    bar   = col_freq.plot(kind='barh', title=title, fontsize=8, figsize=(12,8), stacked=False, width=1, color=color)
    bar.figure.savefig(fname)

def plot_by_day(df, column, title, fname, hue):
    plt.figure() #get a new figure
    col_freq  = df.groupby(column).size().reset_index()
    col_freq.columns.values[1] = "count"
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    mapping = {day: i for i, day in enumerate(days)}
    key = col_freq['DayOfWeek'].map(mapping)    
    col_freq = col_freq.iloc[key.argsort()]

    color = sns.color_palette(hue, len(col_freq))
    bar   = col_freq.plot(kind='bar', x='DayOfWeek', y='count', title=title, fontsize=8, figsize=(12,8), stacked=False, width=1, color=color)
    bar.figure.savefig(fname)
    
def periodOfDay(row, applyday=1):
    day_offset = {'Monday':0 , 'Tuesday':24, 'Wednesday':48, 'Thursday':72, 'Friday':96, 'Saturday':120, 'Sunday':144 }
    t = row['Time'].split(':')
    res = day_offset[row['DayOfWeek']]*applyday + int(t[0])
    return res

def plot_by_hour(df, title, fname, hue):
    plt.close('all')
    f, (ax1, ax2) = plt.subplots(2, 1)

    #group by every hour during a week.
    col_freq = df.groupby('HourOfWeek').size()
    line = col_freq.plot(ax=ax1, title=title + ' distribution during a week', fontsize=8, stacked=False, marker='o', xticks=np.arange(0, 168, 12))

    col_freq2 = df.groupby('Hour').size()
    line2 = col_freq2.plot(ax=ax2, title=title + ' distribution during a day', fontsize=8, stacked=False, marker='o', xticks=np.arange(0, 24, 6))

    f.subplots_adjust(hspace=.5)
    f.savefig(fname)
    

def plot_location(df, fname):
    # map from https://www.kaggle.com/benhamner/sf-crime/saving-the-python-maps-file
    map_sf = np.loadtxt("sf_map_copyright_openstreetmap_contributors.txt")
    clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]
    lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
    asp = map_sf.shape[0] * 1.0 / map_sf.shape[1]
    plt.figure(figsize=(16,16*asp))

    map_loc = sns.kdeplot(df['X'], df['Y'], shade='True', clip=clipsize, aspect=1/asp)
    # see issue https://github.com/mwaskom/seaborn/issues/393
    alpha_list = [0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.8, 0.8]
    for i in range(len(map_loc.collections)):
         map_loc.collections[i].set_alpha(alpha_list[i])

    map_loc.imshow(map_sf,cmap='gray',extent=lon_lat_box,aspect=1/asp)
    map_loc.set_axis_off()
    plt.savefig(fname)


def input_transformer(filepath):
    """
    Read file input and transform it into a pandas DataFrame
    """
    fp = open(filepath,'rb')
    raw = fp.read().decode('utf-8')
    return pd.read_csv(io.StringIO(raw), parse_dates=True, index_col=0, na_values='NONE')

def main(filepath):
    """
    Script Entry Point
    """
    print 'reading data'
    df = input_transformer(filepath)
    
    #filter unused data and add 2 new columns
    print 'filter data'
    df.drop(['Descript', 'PdDistrict', 'Address'],inplace=True,axis=1)
    #so Monday [00:00, 00:59] = 0; Monday [02:00,02:59] = 2;
    df['HourOfWeek'] = df.apply (lambda row: periodOfDay(row),axis=1)
    df['Hour'] = df.apply (lambda row: periodOfDay(row, 0),axis=1)
    
    #plot by crime category
    plot_bar(df, 'Category', 'Top Crime Categories', 'category.png', 'spectral')

    print 'plot by day'
    plot_by_day(df, 'DayOfWeek', 'crimes per day', 'crimes_per_day.png', 'muted')
    plot_by_hour(df, 'crime', 'crimes_distribution_per_hour.png', 'pastel')

    print "map all crimes"
    plot_location(df, 'map_all_crimes.png')

    ## focus on larceny
    df_larceny = df[df['Category']=='LARCENY/THEFT']

    #plot by location
    print 'plot Larceny by location'
    plot_location(df_larceny, 'map_larceny.png')

    #plot by time and day
    print 'plot Larceny by day'
    plot_by_day(df_larceny, 'DayOfWeek', 'Larceny crime per day', 'larceny_per_day.png', 'muted')
    plot_by_hour(df_larceny, 'Larceny', 'larceny_per_hour.png', 'pastel')

    #plt.show()


if __name__ == '__main__':
    sys.exit(main('sanfrancisco_incidents_summer_2014.csv'))
