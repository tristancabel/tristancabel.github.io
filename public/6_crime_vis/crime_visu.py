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
    
def periodOfDay(row):
    day_offset = {'Monday':0 , 'Tuesday':24, 'Wednesday':48, 'Thursday':72, 'Friday':96, 'Saturday':120, 'Sunday':144 }
    t = row['Time'].split(':')
    res = day_offset[row['DayOfWeek']] + int(t[0])
    return res

def plot_by_hour(df, title, fname, hue):
    #group by every hour during a week.
    #so Monday [00:00, 00:59] = 0; Monday [02:00,02:59] = 2;
    df['HourOfWeek'] = df.apply (lambda row: periodOfDay(row),axis=1)

    col_freq = df.groupby('HourOfWeek').size()
    color = sns.color_palette(hue, len(col_freq))
    plt.figure()
    line2d   = col_freq.plot(title=title, fontsize=8, figsize=(12,8), stacked=False, marker='o')

    # major ticks every 12, minor ticks every 6
    line2d.set_xticks(np.arange(0, 168, 12))
    line2d.set_xticks( np.arange(0, 168, 6), minor=True)
    line2d.figure.savefig(fname)
    

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


def main(filepath):
    print 'reading data'
    fp = open(filepath,'rb').read().decode('utf-8')
    df = pd.read_csv(io.StringIO(fp), parse_dates=True, index_col=0, na_values='NONE')
    
    #filter unused data
    print 'filter data'
    df.drop(['Descript', 'PdDistrict', 'Address'],inplace=True,axis=1)

    #plot by crime category
    plot_bar(df, 'Category', 'Top Crime Categories', 'category.png', 'spectral')

    print 'plot by day'
    plot_by_day(df, 'DayOfWeek', 'crimes per day', 'crimes_per_day.png', 'muted')
    plot_by_hour(df, 'crime distribution during a week', 'crimes_distribution_per_hour.png', 'pastel')
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
    plot_by_hour(df_larceny, 'Larceny crime during a week', 'larceny_per_hour.png', 'pastel')


if __name__ == '__main__':
    sys.exit(main('sanfrancisco_incidents_summer_2014.csv'))
