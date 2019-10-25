#coding:utf-8
import os
import sys
import json
from collections import defaultdict
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import math
import numpy as np
import random
import logging
import networkx as nx
from itertools import combinations
import pylab
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import spline
from multiprocessing.dummy import Pool as ThreadPool
from networkx.algorithms import isomorphism
from matplotlib import cm as CM
from collections import Counter
from scipy.signal import wiener
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from networkx.algorithms.core import core_number
from networkx.algorithms.core import k_core
import psycopg2
from cycler import cycler
# from gini import gini

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

mpl.rcParams['agg.path.chunksize'] = 10000

color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
mpl.rcParams['axes.prop_cycle'] = cycler('color', color_sequence)

# color = plt.cm.viridis(np.linspace(0.01,0.99,6)) # This returns RGBA; convert:
# hexcolor = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),
#                tuple(color[:,0:-1]))

# mpl.rcParams['axes.prop_cycle'] = cycler('color', hexcolor)


params = {'legend.fontsize': 8,
         'axes.labelsize': 8,
         'axes.titlesize':10,
         'xtick.labelsize':8,
         'ytick.labelsize':8}
pylab.rcParams.update(params)


# from paths import *


def circle(ax,x,y,radius=0.15):

    circle = Circle((x, y), radius, clip_on=False, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=(0, 0, 0, .0125),
                    path_effects=[withStroke(linewidth=5, foreground='w')])
    ax.add_artist(circle)


def autolabel(rects,ax,total_count=None,step=1,):
    """
    Attach a text label above each bar displaying its height
    """
    for index in np.arange(len(rects),step=step):
        rect = rects[index]
        height = rect.get_height()
        # print height
        if not total_count is None:
            ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,
                    '{:}\n({:.6f})'.format(int(height),height/float(total_count)),
                    ha='center', va='bottom')
        else:
            ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,
                    '{:}'.format(int(height)),
                    ha='center', va='bottom')


class dbop:

    def __init__(self,insert_index=0):

        self._insert_index=insert_index
        self._insert_values=[]
        logging.debug("connect database with normal cursor.")
        self._db = psycopg2.connect(database='core_data',user="buyi",password = "ruth_hardtop_isthmus_bubbly")
        self._cursor = self._db.cursor()



    def query_database(self,sql):
        self._cursor.close()
        self._cursor = self._db.cursor()
        self._cursor.execute(sql)
        logging.debug("query database with sql {:}".format(sql))
        return self._cursor

    def insert_database(self,sql,values):
        self._cursor.close()
        self._cursor = self._db.cursor()
        self._cursor.executemany(sql,values)
        logging.debug("insert data to database with sql {:}".format(sql))
        self._db.commit()


    def batch_insert(self,sql,row,step,is_auto=True,end=False):
        if end:
            if len(self._insert_values)!=0:
                logging.info("insert {:}th data into database,final insert.".format(self._insert_index))
                self.insert_database(sql,self._insert_values)
        else:
            self._insert_index+=1
            if is_auto:
                row[0] = self._insert_index
            self._insert_values.append(tuple(row))
            if self._insert_index%step==0:
                logging.info("insert {:}th data into database".format(self._insert_index))
                self.insert_database(sql,self._insert_values)
                self._insert_values=[]

    def get_insert_count(self):
        return self._insert_index

    def execute_del_update(self,sql):
        self._cursor.execute(sql)
        self._db.commit()
        logging.debug("execute delete or update sql {:}.".format(sql))

    def execute_sql(self,sql):
        self._cursor.execute(sql)
        self._db.commit()
        logging.debug("execute sql {:}.".format(sql))

    def close_db(self):
        self._db.close()




def hist_2_bar(data,bins=50):
    n,bins,patches = plt.hist(data,bins=bins)
    return [x for x in bins[:-1]],[x for x in n]




