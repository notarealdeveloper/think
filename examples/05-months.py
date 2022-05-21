#!/usr/bin/env python3

import itertools
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import matplotlib.pyplot as plt

import fast
import slow
import think
from think import Date, After, Before

dates = {}
for dt in pd.date_range('1880-01-01', '2020-01-01', freq='D'):
    date = dt.date()
    date = f"{date.year}-{date.month:02d}"
    if date in dates:
        continue
    dates[date] = Date(date)
    print(date)

thoughts = {d.object: d.think() for d in Date.instances().values()}
sims = {}
def get_similarity(a,b):
    key = (a,b)
    val = fast.cos(thoughts[a], thoughts[b]).item()
    print(key)
    return (key, val)

with mp.Pool(mp.cpu_count()) as pool:
    sims = pool.starmap(get_similarity, itertools.product(thoughts, thoughts))
sims = dict(sims)
sims = pd.Series(sims.values(), index=sims.keys()).unstack()

sns.set(font_scale=0.30)
c = sns.heatmap(pd.DataFrame(sims), xticklabels=12, yticklabels=12)
c.figure.set_figheight(20)
c.figure.set_figwidth(30)
plt.title(f"Pairwise similarities of dates from 1880-01 to 2020-01", fontsize=35)
plt.savefig(f"pairwise-similarities-of-months.png")
plt.close('all')

