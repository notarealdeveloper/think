#!/usr/bin/env python3

import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import fast
import slow
import think
from think import Date, After, Before

dates = []
for dt in pd.date_range('1998-01-01', '2003-01-01'):
    date = dt.date().isoformat()
    dates.append(Date(date))

Date.learn()

thoughts = {d.object: d.think() for d in Date.instances().values()}
sims = {}
for a,b in itertools.product(thoughts, thoughts):
    key = (a,b)
    val = fast.cos(thoughts[a], thoughts[b]).item()
    sims[key] = val
    print(key)
sims = pd.Series(sims.values(), index=sims.keys()).unstack()

sns.set(font_scale=0.30)
c = sns.heatmap(pd.DataFrame(sims), xticklabels=10, yticklabels=10)
c.figure.set_figheight(20)
c.figure.set_figwidth(30)
plt.title(f"Pairwise similarities of dates from 1998-01-01 to 2003-01-01", fontsize=35)
plt.savefig(f"pairwise-similarities-of-dates.png")
plt.close('all')

