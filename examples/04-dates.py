#!/usr/bin/env python

import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import think
from think import fast, slow
from think import Date, After, Before

dates = []
from_date = '1945-01-01'
to_date = '2025-12-31'
for dt in pd.date_range(from_date, to_date):
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
plt.title(f"Pairwise similarities of dates from {from_date} to {to_date}", fontsize=35)
plt.savefig(f"pairwise-similarities-of-dates.png")
plt.close('all')

