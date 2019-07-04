'''
Created on 12 oct. 2017

@author: labroche
'''
import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Mahfoud/dopan/logsDopan-labels/agreedMetricsWithLabels.csv'
#path='/Users/patrick/Documents/RECHERCHE/STUDENTS/Mahfoud/featuresSmartBI/output_2018-02-21-ToutesLignesCorrectes.csv'
#dataset = np.genfromtxt(csvfile, dtype=float, delimiter=';', skip_header=1)

#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-float.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-legros-v1.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/smartbi-legros-v1.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v3.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v4.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v5.csv'

path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-v2.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-legros-v2.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/smartbi-legros-v2.csv'



#targettab=np.genfromtxt(metricspathfloat, dtype=float, delimiter=';',  encoding='utf8', skip_header=1)


#df = pd.read_csv(path, sep=';', usecols=[3,4,5,6,7,8,9,10,11]) # dopan-smartbi metrics + gt
#df = pd.read_csv(path, sep=';', usecols=[24,25,26,27,28,31]) # dopan-smartbi indexes +gt
df = pd.read_csv(path, sep=';', usecols=[3,4,5,6,7,8,9,10,11,12,24,25,26,27,28,31,32]) # dopan-smartbi all +gt
#df = pd.read_csv(path, sep=';', usecols=[13,14,15,16,17,18,19,20,21,22,23,31]) # dopan-smartbi all +gt
#df = pd.read_csv(path, sep=';', usecols=[3,4,5,6,7,8,9,10,11]) # dopan
#df = pd.read_csv(path, sep=';', usecols=[3,4,5,6,8,9,10,11]) # dopan without Att
#df = pd.read_csv(path, sep=';', usecols=[3,4,5,6,7,8,9,10,11]) # smartbi
#df = pd.read_csv(path, sep=';', usecols=[3,4,5,6,8,9,10,11]) # smartbi without Att
#df = pd.read_csv(path, sep=';', usecols=[2,3,4,5,6,7,8,9,10,11]) # sqlshare
#df = pd.read_csv(path, sep=',', usecols=[3,4,5,6,7,8,9,10,11]) # sqlshare v4
#df = pd.read_csv(path, sep=',', usecols=[3,4,5,6,8,9,10,11]) # sqlshare v4 without Att
#df = pd.read_csv(path, sep=';', usecols=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,31])

corr = df.corr(method='pearson', min_periods=100)

corr.to_csv(path +'correlation.csv', sep=',', header=True)
print('Output file created')

plt.matshow(corr)
#ax = plt.gca()
#ax.set_xticklabels(['']+corr.columns.values)
#ax.set_yticklabels(['']+corr.columns.values)

sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            #annot=True,
            cmap="YlGnBu")

plt.show()