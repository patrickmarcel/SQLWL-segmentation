import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from sklearn.metrics.cluster import adjusted_rand_score

from clustering import DesC


#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/smartBI-legros-v1.csv'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-legros-v1.csv'
path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-float.csv' # dopan-smartbi ground truth=124 cuts

datatab=np.genfromtxt(path, dtype=float, delimiter=';',  encoding='utf8', skip_header=1)
#data=datatab[:,3:13] # intrinsic
#data=datatab[:,3:29] # all
#data=datatab[:,3:20] # all z-scores

data=datatab[:,24:29] # indexes concatenate
#data=datatab[:,26:31] # smartBI / dopan indexes



desc = DesC().fit(data)

print('dopan-smartbi (expected 124 cuts)')
print(desc.K_, ' clusters')

desc.eval_graph_
dopanSmartBiLabels=desc.labels_



#metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-legros-v1.csv'
#metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/smartBI-legros-v1.csv'
metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-float.csv'


# in that file, we have sessionID first and then queryId
# while candidates are queryID,sessionID

#fieldNames="QuerySId,SessionSId,UserSId,NoP,NoF,NoA,NoT,NoAtt,NCP,NCF,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,Lenght,RED,Edit-index,Jackard-index,Cosine-index,Common-fragments-index,Common-tables-index,Vote,ExplorationSId,GroundTruth,ChangeSession"


fieldNames="QuerySId,SessionSId,UserSId,NoP,NoF,NoA,NoT,NoAtt,NCP,NCF,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,RED,EditIndex,JackardIndex,CosineIndex,CommonFragmentsIndex,CommonTablesIndex,Vote,ExplorationSId,GroundTruth,ChangeSession,a"

metrics=np.genfromtxt(metricspath, dtype=None, delimiter=';', names=fieldNames, encoding='utf8', skip_header=1)



prec=0
i=0
tp=0
tn=0
fp=0
fn=0
for l in dopanSmartBiLabels:
    if l!=prec:
        clust=1
    else:
        clust=0
    prec=l

    cut=metrics[i][31] #ground truth for query i # concatenate
#    cut=metrics[i][33] #ground truth for query i # smartBI or Dopan

    if cut==1 and clust==1:
        tp=tp+1
    if cut==0 and clust==0:
        tn=tn+1
    if cut==0 and clust==1:
        fp=fp+1
    if cut==1 and clust==0:
        fn=fn+1
    i=i+1
wprecision=tp/(tp+fp)
wrecall=tp/(tp+fn)
waccuracy=(tp+tn)/(tp+fn+fp+tn)
wfmeasure=(2*wprecision*wrecall)/(wprecision+wrecall)
print('F-measure=',wfmeasure)
print('precision=',wprecision)
print('recall=',wrecall)
print('accuracy=',waccuracy)


####### ARI

truthClusters=[]
clusterNb=0
prec=0
for i in metrics:
    if i[31]==1: # ground truth concatenate
#    if i[33]==1:
        clusterNb=clusterNb+1
    truthClusters.append(clusterNb)

ari=adjusted_rand_score(truthClusters,dopanSmartBiLabels)
print('ARI=',ari)


######### SQLShare


#
# path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v3.csv'
#
#
#
# print('sqlshare')
#
# datatab=np.genfromtxt(path, dtype=float, delimiter=';',  encoding='utf8', skip_header=1)
# data=datatab[:,2:12] # intrinsic
# data=datatab[:,23:28] # indexes
#
# desc = DesC().fit(data)
#
# print(desc.K_, ' clusters')
#
# #desc.eval_graph_
# sqlsharelabels=desc.labels_
#
#
# sqlshareCustPath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlShareCuts-DesC.csv'
#
#
# with open(sqlshareCustPath, mode='a') as cut_file:
#     cut_writer = csv.writer(cut_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     prec=0
#     for l in sqlsharelabels:
#         if l!=prec:
#             cut_writer.writerow([1])
#         else:
#             cut_writer.writerow([0])
#         prec=l