
##### IMPORTS

import os

from snorkel import SnorkelSession
from snorkel.parser import TSVDocPreprocessor

from snorkel.parser.spacy_parser import Spacy
from snorkel.parser.rule_parser import RuleBasedParser,RegexTokenizer
from snorkel.parser import CorpusParser

from snorkel.models import Document, Sentence
from snorkel.models import candidate_subclass, TemporarySpan

from snorkel.candidates import Ngrams, CandidateExtractor, CandidateSpace
from snorkel.matchers import RegexMatch,PersonMatcher, RegexMatchEach,RegexMatchSpan

from snorkel.contrib.brat import BratAnnotator

import re
from snorkel.lf_helpers import (
    get_left_tokens, get_right_tokens, get_between_tokens,
    get_text_between, get_tagged_text,
)

from snorkel.annotations import LabelAnnotator

import numpy as np
from numpy import dot
from numpy.linalg import norm

from snorkel.learning import GenerativeModel
import matplotlib.pyplot as plt

import csv


##### CANDIDATE DEFINITION

class queryCandidate(CandidateSpace):
    """
    Defines the space of candidates as pairs of consecutive queries.
    """
    def __init__(self):
        CandidateSpace.__init__(self)

    def apply(self, context):
        seen = set()
        text=context.text # gets sentence as string
        #print(text)
        i=0
        while i < len(text)-1:
            j=i+1
            while text[j]!='|':
                j=j+1
            # must continue until next one
            k=j+1
            j=j+1
            if j<len(text)-1:
                #print(j,len(text))
                while text[j]!='|':
                    j=j+1
                start=i
                end=j
                #i=j+1
                i=k

                #print(start,end)
                #print(text[start:end])
                ts    = TemporarySpan(char_start=start, char_end=end, sentence=context)
                #print(ts)

                if ts not in seen:
                    seen.add(ts)
                    yield ts
            else:
                i=j



##### MISC FUNCTIONS

def recall(name,idQuery1,idQuery2,idSession):
    num=findMetric(name,idQuery2,idSession)
    metric2=name.replace('C','o')
    denom=findMetric(metric2,idQuery1,idSession)
    if denom!=0:
        return num/denom
    else:
        return 0

def precision(name,idQuery,idSession):
    num=findMetric(name,idQuery,idSession)
    metric2=name.replace('C','o')
    denom=findMetric(metric2,idQuery,idSession)
    if denom!=0:
        return num/denom
    else:
        return 0

def lessInCommon(name,idQuery1,idQuery2,idSession):
    numQ1=findMetric(name,idQuery1,idSession)
    numQ2=findMetric(name,idQuery2,idSession)
    if numQ2<=numQ1:
        return True
    else:
        return False

def findMetric(name,idQuery,idSession):
#    print('idSession is: ', idSession)
#    print('idQuery is: ', idQuery)
    if idQuery==0:
        return 0
    else:
        tmp=metrics[metrics['SessionSid']==idSession]
        return tmp[tmp['QuerySId']==idQuery][name][0]



def getCandidatesIDs(c):
    pair=c.queryPair.get_span()
    q1=pair.split(sep='|')[0]
    q2=pair.split(sep='|')[1]
    idQ1=q1.split(sep=',')[0]
    idSession=q1.split(',')[1]
    idQ2=q2.split(sep=',')[0]
    return (int(idQ1),int(idQ2),int(idSession))



def get_power_set(s):
  power_set=[[]]
  for elem in s:
    # iterate over the sub sets so far
    for sub_set in power_set:
      # add a new subset consisting of the subset at hand added elem
      power_set=power_set+[list(sub_set)+[elem]]
  return power_set



##### LABELLING FUNCTIONS

# LF functions:
# 1: queries should stay together
# -1: queries should not stay together
# 0: don't know



# indexes as defined in DOLAP 2019 paper:
#


def LF_edit_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    edit_index = findMetric('Edit_index', idQ2, idSession)
    if edit_index > 0:
        return 1
    else:
        return -1


def LF_jackard_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    jackard_index = findMetric('Jackard_index', idQ2, idSession)
    if jackard_index > 0:
        return 1
    else:
        return -1



def LF_Common_Tables_Index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    common_tables_index = findMetric('Common_tables_index', idQ2, idSession)
    if common_tables_index > 0:
        return 1
    else:
        return -1



def LF_recall_projections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCP',idQ1,idQ2,idSession)
    if rec!=0:
        return 1
    else:
        score=findMetric('NoP',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1






##### FILE TO IMPORT


# one big document with one sentence per exploration
# if -small -> just a little sample for testing purpose
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshareQueries-IDs-OLD-another.txt'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshareQueries-IDs-small.txt'
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshareQueries-IDs-tiny.txt'
path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshareQueries-IDs.txt' # huge pb with this file, for some reason parser finds two sentences when there shouls be only one




#metricspath='/Users/patrick/Documents/RECHERCHE/STUDENTS/Willeme/metric_sql_share.csv'
metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v5.csv'

#fieldNames="ID,ID_EXPLORATION,Number_of_projections,Number_of_tables,Number_of_aggregation_functions,Number_of_selections,Common_projections_number,Common_selections_number,Common_aggregation_functions_number,Common_tables_number,QUERY_CHARACTER_NUMBER,Number_of_columns,Common_columns_number"

fieldNames="QuerySId,SessionSid,UsersId,NoP,NoS,NoA,NoT,NoAtt,NCP,NCS,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,Length,RED,Edit_index,Jackard_index,Cosine_index,Common_fragments_index,Common_tables_index,vote,ExplorationSid"

metrics=np.genfromtxt(metricspath, dtype=None, delimiter=';', names=fieldNames,  encoding='utf8',  skip_header=1)



##### LIST OF LF FUNCTIONS TO CHECK


LFs=[LF_edit_index,LF_recall_projections2,LF_jackard_index]
#LFs=[LF_edit_index,LF_jackard_index]


##### snorkeling


session = SnorkelSession()

doc_preprocessor = TSVDocPreprocessor(path)

corpus_parser = CorpusParser(parser=Spacy())
corpus_parser.apply(doc_preprocessor)


pairs = candidate_subclass('pairs1', ['queryPair'])
regexpmatch=RegexMatchSpan(rgx=".*")
cs=queryCandidate()
cand_extractor = CandidateExtractor(pairs, [cs], [regexpmatch])


docs = session.query(Document).order_by(Document.name).all()
sentences = session.query(Sentence).all()
#print(sentences)

sents=set();
for i,doc in enumerate(docs):
    for s in doc.sentences:
        sents.add(s)


cand_extractor.apply(sents)

print("Number of candidates:", session.query(pairs).count())


labeler = LabelAnnotator(lfs=LFs)

L_train = labeler.apply()

print(L_train.lf_stats(session))


# generative model, training_marginals are probabilistic training labels
gen_model = GenerativeModel()
gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)


print(gen_model.weights.lf_accuracy)

train_marginals = gen_model.marginals(L_train)

plt.hist(train_marginals, bins=20)
plt.show()

print(gen_model.learned_lf_stats())


#L_dev = labeler.apply_existing()

##### writing down labels


matchpath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-cuts-snorkel-tmp.csv'
#matchpath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/check_match.csv'
with open(matchpath, mode='w') as match_file:
    match_writer = csv.writer(match_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    match_writer.writerow(['idSession','idQuery','cut'])
    i=0
    for c in session.query(pairs):
        #print('i=',i)
        #print('marginal=',train_marginals[i])
        (idQ1,idQ2,idSession)=getCandidatesIDs(c)
        #print('session=',idSession)
        #print('idQ1=',idQ1)
        #print('idQ2=',idQ2)
        #print('--------')
        tmp=metrics[metrics['SessionSid']==idSession]
        #cut=tmp[tmp['QuerySId']==idQ2]['GroundTruth'][0]
        session=tmp[tmp['QuerySId']==idQ2]['SessionSid'][0]
        query=tmp[tmp['QuerySId']==idQ2]['QuerySId'][0]
        marginal=train_marginals[i]
        if marginal <0.8:
            cut=1
        else:
            cut=0
        #print('session=',session)
        #print('query order=',query)
        #print(' cut=',1-cut)
        match_writer.writerow([idSession,idQ2,cut])
        i=i+1






# matchpath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-cuts-snorkel.csv'
# with open(matchpath, mode='w+') as match_file:
#     match_writer = csv.writer(match_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for marginal in train_marginals:
#         if marginal<0.1:
#             cut=1
#         else:
#             cut=0
#         match_writer.writerow([cut])


