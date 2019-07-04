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




# one big document with one sentence per exploration
# if -small -> just a little sample for testing purpose
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshareQueries-IDs-small.txt' 
#path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/DOPANQueries-IDs.txt' 
#path='/home/hadoop/Desktop/ACP/Data_pour_papier_Information_System/DOPAN-smartBI-queries-IDs.txt'

path='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/DOPAN-smartBI-queries-IDs.txt'


session = SnorkelSession()

doc_preprocessor = TSVDocPreprocessor(path)

corpus_parser = CorpusParser(parser=Spacy())

corpus_parser.apply(doc_preprocessor)
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())
#thedoc=session.query(Document).all()[0]



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

def printpairs(pairs):
    for cand in session.query(pairs):
        print('--------------------------------------------------')
        print(cand.queryPair)
        print('--------------------------------------------------')

#printpairs(pairs)

# 1=true, -1=false, 0=no label


#metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-legros-v1.csv'
#metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-legros-v1-nolength.csv'
#metricspath='/home/hadoop/Desktop/ACP/Data_pour_papier_Information_System/dopan-smartBI-concat-float.csv'
#metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v5.csv'



#fieldNames="ID,ID_EXPLORATION,Number_of_projections,Number_of_tables,Number_of_aggregation_functions,Number_of_selections,Common_projections_number,Common_selections_number,Common_aggregation_functions_number,Common_tables_number,QUERY_CHARACTER_NUMBER,Number_of_columns,Common_columns_number"

#fieldNames="QuerySId,SessionSid,UsersId,NoP,NoS,NoA,NoT,NoAtt,NCP,NCS,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,Length,RED,Edit_index,Jackard_index,Cosine_index,Common_fragments_index,Common_tables_index,vote,ExplorationSid"


# in that file, we have sessionID first and then queryId
# while candidates are queryID,sessionID

#fieldNames="QuerySId,SessionSid,UserSId,NoP,NoF,NoA,NoT,NoAtt,NCP,NCF,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,Lenght,RED,Edit-index,Jackard-index,Cosine-index,Common-fragments-index,Common-tables-index,Vote,ExplorationSId,GroundTruth,ChangeSession"


#fieldNames="QuerySId,SessionSid,UserSId,NoP,NoF,NoA,NoT,NoAtt,NCP,NCF,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,NoQ,RED,EditIndex,JackardIndex,CosineIndex,CommonFragmentsIndex,CommonTablesIndex,Vote,ExplorationSId,GroundTruth,ChangeSession"

metricspath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-v2.csv'


fieldNames="QuerySId,SessionSid,UsersId,NoP,NoS,NoA,NoT,NoAtt,NCP,NCS,NCA,NCT,NCAtt,zNoP,zNoS,zNoA,zNoT,zNoAtt,zNCP,zNCF,zNCA,zNCT,zNCAtt,RED,Edit_index,Jackard_index,Cosine_index,Common_fragments_index,Common_tables_index,vote,ExplorationSid,GroundTruth,ChangeSession"



metrics=np.genfromtxt(metricspath, dtype=None, delimiter=';', names=fieldNames, encoding='utf8', skip_header=1)

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
    if idQuery==0:
        return 0
    else:
        tmp=metrics[metrics['SessionSid']==idSession]
        #print(idSession,idQuery,name)
        return tmp[tmp['QuerySId']==idQuery][name][0]
    

    
def getCandidatesIDs(c):
    pair=c.queryPair.get_span()
    q1=pair.split(sep='|')[0]
    q2=pair.split(sep='|')[1]
    idQ1=q1.split(sep=',')[0]
    idSession=q1.split(',')[1]
    idQ2=q2.split(sep=',')[0]
    return (int(idQ1),int(idQ2),int(idSession))
    
    
    
    
    
# LF functions 
# 1: together
# -1: split
# 0: otherwise
def LF_recall_projections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCP',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_projections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCP',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_selections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCS',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_selections(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCS',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_aggregation(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCA',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_aggregation(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCA',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_tables(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCT',idQ1,idQ2,idSession)
    if rec==1: 
        return 1
    elif rec==0:
        return -1
    else:
        return 0
        

def LF_precision_tables(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCT',idQ2,idSession)
    if prec==1: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        

## version 2
##
## favors keeping queries together 
##
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
        

def LF_precision_projections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCP',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_selections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCS',idQ1,idQ2,idSession)
    if rec!=0: 
        return 1
    else:
        score=findMetric('NoS',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1
        

def LF_precision_selections2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCS',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_aggregation2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCA',idQ1,idQ2,idSession)
    if rec!=0: 
        return 1
    else:
        score=findMetric('NoA',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1
        

def LF_precision_aggregation2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCA',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
def LF_recall_tables2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    rec=recall('NCT',idQ1,idQ2,idSession)
    if rec!=0: 
        return 1
    else:
        score=findMetric('NoT',idQ1,idSession)
        if score==0:
            return 0
        else:
            return -1
        

def LF_precision_tables2(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    prec=precision('NCT',idQ2,idSession)
    if prec!=0: 
        return 1
    elif prec==0:
        return -1
    else:
        return 0
        
##
## less in common

def LF_lessInCommonProjection(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCP',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
      
def LF_lessInCommonSelection(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCS',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
        
def LF_lessInCommonAggregation(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCA',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
  
def LF_lessInCommonTable(c):
    (idQ1,idQ2,idSession)=getCandidatesIDs(c)
    less=lessInCommon('NCT',idQ1,idQ2,idSession)
    if less==True:
        return -1
    else:
        return 1
          
                
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


def LF_cosine_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    cosine_index = findMetric('Cosine_index', idQ2, idSession)
    if cosine_index > 0:
        return 1
    else:
        return -1


def LF_common_fragment_index(c):
    (idQ1, idQ2, idSession) = getCandidatesIDs(c)
    common_fragment_index = findMetric('Common_fragments_index', idQ2, idSession)
    if common_fragment_index > 0:
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
        
    
# majority class    
        
def LF_always1(c):
    return 1

# testing   
def testLFs(): 
    for c in session.query(pairs):
        print(c.queryPair.sentence.text[c.queryPair.char_start:c.queryPair.char_end])
        print(c.labels)


def get_power_set(s):
    power_set=[[]]
    for elem in s:
        # iterate over the sub sets so far
        for sub_set in power_set:
          # add a new subset consisting of the subset at hand added elem
          power_set=power_set+[list(sub_set)+[elem]]
    return power_set

def nombreGroupes():
    nombre = 0
    while(nombre != 3 and nombre != 4):
        print ('Combien de groupes souhaitez-vous ? (3 ou 4)')
        valeur = input()
        try:
            nombre = int(valeur)
        except ValueError:
            print ('')
    return nombre

def constitutionGroupes():
    nombre = nombreGroupes()
    groupe = 0
    groupes = []
    for i in range(nombre):
        groupes.append([])
    
    for x in LFs:
        groupe = 0
        while(groupe < 1 or groupe > nombre):
            print ('Dans quel groupe voulez-vous mettre ', x)
            valeur = input()
            try:
                groupe = int(valeur)
                if(groupe < 1 or groupe > nombre):
                    print('Entrez un chiffre entre 1 et ',nombre)
                else:
                    groupes[groupe-1].append(x)
            except ValueError:
                print('Entrez un chiffre entre 1 et ',nombre)
    
    return groupes

# retourne un tableau de 2 valeurs : indice de la valeur maxi et la valeur maxi
def bon_indice(liste):
    maxi = 0
    bon = 0
    for i in range(len(liste)):
        if(liste[i] > maxi):
            maxi = liste[i]
            bon = i
    return [bon,maxi]

# LFs = [LF_recall_projections, LF_precision_projections, LF_recall_selections,
#        LF_precision_selections, LF_recall_aggregation, LF_precision_aggregation,
#        LF_recall_tables, LF_precision_tables,
#        LF_recall_projections2, LF_precision_projections2, LF_recall_selections2,
#        LF_precision_selections2, LF_recall_aggregation2, LF_precision_aggregation2,
#        LF_recall_tables2, LF_precision_tables2, LF_edit_index, LF_jackard_index,
#        LF_cosine_index, LF_common_fragment_index, LF_Common_Tables_Index]
       
LFs = [LF_recall_projections2,  LF_recall_selections2,
       LF_recall_tables2, LF_edit_index, LF_jackard_index,
       LF_common_fragment_index, LF_Common_Tables_Index]

# nb tested combinations=512+32+10
# recallproj edit jaccard / recall table, CTI, CFI / recall select
# recall proj CTI / recall table jaccard / CFI selection edit

filePath = '/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/searchScores-bestComb.csv';

 
# As file at filePath is deleted now, so we should check if file exists or not not before deleting them
if os.path.exists(filePath):
    os.remove(filePath)
    
groupes = constitutionGroupes()

ListFs = []
ListFs2 = []

for groupe in groupes:
    ListFmesure = [[],[]]
    
    posetLF=get_power_set(groupe)
    posetLF=posetLF[1:]

    for LFs in posetLF:
        labeler = LabelAnnotator(lfs=LFs)

        session = SnorkelSession()

        doc_preprocessor = TSVDocPreprocessor(path)
        corpus_parser = CorpusParser(parser=Spacy())
        corpus_parser.apply(doc_preprocessor)
    #    print("Documents:", session.query(Document).count())
    #    print("Sentences:", session.query(Sentence).count())

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

        #np.random.seed(1701)
        L_train = labeler.apply()
        #print(L_train)
        print(L_train.lf_stats(session))


        # generative model, training_marginals are probabilistic training labels
        gen_model = GenerativeModel()
        gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)


        print(gen_model.weights.lf_accuracy)

        train_marginals = gen_model.marginals(L_train)

        plt.hist(train_marginals, bins=20)
        plt.show()

        print(gen_model.learned_lf_stats())


        # check ground truth


        matchpath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/searchScores-bestComb.csv'
        with open(matchpath, mode='a') as match_file:
            match_writer = csv.writer(match_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            i=0
            tp=0
            tn=0
            fp=0
            fn=0
            for c in session.query(pairs):
                (idQ1,idQ2,idSession)=getCandidatesIDs(c)
                tmp=metrics[metrics['SessionSid']==idSession]
                cut=tmp[tmp['QuerySId']==idQ2]['GroundTruth'][0]
                session=tmp[tmp['QuerySId']==idQ2]['SessionSid'][0]
                query=tmp[tmp['QuerySId']==idQ2]['QuerySId'][0]
                marginal=train_marginals[i]
                if cut==1 and marginal<0.8:
                    tp=tp+1
                if cut==0 and marginal>=0.8:
                    tn=tn+1
                if cut==0 and marginal<0.8:
                    fp=fp+1
                if cut==1 and marginal>=0.8:
                    fn=fn+1
                i=i+1
            wprecision=tp/(tp+fp)
            wrecall=tp/(tp+fn)
            waccuracy=(tp+tn)/(tp+fn+fp+tn)
            wfmeasure=(2*wprecision*wrecall)/(wprecision+wrecall)
            print(LFs)    
            print('F-measure=',wfmeasure)
            print('precision=',wprecision)
            print('recall=',wrecall)
            print('accuracy=',waccuracy)
            match_writer.writerow([LFs, wfmeasure, waccuracy, wprecision, wrecall])
            ListFmesure[0].append(LFs)
            ListFmesure[1].append(wfmeasure)

            
    if(len(ListFmesure[0])):
        indice = bon_indice(ListFmesure[1])[0]
        maxi = bon_indice(ListFmesure[1])[1]
        ListFs.append((ListFmesure[0][indice],maxi))
        
print('FIN !!')
print()

boucle = True
print('Début boucle !!')
while(boucle):
    changement = False
    groupes = []
    ListFmesure = [[],[]]
    l = get_power_set(ListFs)
    for element in l:
        if(len(element) > 1):
            fonctions = []
            fmesure = []
            for tuple_ in element:
                indice = 0
                for t in tuple_ :
                    if(indice == 0):
                        for f in t:
                            fonctions.append(f)
                    else:
                        fmesure.append(t)
                    indice = 1
            groupes.append((fonctions,bon_indice(fmesure)[1]))

    print('GROUPES : ')
    print(groupes)
    if(len(groupes) == 0):
        boucle = False
    else:
        for groupe in groupes:
            indice = 0
            for LFs in groupe:
                if(indice == 0):
                    print(LFs)
                    session = SnorkelSession()
                    doc_preprocessor = TSVDocPreprocessor(path)
                    corpus_parser = CorpusParser(parser=Spacy())
                    corpus_parser.apply(doc_preprocessor)
                #    print("Documents:", session.query(Document).count())
                #    print("Sentences:", session.query(Sentence).count())

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

                    #np.random.seed(1701)
                    L_train = labeler.apply()
                    #print(L_train)
                    print(L_train.lf_stats(session))


                    # generative model, training_marginals are probabilistic training labels
                    gen_model = GenerativeModel()
                    gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)


                    print(gen_model.weights.lf_accuracy)

                    train_marginals = gen_model.marginals(L_train)

                    plt.hist(train_marginals, bins=20)
                    plt.show()

                    print(gen_model.learned_lf_stats())


                    # check ground truth


                    matchpath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/searchScores-bestComb.csv'
                    with open(matchpath, mode='a') as match_file:
                        match_writer = csv.writer(match_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        i=0
                        tp=0
                        tn=0
                        fp=0
                        fn=0
                        for c in session.query(pairs):
                            (idQ1,idQ2,idSession)=getCandidatesIDs(c)
                            tmp=metrics[metrics['SessionSid']==idSession]
                            cut=tmp[tmp['QuerySId']==idQ2]['GroundTruth'][0]
                            session=tmp[tmp['QuerySId']==idQ2]['SessionSid'][0]
                            query=tmp[tmp['QuerySId']==idQ2]['QuerySId'][0]
                            marginal=train_marginals[i]
                            if cut==1 and marginal<0.8:
                                tp=tp+1
                            if cut==0 and marginal>=0.8:
                                tn=tn+1
                            if cut==0 and marginal<0.8:
                                fp=fp+1
                            if cut==1 and marginal>=0.8:
                                fn=fn+1
                            i=i+1
                        wprecision=tp/(tp+fp)
                        wrecall=tp/(tp+fn)
                        waccuracy=(tp+tn)/(tp+fn+fp+tn)
                        wfmeasure=(2*wprecision*wrecall)/(wprecision+wrecall)
                        print(LFs)    
                        print('F-measure=',wfmeasure)
                        print('precision=',wprecision)
                        print('recall=',wrecall)
                        print('accuracy=',waccuracy)
                        match_writer.writerow([LFs, wfmeasure, waccuracy, wprecision, wrecall])
                        ListFmesure[0].append(LFs)
                        ListFmesure[1].append(wfmeasure)
                elif(indice == 1):
                    fmesureMax = LFs
                indice = indice+1

    ListFs2 = ListFs
    ListFs = []
    if(len(ListFmesure[0]) > 0):
        indice = bon_indice(ListFmesure[1])[0]
        maxi = bon_indice(ListFmesure[1])[1]
        if(maxi > fmesureMax):
            ListFs.append((ListFmesure[0][indice],maxi))
            changement = True
    if(changement == False):
        boucle = False

print()
print('Fin boucle !!')
print()
print(ListFs2)
