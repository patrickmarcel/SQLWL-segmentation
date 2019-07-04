import csv

def creer_vecteur(filename):
    tab = []
    cr = csv.reader(open(filename,'r'))
    for row in cr:
        tab.append(float(row[0]))
    return tab

def creer_matrice(matrix):
    matrice = []
    for i in matrix[0]:
        matrice.append([0,0])
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if(matrix[i][j] == 0):
                matrice[j][1] = matrice[j][1] + 1
            else:
                matrice[j][0] = matrice[j][0] + 1
    return matrice

def creer_matrice2(matrix):
    matrice = [[0,0],[0,0]]
    for i in range(len(matrix[0])):
        if(matrix[0][i] == matrix[1][i]):
            if(matrix[0][i] == 1):
                matrice[0][0] = matrice[0][0]+1
            else:
                matrice[1][1] = matrice[1][1]+1
        else:
            if(matrix[0][i] == 1):
                matrice[0][1] = matrice[0][1]+1
            else:
                matrice[1][0] = matrice[1][0]+1
    return matrice

def divise_tableau(tab,nombre):
    tableau = [];
    for i in tab:
        tableau.append(i/nombre)
    return tableau

def somme_tableau(tab):
    somme = 0
    for i in tab:
        somme = somme + i
    return somme

def somme_carre_tableau(tab):
    somme = 0
    for i in tab:
        somme = somme + i**2
    return somme

def calcul_ligne(tab,observateurs):
    return 1/(observateurs*(observateurs-1)) * (somme_carre_tableau(tab) - observateurs)

def fleiss_kappa(matrix):
    matrix = creer_matrice(matrix)
    total = 0
    accord = 0
    nombreDesaccord = 0
    nombreConsensus = 0
    observateurs = somme_tableau(matrix[0])
    tabColonne = [0] * len(matrix[0])
    tabLigne = []
    for i in range(len(matrix)):
        tabLigne.append(calcul_ligne(matrix[i],observateurs))
        maximum = 0
        for j in range(len(matrix[i])):
            if(j == 0):
                if(matrix[i][j] != 0 and matrix[i][j] != observateurs):
                    nombreDesaccord = nombreDesaccord + 1
            total = total + matrix[i][j];
            tabColonne[j] = tabColonne[j]+matrix[i][j]

            if(matrix[i][j] > maximum):
                maximum = matrix[i][j]

        nombreConsensus = nombreConsensus + maximum
    tabColonne = divise_tableau(tabColonne, total)
    p = 1/len(matrix) * somme_tableau(tabLigne)
    pe = somme_carre_tableau(tabColonne)
    accord = (p - pe)/(1 - pe)
    pdesaccord = nombreDesaccord / len(matrix)
    consensus = nombreConsensus / (len(matrix)*observateurs)
    # print('Ratio desaccord : ',pdesaccord)
    # print('Consensus : ',consensus)
    return accord

def cohen_kappa(matrix):
    matrix = creer_matrice2(matrix)
    daccord = 0;
    total = 0;
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            total = total + matrix[i][j]
            if(i == j):
                daccord = daccord + matrix[i][j]
    paccord = daccord / total
    pdesaccord = 1-paccord
    # print('Ratio dessacord : ',pdesaccord)
    poui = ( (matrix[0][0] + matrix[0][1]) * (matrix[0][0] + matrix[1][0]) )/(total**2)
    pnon = ( (matrix[1][0] + matrix[1][1]) * (matrix[0][1] + matrix[1][1]) )/(total**2)
    phasard = poui+pnon

    accord = (paccord - phasard)/(1 - phasard)
    return accord


def inverse(binary_vector):
    inverse=[]
    for i in binary_vector:
        if i==1:
            inverse.append(float(0))
        else:
            inverse.append(float(1))
    return inverse

def confusion3(v1,v2,v3):
    nb000=0
    nb001=0
    nb010=0
    nb011=0
    nb100=0
    nb101=0
    nb110=0
    nb111=0
    for i in range(0,len(v1)):
        if v1[i]==0 and v2[i]==0 and v3[i]==0:
            nb000=nb000+1
        if v1[i]==0 and v2[i]==0 and v3[i]==1:
            nb001=nb001+1
        if v1[i]==0 and v2[i]==1 and v3[i]==0:
            nb010=nb010+1
        if v1[i]==0 and v2[i]==1 and v3[i]==1:
            nb011=nb011+1
        if v1[i]==1 and v2[i]==0 and v3[i]==0:
            nb100=nb100+1
        if v1[i]==1 and v2[i]==0 and v3[i]==1:
            nb101=nb101+1
        if v1[i]==1 and v2[i]==1 and v3[i]==0:
            nb110=nb110+1
        if v1[i]==1 and v2[i]==1 and v3[i]==1:
            nb111=nb111+1
    return nb000/len(v1),nb001/len(v1),nb010/len(v1),nb011/len(v1),nb100/len(v1),nb101/len(v1),nb110/len(v1),nb111/len(v1)


def scores(v1,v2):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(0,len(v1)):
        if v1[i]==1 and v2[i]==1:
            tp=tp+1
        if v1[i]==0 and v2[i]==0:
            tn=tn+1
        if v1[i]==1 and v2[i]==0:
            fp=fp+1
        if v1[i]==0 and v2[i]==1:
            fn=fn+1

    wprecision=tp/(tp+fp)
    wrecall=tp/(tp+fn)
    waccuracy=(tp+tn)/(tp+fn+fp+tn)
    wfmeasure=(2*wprecision*wrecall)/(wprecision+wrecall)
    return waccuracy,wprecision,wrecall,wfmeasure


#adasyn = creer_vecteur('/home/hadoop/Desktop/ACP/Data_pour_papier_Information_System/sqlShareCuts-ADASYN.csv')
#smote = creer_vecteur('/home/hadoop/Desktop/ACP/Data_pour_papier_Information_System/sqlShareCuts-SMOTE.csv')
#under=creer_vecteur('/home/hadoop/Desktop/ACP/Data_pour_papier_Information_System/sqlShareCuts-RandomUnderSampler.csv')
#over=creer_vecteur('/home/hadoop/Desktop/ACP/Data_pour_papier_Information_System/sqlShareCuts-RandomOverSampler.csv')


snorkel=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-cuts-snorkel.csv') # with edit and jaccard
#snorkel=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-cuts-snorkel_v2.csv') # with edit and CTI
transfer=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlShareCuts-SMOTE.csv')
vote=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshareCuts-vote.csv')

voteConcatenate=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/concatenateCuts-vote.csv')
transferConcatenate=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/concatenateCuts-transfer.csv')
snorkelConcatenate=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/concatenate-cuts-snorkel.csv')
groundTruth=creer_vecteur('/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/concatenate-groundTruth.csv')

vote=inverse(vote)



matrice = [vote,transfer,snorkel]
m0 = [snorkel,transfer]
m1 = [snorkel, vote]
m2 = [vote, transfer]

matrice2 = [voteConcatenate,transferConcatenate,snorkelConcatenate]
m3 = [voteConcatenate, transferConcatenate]
m4 = [voteConcatenate, groundTruth]
m5 = [transferConcatenate, groundTruth]
m6 = [transferConcatenate, snorkelConcatenate]
m7 = [groundTruth, snorkelConcatenate]
m8 = [voteConcatenate, snorkelConcatenate]


#### print results
print("--------------------------------------------------")
print('Agreements on ground truth - concatenate dataset')
print("--------------------------------------------------")

print('vote, transfer: ',cohen_kappa(m3))
print('vote - gt: ',cohen_kappa(m4))
print('transfer - gt: ',cohen_kappa(m5))
print('snorkel, transfer: ',cohen_kappa(m6))
print('snorkel - gt: ',cohen_kappa(m7))
print('vote - snorkel: ',cohen_kappa(m8))

print('Fleiss: ',fleiss_kappa(matrice2))

print('confusion for snorkel,vote,transfer: ',confusion3(voteConcatenate,transferConcatenate,snorkelConcatenate))



print("--------------------------------------------------")
print('Scores on ground truth')
print("--------------------------------------------------")

print('Scores vote (Accuracy, Precision, Recall, F1)',scores(voteConcatenate,groundTruth))
print('Scores transfer (Accuracy, Precision, Recall, F1)',scores(transferConcatenate,groundTruth))
print('Scores snorkel (Accuracy, Precision, Recall, F1)',scores(snorkelConcatenate,groundTruth))



print("--------------------------------------------------")
print('Agreements on SQLShare')
print("--------------------------------------------------")

print('snorkel, transfer: ',cohen_kappa(m0))
print('snorkel, vote: ',cohen_kappa(m1))
print('vote, transfer: ',cohen_kappa(m2))

print('Fleiss: ',fleiss_kappa(matrice))

print('confusion for snorkel,vote,transfer: ',confusion3(vote,transfer,snorkel))


