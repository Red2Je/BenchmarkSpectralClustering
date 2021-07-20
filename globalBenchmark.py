import numpy as np
import math
from numpy.linalg import inv
from time import time
from numpy.core.numeric import full
from numpy.random import rand
from scipy.linalg.misc import norm
from scipy.sparse.linalg.eigen import lobpcg
from scipy.sparse.linalg import eigs,eigsh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

k = 3

def eps(X,e):
    sX,column = X.shape
    output = np.zeros((sX,sX))
    for i in range(0,sX):
        for j in range(0,sX):
            if(norm(X[i,:]-X[j,:])<e):
                output[i,j] = 1
    return(output)

def kv(X,k):
    sX,colum = X.shape
    output = np.zeros((sX,sX))
    for i in range(0,sX):
        lvoisin = np.zeros(sX)
        for j in range(0,sX):
            if(i != j):
                lvoisin[j] = norm(X[i,:]-X[j,:])
        ii = np.argsort(lvoisin)
        ii = ii[1:k+1]
        for index in ii:
            output[i,index] = lvoisin[index]
            output[index,i] = lvoisin[index]
    return output

def fully(X,sigma):
    sX = X.shape[0]
    output = np.zeros((sX,sX))
    for i in range(0,sX):
        for j in range(0,sX):
            dist = norm(X[i,:]-X[j,:])
            output[i,j] = math.exp(-(dist ** 2)/(2*(sigma**2)))
    return output

#print("data = \n",data)
#epsilon = 2
#sigma = 5
def computeD(S):
    sS = S.shape[0]
    output = np.zeros(S.shape)
    for i in range(0,sS):
        output[i,i] = np.sum(S[i,:])
    return output
def computeL(S):
    D = computeD(S)
    return D - S

def computeLsym(S):
    L = computeL(S)
    D = computeD(S)
    invD = inv(D)**(1/2)
    return(invD*L*invD)

def computeLrw(S):
    L = computeL(S)
    D = computeD(S)
    return(L,D)


def LsymOrtho(vecs):
    svecs,k = vecs.shape
    T = np.zeros(vecs.shape)
    for i in range(0,svecs):
        s = 0
        if(np.sum(vecs[i,:]**2) != 0):
            s = np.sum(vecs[i,:]**2)**(-1/2)
        for j in range(0,k):
            T[i,j] = vecs[i,j]*s
    return T


#unnormalized Laplacian
tEpsUnArn = []
tKnnUnArn = []
tFulUnArn = []

tEpsUnLan = []
tKnnUnLan = []
tFulUnLan = []

tEpsUnLOB = []
tKnnUnLOB = []
tFulUnLOB = []

#Lsym

tEpsLsymArn = []
tKnnLsymArn = []
tFulLsymArn = []

tEpsLsymLan = []
tKnnLsymLan = []
tFulLsymLan = []

tEpsLsymLOB = []
tKnnLsymLOB = []
tFulLsymLOB = []

#Lrw

tEpsLrwArn = []
tKnnLrwArn = []
tFulLrwArn = []

tEpsLrwLan = []
tKnnLrwLan = []
tFulLrwLan = []

tEpsLrwLOB = []
tKnnLrwLOB = []
tFulLrwLOB = []

matrixAmount = 1000

for n in range(5,matrixAmount):
    print("iter = ",n)
    data = rand(n,n)
    
    #computing similarity graph times
    diffEpsStart = time()
    listedDiff = [norm(data[i,:]-data[i+1,:]) for i in range(0,data.shape[0]-1)]
    simiEps = eps(data,np.average(listedDiff))
    diffEpsStop = time()
    tEps = diffEpsStop - diffEpsStart

    diffKnnStart = time()
    simiKnn = kv(data,int(math.log(n)))
    diffKnnStop = time()
    tKnn = diffKnnStop - diffKnnStart
    
    diffFulStart = time()
    simiFul = fully(data,5)
    diffFulStop = time()
    tFul = diffFulStop - diffFulStart
    #to add : tEps,tKnn,tFul

    #computing Laplacian creation time
    
    #UnNormalized
    diffUnEpsStart = time()
    unEps = computeL(simiEps)
    diffUnEpsStop = time()
    tUnEps = diffUnEpsStop - diffUnEpsStart

    diffUnKnnStart = time()
    unKnn = computeL(simiKnn)
    diffUnKnnStop = time()
    tUnKnn = diffUnKnnStop - diffUnKnnStart

    diffUnFulStart = time()
    unFul = computeL(simiFul)
    diffUnFulStop = time()
    tUnFul = diffUnFulStop - diffUnFulStart

    #Lsym
    diffLsymEpsStart = time()
    LsymEps = computeLsym(simiEps)
    diffLsymEpsStop = time()
    tLsymEps = diffLsymEpsStop - diffLsymEpsStart

    diffLsymKnnStart = time()
    LsymKnn = computeLsym(simiKnn)
    diffLsymKnnStop = time()
    tLsymKnn = diffLsymKnnStop - diffLsymKnnStart

    diffLsymFulStart = time()
    LsymFul = computeLsym(simiFul)
    diffLsymFulStop = time()
    tLsymFul = diffLsymFulStop - diffLsymFulStart

    #Lrw
    diffLrwEpsStart = time()
    LrwEps,DrwEps = computeLrw(simiEps)
    diffLrwEpsStop = time()
    tLrwEps = diffLrwEpsStop - diffLrwEpsStart

    diffLrwKnnStart = time()
    LrwKnn,DrwKnn = computeLrw(simiKnn)
    diffLrwKnnStop = time()
    tLrwKnn = diffLrwKnnStop - diffLrwKnnStart

    diffLrwFulStart = time()
    LrwFul,DrwFul = computeLrw(simiFul)
    diffLrwFulStop = time()
    tLrwFul = diffLrwFulStop - diffLrwFulStart

    #finding eigenVectors

    #########################################################################################
    #UnNormalized Arnoldi
    t0 = time()
    vals,unEpsArn = eigs(unEps,k,which = 'SR',ncv = 2*n+1)
    unEpsArn = np.real(unEpsArn)
    t1 = time()
    tUnEpsArn = t1-t0 

    t0 = time()
    vals,unKnnArn = eigs(unKnn,k,which = 'SR')
    unKnnArn = np.real(unKnnArn)
    t1 = time()
    tUnKnnArn = t1-t0

    t0 = time()
    vals,unFulArn = eigs(unFul,k,which = 'SR')
    unFulArn = np.real(unFulArn)
    t1 = time()
    tUnFulArn = t1-t0

    #UnNormalized Lanczos
    t0 = time()
    vals,unEpsLan = eigsh(unEps,k,which = 'SA')
    t1 = time()
    tUnEpsLan = t1-t0 

    t0 = time()
    vals,unKnnLan = eigsh(unKnn,k,which = 'SA')
    t1 = time()
    tUnKnnLan = t1-t0

    t0 = time()
    vals,unFulLan = eigsh(unFul,k,which = 'SA')
    t1 = time()
    tUnFulLan = t1-t0

    #UnNormalized, Lobpcg
    X = rand(n,k)
    t0 = time()
    vals,unEpsLob = lobpcg(unEps,X,largest=False, maxiter=2000)
    t1 = time()
    tUnEpsLob = t1-t0 

    t0 = time()
    vals,unKnnLob = lobpcg(unKnn,X,largest=False,maxiter=2000)
    t1 = time()
    tUnKnnLob = t1-t0

    t0 = time()
    vals,unFulLob = lobpcg(unFul,X,largest=False,maxiter=2000)
    t1 = time()
    tUnFulLob = t1-t0

    #########################################################################################
    #Lsym Arnoldi
    t0 = time()
    vals,LsymEpsArn = eigs(LsymEps,k,which = 'SR')
    LsymEpsArn = np.real(LsymEpsArn)
    LsymEpsArn = LsymOrtho(LsymEpsArn)
    t1 = time()
    tLsymEpsArn = t1-t0 

    t0 = time()
    vals,LsymKnnArn = eigs(LsymKnn,k,which = 'SR',tol = 0,ncv = 2*n+1)
    LsymKnnArn = np.real(LsymKnnArn)
    LsymKnnArn = LsymOrtho(LsymKnnArn)
    t1 = time()
    tLsymKnnArn = t1-t0

    t0 = time()
    vals,LsymFulArn = eigs(LsymFul,k,which = 'SR')
    LsymFulArn = np.real(LsymFulArn)
    LsymFulArn = LsymOrtho(LsymFulArn)
    t1 = time()
    tLsymFulArn = t1-t0

    #Lsym Lanczos
    t0 = time()
    vals,LsymEpsLan = eigsh(LsymEps,k,which = 'SA')
    LsymEpsLan = LsymOrtho(LsymEpsLan)
    t1 = time()
    tLsymEpsLan = t1-t0 

    t0 = time()
    vals,LsymKnnLan = eigsh(LsymKnn,k,which = 'SA',ncv = 2*n+1)
    LsymKnnLan = LsymOrtho(LsymKnnLan)
    t1 = time()
    tLsymKnnLan = t1-t0

    t0 = time()
    vals,LsymFulLan = eigsh(LsymFul,k,which = 'SA')
    LsymFulLan = LsymOrtho(LsymFulLan)
    t1 = time()
    tLsymFulLan = t1-t0

    #Lsym, Lobpcg
    X = rand(n,k)
    t0 = time()
    vals,LsymEpsLob = lobpcg(LsymEps,X,largest=False,maxiter=2000)
    t1 = time()
    tLsymEpsLob = t1-t0 

    t0 = time()
    vals,LsymKnnLob = lobpcg(LsymKnn,X,largest=False,maxiter=2000)
    t1 = time()
    tLsymKnnLob = t1-t0

    t0 = time()
    vals,LsymFulLob = lobpcg(LsymFul,X,largest=False,maxiter=2000)
    t1 = time()
    tLsymFulLob = t1-t0

    #########################################################################################
    invDeps = inv(DrwEps)
    invDknn = inv(DrwKnn)
    invDful = inv(DrwFul)
    #Lrw Arnoldi
    t0 = time()
    vals,LrwEpsArn = eigs(invDeps*LrwEps,k,which = 'SR')
    LrwEpsArn = np.real(LrwEpsArn)
    t1 = time()
    tLrwEpsArn = t1-t0 

    t0 = time()
    vals,LrwKnnArn = eigs(invDknn*LrwKnn,k,which = 'SR', ncv = 2*n+1)
    LrwKnnArn = np.real(LrwKnnArn)
    t1 = time()
    tLrwKnnArn = t1-t0

    t0 = time()
    vals,LrwFulArn = eigs(invDful*LrwFul,k,which = 'SR')
    LrwFulArn = np.real(LrwFulArn)
    t1 = time()
    tLrwFulArn = t1-t0

    #Lrw Lanczos
    t0 = time()
    vals,LrwEpsLan = eigsh(invDeps*LrwEps,k,which = 'SA')
    t1 = time()
    tLrwEpsLan = t1-t0 

    t0 = time()
    vals,LrwKnnLan = eigsh(invDknn*LrwKnn,k,which = 'SA',ncv = 2*n+1)
    t1 = time()
    tLrwKnnLan = t1-t0

    t0 = time()
    vals,LrwFulLan = eigsh(invDful*LrwFul,k,which = 'SA')
    t1 = time()
    tLrwFulLan = t1-t0

    #Lrw, Lobpcg
    X = rand(n,k)
    t0 = time()
    vals,LrwEpsLob = lobpcg(LrwEps,X,B = DrwEps,largest=False,maxiter=2000)
    t1 = time()
    tLrwEpsLob = t1-t0 

    t0 = time()
    vals,LrwKnnLob = lobpcg(LrwKnn,X,B = DrwKnn,largest=False,maxiter=2000)
    t1 = time()
    tLrwKnnLob = t1-t0

    t0 = time()
    vals,LrwFulLob = lobpcg(LrwFul,X,B = DrwFul,largest=False,maxiter=2000)
    t1 = time()
    tLrwFulLob = t1-t0

    #########################################################################################
    #K-means
    #UnNormalized Arnoldi
    t0 = time()
    kmUnEpsArn = KMeans(n_clusters=k).fit(unEpsArn)
    t1 = time()
    tKmUnEpsArn = t1-t0

    t0 = time()
    kmUnKnnArn = KMeans(n_clusters=k).fit(unKnnArn)
    t1 = time()
    tKmUnKnnArn = t1-t0

    t0 = time()
    kmUnFulArn = KMeans(n_clusters=k).fit(unFulArn)
    t1 = time()
    tKmUnFulArn = t1-t0

    #UnNormalized Lanczos
    t0 = time()
    kmUnEpsLan = KMeans(n_clusters=k).fit(unEpsLan)
    t1 = time()
    tKmUnEpsLan = t1-t0

    t0 = time()
    kmUnKnnLan = KMeans(n_clusters=k).fit(unKnnLan)
    t1 = time()
    tKmUnKnnLan = t1-t0

    t0 = time()
    kmUnFulLan = KMeans(n_clusters=k).fit(unFulLan)
    t1 = time()
    tKmUnFulLan = t1-t0

    #UnNormalized LOOBPCG
    t0 = time()
    kmUnEpsLob = KMeans(n_clusters=k).fit(unEpsLob)
    t1 = time()
    tKmUnEpsLob = t1-t0

    t0 = time()
    kmUnKnnLob = KMeans(n_clusters=k).fit(unKnnLob)
    t1 = time()
    tKmUnKnnLob = t1-t0

    t0 = time()
    kmUnFulLob = KMeans(n_clusters=k).fit(unFulLob)
    t1 = time()
    tKmUnFulLob = t1-t0

    #Lsym Arnoldi
    t0 = time()
    kmLsymEpsArn = KMeans(n_clusters=k).fit(LsymEpsArn)
    t1 = time()
    tKmLsymEpsArn = t1-t0

    t0 = time()
    kmLsymKnnArn = KMeans(n_clusters=k).fit(LsymKnnArn)
    t1 = time()
    tKmLsymKnnArn = t1-t0

    t0 = time()
    kmLsymFulArn = KMeans(n_clusters=k).fit(LsymFulArn)
    t1 = time()
    tKmLsymFulArn = t1-t0

    #Lsym Lanczos
    t0 = time()
    kmLsymEpsLan = KMeans(n_clusters=k).fit(LsymEpsLan)
    t1 = time()
    tKmLsymEpsLan = t1-t0

    t0 = time()
    kmLsymKnnLan = KMeans(n_clusters=k).fit(LsymKnnLan)
    t1 = time()
    tKmLsymKnnLan = t1-t0

    t0 = time()
    kmLsymFulLan = KMeans(n_clusters=k).fit(LsymFulLan)
    t1 = time()
    tKmLsymFulLan = t1-t0

    #Lsym LOOBPCG
    t0 = time()
    kmLsymEpsLob = KMeans(n_clusters=k).fit(LsymEpsLob)
    t1 = time()
    tKmLsymEpsLob = t1-t0

    t0 = time()
    kmLsymKnnLob = KMeans(n_clusters=k).fit(LsymKnnLob)
    t1 = time()
    tKmLsymKnnLob = t1-t0

    t0 = time()
    kmLsymFulLob = KMeans(n_clusters=k).fit(LsymFulLob)
    t1 = time()
    tKmLsymFulLob = t1-t0

    #Lrw Arnoldi
    t0 = time()
    kmLrwEpsArn = KMeans(n_clusters=k).fit(LrwEpsArn)
    t1 = time()
    tKmLrwEpsArn = t1-t0

    t0 = time()
    kmLrwKnnArn = KMeans(n_clusters=k).fit(LrwKnnArn)
    t1 = time()
    tKmLrwKnnArn = t1-t0

    t0 = time()
    kmLrwFulArn = KMeans(n_clusters=k).fit(LrwFulArn)
    t1 = time()
    tKmLrwFulArn = t1-t0

    #Lrw Lanczos
    t0 = time()
    kmLrwEpsLan = KMeans(n_clusters=k).fit(LrwEpsLan)
    t1 = time()
    tKmLrwEpsLan = t1-t0

    t0 = time()
    kmLrwKnnLan = KMeans(n_clusters=k).fit(LrwKnnLan)
    t1 = time()
    tKmLrwKnnLan = t1-t0

    t0 = time()
    kmLrwFulLan = KMeans(n_clusters=k).fit(LrwFulLan)
    t1 = time()
    tKmLrwFulLan = t1-t0

    #Lrw LOOBPCG
    t0 = time()
    kmLrwEpsLob = KMeans(n_clusters=k).fit(LrwEpsLob)
    t1 = time()
    tKmLrwEpsLob = t1-t0

    t0 = time()
    kmLrwKnnLob = KMeans(n_clusters=k).fit(LrwKnnLob)
    t1 = time()
    tKmLrwKnnLob = t1-t0

    t0 = time()
    kmLrwFulLob = KMeans(n_clusters=k).fit(LrwFulLob)
    t1 = time()
    tKmLrwFulLob = t1-t0


    ###########################################################
    #putting inside time list
    #UnNormalized
    tEpsUnArn.append(tEps+tUnEps+tUnEpsArn+tKmUnEpsArn)
    tKnnUnArn.append(tKnn+tUnKnn+tUnKnnArn+tKmUnKnnArn)
    tFulUnArn.append(tFul+tUnFul+tUnFulArn+tKmUnFulArn)

    tEpsUnLan.append(tEps+tUnEps+tUnEpsLan+tKmUnEpsLan)
    tKnnUnLan.append(tKnn+tUnKnn+tUnKnnLan+tKmUnKnnLan)
    tFulUnLan.append(tFul+tUnFul+tUnFulLan+tKmUnFulLan)

    tEpsUnLOB.append(tEps+tUnEps+tUnEpsLob+tKmUnEpsLob)
    tKnnUnLOB.append(tKnn+tUnKnn+tUnKnnLob+tKmUnKnnLob)
    tFulUnLOB.append(tFul+tUnFul+tUnFulLob+tKmUnFulLob)

    #Lsym

    tEpsLsymArn.append(tEps+tLsymEps+tLsymEpsArn+tKmLsymEpsArn)
    tKnnLsymArn.append(tKnn+tLsymKnn+tLsymKnnArn+tKmLsymKnnArn)
    tFulLsymArn.append(tFul+tLsymFul+tLsymFulArn+tKmLsymFulArn)

    tEpsLsymLan.append(tEps+tLsymEps+tLsymEpsLan+tKmLsymEpsLan)
    tKnnLsymLan.append(tKnn+tLsymKnn+tLsymKnnLan+tKmLsymKnnLan)
    tFulLsymLan.append(tFul+tLsymFul+tLsymFulLan+tKmLsymFulLan)

    tEpsLsymLOB.append(tEps+tLsymEps+tLsymEpsLob+tKmLsymEpsLob)
    tKnnLsymLOB.append(tKnn+tLsymKnn+tLsymKnnLob+tKmLsymKnnLob)
    tFulLsymLOB.append(tFul+tLsymFul+tLsymFulLob+tKmLsymFulLob)

    #Lrw

    tEpsLrwArn.append(tEps+tLrwEps+tLrwEpsArn+tKmLrwEpsArn)
    tKnnLrwArn.append(tKnn+tLrwKnn+tLrwKnnArn+tKmLrwKnnArn)
    tFulLrwArn.append(tFul+tLrwFul+tLrwFulArn+tKmLrwFulArn)

    tEpsLrwLan.append(tEps+tLrwEps+tLrwEpsLan+tKmLrwEpsLan)
    tKnnLrwLan.append(tKnn+tLrwKnn+tLrwKnnLan+tKmLrwKnnLan)
    tFulLrwLan.append(tFul+tLrwFul+tLrwFulLan+tKmLrwFulLan)

    tEpsLrwLOB.append(tEps+tLrwEps+tLrwEpsLob+tKmLrwEpsLob)
    tKnnLrwLOB.append(tKnn+tLrwKnn+tLrwKnnLob+tKmLrwKnnLob)
    tFulLrwLOB.append(tFul+tLrwFul+tLrwFulLob+tKmLrwFulLob)


toTraceArn = [tEpsUnArn,tKnnUnArn,tFulUnArn,tEpsLsymArn,tKnnLsymArn,tFulLsymArn,tEpsLrwArn,tKnnLrwArn,tFulLrwArn]
toTraceLan = [tEpsUnLan,tKnnUnLan,tFulUnLan,tEpsLsymLan,tKnnLsymLan,tFulLsymLan,tEpsLrwLan,tKnnLrwLan,tFulLrwLan]
toTraceLob = [tEpsUnLOB,tKnnUnLOB,tFulUnLOB,tEpsLsymLOB,tKnnLsymLOB,tFulLsymLOB,tEpsLrwLOB,tKnnLrwLOB,tFulLrwLOB]

def saveList(l,fileName):
    with open(fileName+'.txt','w') as f:
        for i in l:
            for index,j in enumerate(i):
                if(index != len(i)-1):
                    f.write('%s,' % j)
                else:
                    f.write('%s' % j)
            f.write('\n')

saveList(toTraceArn,'Arn')
saveList(toTraceLan,'Lan')
saveList(toTraceLob,'Lob')


def readListFile(fileName):
    output = []
    with open(fileName+'.txt','r') as f:
        content = f.read()
        mat = content.split('\n')[:-1]
        for entry in mat:
            unitary = entry.split(',')
            unitary = [float(x) for x in unitary]
            output.append(unitary)
    return(output)



#labelSimi = ["epsArn","knnArn","fulArn","epsLan","knnLan","fulLan","epsLob","knnLob","fulLob"]
#legendAxes = ["Un","Un","Un","Lsym","Lsym","Lsym","Lrw","Lrw","Lrw"]
#lines0 = []
#lines1 = []
#lines2 = []

"""
##and then we plot
x = range(5,matrixAmount)

fig,axs = plt.subplots(3)
for ind, l in enumerate(toTraceArn):
    t, = axs[0].plot(x,l,label = legendAxes[ind])
    lines0.append(t)
for ind,l in enumerate(toTraceLan):
    t, = axs[1].plot(x,l,label = legendAxes[ind])
    lines1.append(t)
for ind,l in enumerate(toTraceLob):
    t, = axs[2].plot(x,l,label = legendAxes[ind])
    lines2.append(t)
plt.subplots_adjust(left=0.3)
#checkButtons for similarity
rax = plt.axes([0.05,0.4,0.07,0.3])
labels = labelSimi
visibility = [True,True,True,True,True,True,True,True,True]
checkSimi = CheckButtons(rax,labels,visibility)


def funcSimi(label):
    index = labelSimi.index(label)
    if(index > 2 and index <= 5):
        for i in range(index%3,len(toTraceArn),3):
            lines1[i].set_visible(not lines1[i].get_visible())
    elif(index > 5 ):
        for i in range(index%3,len(toTraceArn),3):
            lines2[i].set_visible(not lines2[i].get_visible())
    else:
        for i in range(index%3,len(toTraceArn),3):
            lines0[i].set_visible(not lines0[i].get_visible())
    
    plt.draw()



checkSimi.on_clicked(funcSimi)
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()
"""