import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

def readListFile(fileName):
    output = []
    with open(fileName+'.txt','r') as f:
        content = f.read()
        mat = content.split('\n')[:-1]
        for entry in mat:
            unitary = entry.split(',')
            unitary = [float(x) for x in unitary]
            output.append(unitary)
    return(np.array(output))

Arn = readListFile("Arn")
Lan = readListFile("Lan")
Lob = readListFile("Lob")

legendAxes = ["Un","Un","Un","Lsym","Lsym","Lsym","Lrw","Lrw","Lrw"]








def traceFull():
    x = range(5,1000)
    fig,axs = plt.subplots(3)
    for ind, l in enumerate(Arn):
        t, = axs[0].plot(x,l,label = legendAxes[ind])
    for ind,l in enumerate(Lan):
        t, = axs[1].plot(x,l,label = legendAxes[ind])
    for ind,l in enumerate(Lob):
        t, = axs[2].plot(x,l,label = legendAxes[ind])
    plt.subplots_adjust(right=0.7)

    axs[0].set_ylim([0,20])
    axs[1].set_ylim([0,20])
    axs[2].set_ylim([0,20])

    axs[0].legend(loc = 'center right', bbox_to_anchor=(1.3,-0.5))
    #axs[1].legend()
    #axs[2].legend()
    plt.show()


def traceDetailled(start):
    x = range(start,1000)
    ArnDet = Arn[:,start-5:]
    LanDet = Lan[:,start-5:]
    LobDet = Lob[:,start-5:]
    fig,axs = plt.subplots(3)
    for ind, l in enumerate(ArnDet):
        t, = axs[0].plot(x,l,label = legendAxes[ind])
    for ind,l in enumerate(LanDet):
        t, = axs[1].plot(x,l,label = legendAxes[ind])
    for ind,l in enumerate(LobDet):
        t, = axs[2].plot(x,l,label = legendAxes[ind])
    plt.subplots_adjust(right=0.7)

    ystart = 7.5
    yend = 16
    axs[0].set_ylim([ystart,yend])
    axs[1].set_ylim([ystart,yend])
    axs[2].set_ylim([ystart,yend])

    axs[0].legend(loc = 'center right', bbox_to_anchor=(1.3,-0.5))
    plt.show()


def traceSelected(select,start):
    x = range(start,1000)
    ArnDet = Arn[:,start-5:]
    LanDet = Lan[:,start-5:]
    LobDet = Lob[:,start-5:]
    fig,axs = plt.subplots(3)
    ArnList = []
    LanList = []
    LobList = []

    for ind, l in enumerate(ArnDet):
        t, = axs[0].plot(x,l,label = legendAxes[ind])
        ArnList.append(t)
    for ind,l in enumerate(LanDet):
        t, = axs[1].plot(x,l,label = legendAxes[ind])
        LanList.append(t)
    for ind,l in enumerate(LobDet):
        t, = axs[2].plot(x,l,label = legendAxes[ind])
        LobList.append(t)
    plt.subplots_adjust(right=0.7)

    ystart = 7.5
    yend = 11
    axs[0].set_ylim([ystart,yend])
    axs[1].set_ylim([ystart,yend])
    axs[2].set_ylim([ystart,yend])

    axs[0].legend(loc = 'center right', bbox_to_anchor=(1.3,-0.5))
    if(select == 'Eps'):
        for i in range(1,len(ArnList),3):
            ArnList[i].set_visible(not ArnList[i].get_visible())
            LanList[i].set_visible(not LanList[i].get_visible())
            LobList[i].set_visible(not LobList[i].get_visible())
        for i in range(2,len(ArnList),3):
            ArnList[i].set_visible(not ArnList[i].get_visible())
            LanList[i].set_visible(not LanList[i].get_visible())
            LobList[i].set_visible(not LobList[i].get_visible())
    elif(select == 'Knn'):
        for i in range(0,len(ArnList),3):
            ArnList[i].set_visible(not ArnList[i].get_visible())
            LanList[i].set_visible(not LanList[i].get_visible())
            LobList[i].set_visible(not LobList[i].get_visible())
        for i in range(2,len(ArnList),3):
            ArnList[i].set_visible(not ArnList[i].get_visible())
            LanList[i].set_visible(not LanList[i].get_visible())
            LobList[i].set_visible(not LobList[i].get_visible())
    else:
        for i in range(0,len(ArnList),3):
            ArnList[i].set_visible(not ArnList[i].get_visible())
            LanList[i].set_visible(not LanList[i].get_visible())
            LobList[i].set_visible(not LobList[i].get_visible())
        for i in range(1,len(ArnList),3):
            ArnList[i].set_visible(not ArnList[i].get_visible())
            LanList[i].set_visible(not LanList[i].get_visible())
            LobList[i].set_visible(not LobList[i].get_visible())

    plt.show()


def find2Best(alg):
    scoreFirst = np.zeros(6)
    scoreSecond = np.zeros(6)
    #go through all the columns
    for i in range(0,alg.shape[1]):
        selected = alg[3:,i]
        
        m = np.argmin(selected)
        scoreFirst[m] += 1
        newSelected = [x for i,x in enumerate(selected) if i!=m]
        msv = min(newSelected)
        ms = np.argwhere(selected == msv)
        scoreSecond[ms] += 1
    return(scoreFirst,scoreSecond)

#//epsUn //knnUn //fulUn epsLsym knnLsym fulLsym epsLrw knnLrw fulLrw
"""
print("Arn")
print(find2Best(Arn))
print("Lan")
print(find2Best(Lan))
print("Lob")
print(find2Best(Lob))
"""


"""
Best : 
Arn : 1- epsilon Lrw
      2- epsilon Lsym
Lan : 1- epsilon Lrw
      2- epsilon Lsym
Lob : 1- epsilon Lsym
      2- knn Lsym
"""

def traceBest(start):
    ArnFirst = Arn[6,start:]
    ArnSecond = Arn[3,start:]
    LanFirst = Lan[6,start:]
    LanSecond = Lan[3,start:]
    LobFirst = Lob[3,start:]
    LobSecond = Lob[4,start:]

    Labels = ["1er Arnoldi : Epsilon Lrw","2eme Arnoldi : Epsilon Lsym","1er Lanczos : Epsilon Lrw","2eme Lanczos : Epsilon Lsym","1er LOBPCG : Epsilon Lsym","2eme LOBPCG : Knn Lsym"]
    toPlot = [ArnFirst,ArnSecond,LanFirst,LanSecond,LobFirst,LobSecond]
    x = range(start+5,1000)
    for index,value in enumerate(toPlot):
        plt.plot(x,value, label = Labels[index])
    plt.legend()
    plt.show()

traceBest(960)

