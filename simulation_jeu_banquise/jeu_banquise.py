
import random
import sys
def main(argv):

    nsim=100000

    nVic=0
    for i in range(nsim):
        nVic+=simule()

    percentWin=nVic/nsim*100

    print("pourcentage de victoire : "+str(percentWin)+"%")


def simule():   
    nbBanquise=4
    nbVies=7

    nbPont=0
    nbIgloo=0

    while nbVies>0:
        val = random.randint(1,6)

        #glace fond
        if(val<=2):
            nbVies -=1
        #pont    
        elif(val<=4 and nbBanquise>0):
            nbPont+=1
            nbBanquise-=1
        #igloo    
        elif(nbPont>0):
            nbIgloo+=1
            nbPont-=1
            if(nbIgloo == 4):
                #print("victoire")
                return 1
            
    #print("défaite")
    return 0

if __name__ == '__main__':
    main(sys.argv)  