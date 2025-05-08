import sys
import math
import random
import matplotlib.pyplot as plt

#donnees reeles
#

def main(argv):
    random.seed(42)

    #une seule simulation
    #simule(True)
    #exit(0)
    nbSimulation = 10000

    nbs = []
    avgs = []
    revenus = []
    for i in range(nbSimulation):

        (nb, avgAttente, revenu) = simule(verb=False)
        nbs.append(nb)
        avgs.append(avgAttente)
        revenus.append(revenu)
    
    #print(nbs)
    #print(avgs)
    #print(revenus)

    print("revenu moyen : "+str(sum(revenus)/len(revenus)))
    print("attente moyenne : "+str(sum(avgs)/len(avgs)))
    print("Nombre de clients moyen : "+str(sum(nbs)/len(nbs)))


def simule(verb):
    #12 h de travail

    attentes = []
    nb= 0
    maxTime=60*10
    time=0
    revenuTotal=0

    lavagesAttente=[]
    while(time<maxTime):
        #choisis un délais
        delai = rand_delai()

        time+= delai



        #simule la file d'attente jusqu'à time+delai
        d=delai
        
        while(d>0 and len(lavagesAttente)>0):
            if(lavagesAttente[0]<d):
                d-=lavagesAttente[0]
                lavagesAttente.pop(0)
            else:
               lavagesAttente[0]= lavagesAttente[0]-d     
               d=0
            if(d<=0):
                break   

        if(verb):
            print("voiture arrive à "+str(time))
            print("  file d'attente : "+str(lavagesAttente))

        #la voiture décide si elle joint le file d'attente
        #
        if(randRejointFile(len(lavagesAttente))):
            
            #calcule le temps d'attente
            #
            attente = sum(lavagesAttente)
            attentes.append(attente)
            nb+=1
            # - choisis le type de lavage
            # - ajoute la voiture à la file d'attente
            # 
            (lavage, revenu) = rand_lavage()
            if(verb):
                print("  la voiture choisis le lavage de "+str(lavage)+" minutes")

            revenuTotal+=revenu
            #print(" duree : "+str(lavage))
            lavagesAttente.append(lavage)       
        else:
            if(verb):
                print("  la voiture n'entre pas dans la file d'attente")

    
    avgAttente=sum(attentes)/len(attentes)    

    return (nb, avgAttente, revenuTotal)


        

        
#decide si une voiture rejoint le file.
# 100% si la file est <= 3 voitures
# 50% si la file est > 3 voitures       
# 25% si la file est > 5 voitures
# 5% si la file est > 10 voitures 
def randRejointFile(nbAttente):
    if(nbAttente<=3):
        return True
    
    passe= 0.5
    if(nbAttente>5):
        passe=0.75
    if(nbAttente>7):
        passe=0.9
    if(nbAttente>10):
        passe=0.95
    rand = random.random()
    if(rand >= passe):
        return True
    else:
        return False


def rand_delai():
    delai_moy = 10

    rand=random.random()
    return -delai_moy*math.log(1-rand)

def rand_lavage():
    lavage_prob= [0.35, 0.55, 0.85, 1]
    lavage_duree = [6.0, 10.0, 12.0, 15.0]
    revenus = [10, 12, 15, 18]

    #nombre aleatoire entre 0 et 1
    rand=random.random()
    for i in range(len(lavage_prob)):
        if(rand <= lavage_prob[i]):
            return (lavage_duree[i], revenus[i])

def rand_lavageV2():
    lavage_prob= [0.45, 0.55, 0.80, 1]
    lavage_duree = [6.0, 10.0, 12.0, 15.0]
    revenus = [8, 12, 15, 18]

    #nombre aleatoire entre 0 et 1
    rand=random.random()
    for i in range(len(lavage_prob)):
        if(rand <= lavage_prob[i]):
            return (lavage_duree[i], revenus[i])


    

if __name__ == '__main__':
    main(sys.argv)  