#funzioni per salvare in txt sperimentale
import numpy as np
#SALVA PESI
def salvaw(weights):
    #i per contare gli indici di matrici
    i = 0
    #apro file con nome
    with open('weights.txt', 'w') as f:
        #faccio stare weights che è una lista in un array di numpy
        weights = np.array(weights)
        #per elemento di weight
        for parte in weights:
            #scrivo un header per formattazione
            f.write('# MATRICE IN INDICE %d\n' %i)
            #scrivo la matrice
            np.savetxt(f, parte)
            #aumento indice
            i+=1
    return

#SALVA BIAS UGUALE MA CON NOME FILE E VARIABILE DIVERSO
def salvab(biases):
    i = 0
    with open('biases.txt', 'w') as f:
        biases = np.array(biases)
        for parte in biases:
            f.write('# MATRICE IN INDICE %d\n' %i)
            np.savetxt(f, parte)
            i+=1
    return

