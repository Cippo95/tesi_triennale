#MOLTIPLICAZIONE TRA MATRICI ITERATIVA, MOLTO PIÙ LENTO CHE PRODOTTO MATRICIALE NUMPY
#a e b sono matrici
def matrixmul(a,b):
    #righe e colonne matrice 1
    r1,c1=a.shape
    #righe e colonne matrice 2
    r2,c2=b.shape
    #se le colonne della prima matrice sono diverse dalle righe della seconda non si può fare prodotto tra matrici
    if c1!=r2:
        print('Le colonne della prima matrice non sono equivalenti alle righe della seconda matrice!') 
    else:
        #se tutto ok creo l3 che conterrà il risultato
        l3=[]
        #vado a selezionare la riga della prima matrice
        for k in range (r1):
            #l2 conterrà le righe che formeranno il risultato
            l2=[]
            #vado a selezionare la colonna della seconda matrice
            for j in range (c2):
                #l1 conterrà i valori temporanei dei riga per colonna
                l1=0
                for i in range (c1):
                    #per ogni elemento della riga di a lo moltiplico per il rispettivo elemento su b e sommo i risultati tra loro.
                    l1+=a[k,i]*b[i,j]
                    #ESEMPIO DI ERRORE:
                    #if k==0 and i==0:  #caso di errore su prodotto del primo ingresso con primo neurone
                        #l1=10          #viene effettuato anche con primo neurone del secondo layer e primo dell'uscita,
                                        #comunque basterebbe qualche condizione in più per risolvere                
                l2.append(l1)           #l2 tiene conto delle righe risultanti
            l3.append(l2)               #l3 mette tutte le righe insieme una dietro l'altra formando il risultato
        print(l3) #Questo fa vedere tutti i calcoli che vengono eseguiti a riga di comando, è un debug per sapere che il codice funziona
    return l3 #ritorno la matrice risultate
