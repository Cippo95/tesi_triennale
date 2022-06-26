#importa interazione a terminale
import sys
#importa funzioni di mnist_loader.py
import mnist_loader
#importa funzioni di network.py
import network
#importa il modulo pickle
import pickle
#importa funzionalità NumPy
import numpy as np
#importa funzionalità random
import random

#carico dati del mnist
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#assegno test_data a una lista per mantenerlo durante l'esecuzione
test_data = list(test_data)
n_test = len(test_data)

#funzione test fa un feedforward e testa l'accuratezza della rete
def test(test_data = None):
    evaluated = net.evaluate(test_data)
    print("Giusti / Totale : {} / {}".format(evaluated,n_test));
    #perf ritorna le performance in percentuale
    perf = (float(net.evaluate(test_data))/n_test)*100;
    return perf
    
#sizes indica il numero di neuroni per strati della rete
sizes = [784, 30, 10]
net = network.Network(sizes)

#CARICAMENTO SALVATAGGIO E TEST PRESTAZIONI DEL SALVATAGGIO
caricare = input("Caricare ultimo salvataggio? s/n.\n")
if caricare == 's':
    with open("save",'rb') as f:
        backup = pickle.load(f)
    net.weights = backup[0]
    net.biases = backup[1]
    weightsback = net.weights
    #TEST DELLE PRESTAZIONI
    print("Test della rete:")
    #prc assume il valore percentuale della prestazione
    prc = test(test_data=test_data)
    print("Accuratezza del %.2f %%\n" % prc)
else:
    #SE NON SI CARICA SI VUOLE ALLENARE LA RETE
    allenamento = input("Allenare la rete? Attenzione: Ci può mettere diversi minuti. s/n.\n")
    if allenamento == 's':
        epoche = input("Inserire per quante epoche si vuole affrontare l'allenamento. Attenzione: Più epoche più tempo ci mette.\n")
        epoche = int(epoche)
        #se sono state inserite le epoche allora alleno
        if epoche:
            net.SGD(training_data, epoche, 10, 3.0, test_data=test_data)
            prc = test(test_data=test_data)
            print("Accuratezza del %.2f %%\n" % prc)
        else:
            #se è stato inserito 0 esco
            exit()
    else:
            exit()            
#ERRORI SINGOLI SUI PESI TRANSITORI, QUINDI RICARICA PESI ALLO STATO ORIGINALE ALLA FINE
errore = input("Vuoi testare la rete con errori singoli random sui pesi? s/n.\n")
if errore == 's':
    #dump su file di testo
    with open("dump_err_transitorio.txt",'w') as f:
        f.write("ERRORE TRANSITORIO\n")
    #inizializzazione di variabili varie
    value = 0.0
    var_vect =[]
    var_vect_sum = 0
    test_min = 100.0
    #RICHIESTA DEL BLOCCO SU CUI TESTARE
    tes=input("Vuoi testare il primo blocco, il secondo blocco o l'intera rete? Rispondere 1/2/3.\n")
    if tes == '1':
        #scrivo anche su file di testo info su che blocco metto gli errori
        with open("dump_err_transitorio.txt",'a') as f:
            f.write("SOLO BLOCCO 1\n")
        blocco=0
    elif tes == '2':
        with open("dump_err_transitorio.txt",'a') as f:
            f.write("SOLO BLOCCO 2\n")
        blocco=1
    elif tes == '3':
        with open("dump_err_transitorio.txt",'a') as f:
            f.write("INTERA RETE\n")
        #non assegno qui la variabile del blocco, anche se fatta random sarebbe poi statica per i cicli
    #RICHIESTA DEI ROUND DI TEST
    rg=input("Inserire per quanti round si vuole testare la rete.\n")
    #conversione da carattere a numero intero
    rg=int(rg)
    #ciclo che cicla i vari round
    for i in range(0,rg):
        #qui nel caso si voglia testare intera viene scelto a random il blocco, per ogni ciclo
        if tes == '3':
            blocco = random.randint(0,1)
        #scrivo sia a terminale che nel file di testo per avere una formattazione leggibile
        print("\nROUND %d --------------------------------------------------------------" %(i+1))
        with open("dump_err_transitorio.txt",'a') as f:
            f.write("\nROUND %d --------------------------------------------------------------" %(i+1))
        #assegno a random i vari indici per primo,secondo strato, indice del bit da flippare e indice terzo strato
        a=random.randint(0,783)
        b=random.randint(0,29)
        c=random.randint(0,63)
        d=random.randint(0,9)
        #eseguo l'errore singolo sui tali indici, notare che sfrutto intelligentemente la range per fare un solo errore
        net.do_err_w(blocco,[a,a+1,1],[b,b+1,1],[d,d+1,1],[c,c+1,1])
        #formato di net.do_err_w(che blocco, che pixel, che neurone centrale, che neurone centrale, che uscita, che bit)
        #se primo blocco scrivo a console informazioni blocco, del pixel e neurone nascosto scelti, essi identificano un peso
        if blocco==0 :
            print("\nBlocco 1 selezionato")
            print("Ho selezionato a random il pixel di ingresso %d" %a)
            print("Ho selezionato a random il neurone nascosto %d" %b)
        #se secondo blocco scrivo a console informazioni blocco, del neurone nascosto e dell'uscita scelti, essi identificano un peso
        if blocco==1 :
            print("\nBlocco 2 selezionato")
            print("Ho selezionato a random il neurone nascosto %d" %b)
            print("Ho selezionato a random il neurone di uscita %d" %d)
        #informo quale bit è stato invertito
        print("Ho invertito il valore al bit in indice %d" %c)
        #testo le performance della rete che vengono anche ritornate a schermo da test()
        tested = test(test_data=test_data)
        #scrivo le performance nel file di testo
        with open("dump_err_transitorio.txt",'a') as f:
            f.write("\nACCURATEZZA : %.2f %%\n" %tested)
        #confronto le performance per stabilire quella minima nel ciclo
        if tested<test_min :
            test_min = tested
        #controllo le performance che escono da più di un centesimo di quella standard e in caso le aggiungo in una lista
        if tested < (prc-0.01) or tested > (prc+0.01):
           var_vect.append(tested)
        #sommo tutte le performance per fare poi la media alla fine
        value = float(value + tested)
        #ricarico il valore iniziale dei pesi alla fine del ciclo così che prossimo errore sia davvero singolo
        with open("save",'rb') as f:
            backup = pickle.load(f)
        net.weights = backup[0]
    #eseguo la media
    valued = float(value)/rg
    #scrivo la media sia a schermo che su file di testo
    print("\nACCURATEZZA MEDIA NEL RANGE DI ERRORI TESTATO: %.2f %%" %valued)
    with open("dump_err_transitorio.txt",'a') as f:
            f.write("\nACCURATEZZA MEDIA NEL RANGE DI ERRORI TESTATO: %.2f %%" %valued)
    #stimo il numero di prestazioni che sono uscite di più di un centesimo da quella standard
    lung = len(var_vect)
    #controllo se ci siano elementi nella lista
    if lung != 0:
        #se ci sono sommo quindi tutti i valori per eseguirne poi la media
        for i in range(lung):
            var_vect_sum = var_vect_sum+var_vect[i]
        #eseguo la media
        var_vect_sum = float(var_vect_sum)/lung
        #scrivo a terminale e testo l'accuratezza media delle prestazioni fuori dal range
        print("\nACCURATEZZA MEDIA FUORI RANGE DEL 0.01%% DI %.2f %%: %.2f %%\n" %(prc,var_vect_sum))
        with open("dump_err_transitorio.txt",'a') as f:
            f.write("\nACCURATEZZA MEDIA FUORI RANGE DEL 0.01%% DI %.2f %%: %.2f %%\n" %(prc,var_vect_sum)) 
    else:
        #se non ci sono elementi nella lista scrivo sia a terminale che testo
        print("\nNESSUN RISULTATO FUORI RANGE DEL 0.01%\n")
        with open("dump_err_transitorio.txt",'a') as f:
            f.write("\nNESSUN RISULTATO FUORI RANGE DEL 0.01%\n")
    #scrivo il valore minimo di prestazione sia a terminale che testo
    print("VALORE MINIMO DI ACCURATEZZA: %.2f %%" %test_min)
    with open("dump_err_transitorio.txt",'a') as f:
            f.write("VALORE MINIMO DI ACCURATEZZA: %.2f %%" %test_min)
    
#ERRORI MULTIPLI RANDOM STATICI, NIENTE RICARICA DEI PESI   
errore1 = input("Vuoi testare la rete inserendo più errori random sui pesi? s/n.\n")
#se si vuole fare allora scrivo anche su file di testo
if errore1 == 's':
    with open("dump_piu_err.txt",'w') as f:
        f.write("ERRORI ADDITIVI\n")
    #inizializzo variabili varie
    value1 = 0.0
    vect = []
    vect_sum = 0.0
    test_min1 = 100.00
    #CHIEDO QUALE BLOCCO TESTARE
    tes1=input("Vuoi testare il primo blocco, il secondo blocco o l'intera rete? Rispondere 1/2/3.\n")
    #se primo blocco
    if tes1 == '1':
        #scrivo a file di testo info blocco
        with open("dump_piu_err.txt",'a') as f:
            f.write("SOLO BLOCCO 1\n")
        blocco=0
    #se secondo blocco
    elif tes1 == '2':
        #scrivo a file di testo info blocco
        with open("dump_piu_err.txt",'a') as f:
            f.write("SOLO BLOCCO 2\n")
        blocco=1
    #se intera rete
    elif tes1 == '3':
        #scrivo a file di testo intera rete
        with open("dump_piu_err.txt",'a') as f:
            f.write("INTERA RETE\n")
        #non inserisco qui blocco, anche se fatto random, sarebbe statico per i cicli successivi
    #RICHIEDO QUANTI ERRORI (CHE SI ACCUMULANO) SI VOGLIONO INSERIRE
    rg1=input("Inserire quanti errori si vuole inserire nella rete.\n")
    #converto da carattere a intero
    rg1=int(rg1)
    #scrivo su file di testo il numero di errori scelto
    with open("dump_piu_err.txt",'a') as f:
            f.write("INIEZIONE DI %d ERRORI" %(rg1))
    #RICHIEDO QUANTI ROUND DI TEST CON QUESTO ERRORE EFFETTUARE
    roundsi=input("Inserire per quanti round si vuole inserire %d errori nella rete.\n" %rg1)
    #converto da carattere a int
    roundsi=int(roundsi)
    #eseguo ciclo che cicla i round
    for r in range(0,roundsi):
        #scrivo a terminale e testo per avere una formattazione leggibile
        print("\nROUND %d --------------------------------------------------------------" %(r+1))
        with open("dump_piu_err.txt",'a') as f:
            f.write("\nROUND %d --------------------------------------------------------------" %(r+1))
        #ciclo il numero di errori da dover iniettare, li posso fare solo uno per volta con la mia funzione
        for i in range(0,rg1):
            print("\nINIEZIONE ERRORE %d ******************************************" %(i+1))
            #se testo intera rete assegno qui il valore random a blocco
            if tes1 == '3':
                blocco= random.randint(0,1)
            #assegno in maniera random gli indici del primo,secondo strato, del bit da flippare e del terzo strato
            a=random.randint(0,783)
            b=random.randint(0,29)
            c=random.randint(0,63)
            d=random.randint(0,9)
            #inietto l'errore desiderato
            net.do_err_w(blocco,[a,a+1,1],[b,b+1,1],[d,d+1,1],[c,c+1,1])
            #net.do_err_w(che blocco, che pixel, che neurone centrale, che neurone centrale, che uscita, che bit)
            #se sono nel primo blocco do informanzioni sul blocco e sui neuroni selezionati che indicano il peso selezionato
            if blocco==0 :
                print("\nBlocco 1 selezionato")
                print("Ho selezionato a random il pixel di ingresso %d" %a)
                print("Ho selezionato a random il neurone nascosto %d" %b)
            #se sono nel secondo blocco do informazioni sul blocco e sui neuroni selezionati che indicano il peso selezionato
            if blocco==1 :
                print("\nBlocco 2 selezionato")
                print("Ho selezionato a random il neurone nascosto %d" %b)
                print("Ho selezionato a random il neurone di uscita %d" %d)
            #informo a terminale quale bit è stato cambiato
            print("Ho invertito il valore al bit in indice %d" %c)
            #assegno a value1 la prestazione in questo ciclo, test inoltre mette a terminale le varie prestazioni
            value1 = test(test_data=test_data)
        #aggiungo la prestazione a un vettore che servirà per fare la media
        vect.append(value1)
        #infine scrivo a terminale e testo l'accuratezza finito il round, quindi iniettati tutti gli errori
        print("\nACCURATEZZA ROUND %d : %.2f %%" %((r+1), value1))
        with open("dump_piu_err.txt",'a') as f:
            f.write("\nACCURATEZZA ROUND %d : %.2f %%" %((r+1), value1))
        #tengo traccia della prestazione peggiore
        if value1 < test_min1:
            test_min1 = value1
        #ricarico i pesi ad ogni nuovo round di errori
        with open("save",'rb') as f:
            saver = pickle.load(f)
        net.weights = saver[0]
        #azzero la variabile value1 ad ogni round di errore
        value1 = 0.0
    #calcolo la lunghezza del vettore delle varie prestazioni
    lung1= len(vect)
    #vado a sommare le varie prestazioni così da farne poi una media
    for l in range(lung1):
        vect_sum = vect_sum+vect[l]
    #eseguo la media
    v_vect_sum = float(vect_sum)/lung1
    #scrivo a terminale e su testo le prestazioni medie
    print("\nACCURATEZZA MEDIA TOTALE %.2f %%" %(v_vect_sum))
    with open("dump_piu_err.txt",'a') as f:
        f.write("\nACCURATEZZA MEDIA TOTALE %.2f %%" %(v_vect_sum))
    #scrivo a terminale e su testo la prestazione minima
    print("VALORE MINIMO DI ACCURATEZZA: %.2f %%" %test_min1)
    with open("dump_piu_err.txt",'a') as f:
        f.write("\nVALORE MINIMO DI ACCURATEZZA: %.2f %%" %test_min1)   
else:
    #se non si vuole testare errori si esce
    exit()
