# Eseguire il programma

Il file principale da eseguire è "test_cippo.py".

Esso contiene diverse opzioni: si può caricare l'ultimo allenamento salvato o 
eseguirne uno nuovo.

Successivamente il programma chiede se si vuole testare un errore temporaneo singolo nei pesi e se inserirlo in un blocco della rete o la rete intera. 
Chiederà per quante volte si vuole testare ed eseguirà questi test
scegliendo indici random per la funzione do_err_w (neurone colpito dall'errore).

Successivamente chiede se si vuole inserire un errore additivo statico, chiede quanti 
errori si vogliono iniettare, in che blocco e per quanti round di test. Anche qui verrano scelti
indici random e si userà la funzione do_err_w.

Verranno visualizzati a schermo informazioni utili alcuni calcoli statistici.

## Breve descrizione dei file:

- "conversioni.py" contiene il codice per le conversioni varie float-binary.

- "matrixmul.py" contiene il codice per i calcoli matriciali riga per colonna.

- "network.py" contiene l'inizializzazione della rete e le varie funzioni legate ad essa.

- "biases.txt" e "weights.txt" contengono il dump dei bias e dei pesi in un formato testuale. 

- "salvatxt.py" contiene le funzioni per salvarli.

- Il file "save" è il salvataggio dell'allenamento, cioè contiene anche lui pesi e biases.

- "mnist_loader.py" serve per caricare i dati MNIST contenuti in "mnsit.pkl.gz".

