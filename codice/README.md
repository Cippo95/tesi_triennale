# Eseguire il programma

Il file principale da eseguire � "test_cippo.py".

Esso contiene diverse opzioni: si pu� caricare l'ultimo allenamento salvato o 
eseguirne uno nuovo.

Successivamente il programma chiede se si vuole testare un errore temporaneo singolo nei pesi e se inserirlo in un blocco della rete o la rete intera. 
Chieder� per quante volte si vuole testare ed eseguir� questi test
scegliendo indici random per la funzione do_err_w (neurone colpito dall'errore).

Successivamente chiede se si vuole inserire un errore additivo statico, chiede quanti 
errori si vogliono iniettare, in che blocco e per quanti round di test. Anche qui verrano scelti
indici random e si user� la funzione do_err_w.

Verranno visualizzati a schermo informazioni utili alcuni calcoli statistici.

## Breve descrizione dei file:

- "conversioni.py" contiene il codice per le conversioni varie float-binary.

- "matrixmul.py" contiene il codice per i calcoli matriciali riga per colonna.

- "network.py" contiene l'inizializzazione della rete e le varie funzioni legate ad essa.

- "biases.txt" e "weights.txt" contengono il dump dei bias e dei pesi in un formato testuale. 

- "salvatxt" contiene le funzioni per salvarli.

- Il file "save" � il salvataggio dell'allenamento, cio� contiene anche lui pesi e biases.

- "mnist_loader.py" serve per caricare i dati MNIST contenuti in "mnsit.pkl.gz".

