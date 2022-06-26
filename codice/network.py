# %load network.py
"""Rete neurale feedforward con apprendimento per discesa del gradiente stocastica"""

#### Librerie
#numeri casuali
import random
#calcoli matriciali efficienti
import numpy as np
#interazioni con utente
import sys
#caricare dati
import pickle
#convertire dati in binario
import conversioni as cv
#salvareintxt
import salvatxt as svtxt
#moltiplicazioni tra matrici iterativa
import matrixmul

#classe network
class Network(object):
    #inizializzazione della rete
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        #metto sizes in una lista
        sizes = list(sizes)
        #con la lunghezza di sizes capisco il numero di strati
        self.num_layers = len(sizes)
        self.sizes = sizes
        #inizializzo in maniera random pesi e bias in base alle dimensioni date da sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]     
    #feedforward tradizionale fatta con np.dot efficiente
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    #FUNZIONE CHE INIETTA I BIT FLIP
    def do_err_w(self,m_block,rl0,rl1,rl2, err_range):
        #controllo in quale blocco voglio eseguire l'errore
        #se è il primo
        if m_block == 0:
            #il primo range vale per lo strato di ingresso
            range1 = rl0
            #il secondo range vale per lo strato nascosto
            range2 = rl1
            #indico che l'errore va inserito nella prima iterazione
            it = 0
        #se è il secondo
        elif m_block == 1:
            #il primo range vale per lo strato nascosto
            range1 = rl1
            #il secondo range vale per lo strato di uscita
            range2 = rl2
            #indico che l'errore va inserito nella seconda iterazione
            it = 1
        #con i conto le iterazioni
        i=0
        for b,w in zip(self.biases,self.weights):
            #se la iterazione è quella che voglio
            if i == it:
                #allora per tutti i k della range1
                for k in range(range1[0],range1[1],range1[2]):
                    #per tutti gli s della range2
                    for s in range(range2[0],range2[1],range2[2]):
                        #prendo il valore definito da k,s nella matrice dei pesi della iterazione corrente
                        peso = (w[s,k])
                        #converto in binario il valore in una nuova variabile
                        w_err = cv.float_to_bin(w[s,k])
                        #scrivo a terminale il peso originale in bit
                        print("\nPeso ORIGINALE in BIT:")
                        print(w_err)
                        #trasformo in lista la stringa di bit
                        w_err = list(w_err)
                        #mi muovo all'interno della lista come detto dalla err_range
                        for j in range(err_range[0],err_range[1],err_range[2]):
                            #eseguo il bit flip, quindi inverto i valori se 1 => 0 se 0 => 1
                            if w_err[j]=='1':
                                w_err[j]=0
                            else:
                                w_err[j]=1
                        #ritrasformo w_err in una stringa
                        w_err = ''.join(str(e) for e in w_err)
                        #scrivo a terminale il valore modificato in bit
                        print("Peso MODIFICATO in BIT:")
                        print(w_err)
                        #scrivo a terminale il valore originale in decimale
                        print("\nPeso ORIGINALE in DEC:")
                        print(peso)
                        #riconverto il valore modificato da binario a float inserendolo nella matrice dei pesi
                        w[s,k] = cv.bin_to_float(w_err)
                        #e lo scrivo a terminale il valore modificato in decimale
                        print("Peso MODIFICATO in DEC:")
                        print(w[s,k])
            #aumento l'iterazione
            i+=1     
        return
    #DISCESA STOCASTICA DEL GRADIENTE È QUELLA ORIGINALE TRANNE PER IL SALVATAGGIO
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        #crea lista con i dati di allenamento e conta quanti siano
        training_data = list(training_data)
        n = len(training_data)
        #se sono passato anche i dati di test creo una lista e conto quanti siano
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        #ciclo per le epoche
        for j in range(epochs):
            #faccio uno shuffle quindi mescolo i dati di allenamento
            random.shuffle(training_data)
            #creo una lista contente mini lotto di ingressi di grandezza mini_batch_sizes
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            #per ogni mini lotto 
            for mini_batch in mini_batches:
                #chiamo la funzione per aggiornare i valori della rete con tasso di apprendimento eta
                self.update_mini_batch(mini_batch, eta)
            #se ho passato i dati di test, quindi voglio testare le performance
            if test_data:
                #scrivo il numero della epoca e quanti numeri la rete ha indovinato sul test di dati
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                #se non ho passato i dati di test dico solo di aver allenato la rete per la epoca corrente
                print("Epoch {} complete".format(j))
            #SALVATAGGIO ALLENAMENTO
            #se sono alla fine dell'allenamento
            if j == epochs-1:
                #chiedo a terminale se si vuole salvare l'allenamento
                answer = input("Vuoi salvare l'allenamento? s/n.\n")
                #se si faccio salvo
                if answer == 's' :
                    #salvo con pickle
                    file = open('save','wb')
                    backup = np.array([self.weights,self.biases])
                    pickle.dump(backup,file)
                    file.close()
                    #salvo in formato txt invocando la mia funzione per il salvataggio in txt
                    svtxt.salvaw(self.weights)
                    svtxt.salvab(self.biases)
                else:
                    pass
    #FUNZIONE ORIGINALE CHE ESEGUE L'ALLENAMENTO AGGIORNANDO I VALORI DELLA RETE            
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        #creo un oggetto con degli 0, con la stessa struttura dell'oggetto contenente i bias
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #creo un oggetto con degli 0, con la stessa struttura dell'oggetto contenente i pesi
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #per x e y che rappresentano rispettivamente l'ingresso alla rete e l'uscita voluta del set di allenamento
        for x, y in mini_batch:
            #calcolo l'errore attuale nei bias e pesi attraverso la backprop
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #sommo gli errori per i bias all'oggetto nabla_b elemento per elemento, somma 0 più il calcolo
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            #sommo gli errori per i pesi all'oggetto nabla_w elemento per elemento, somma 0 più il calcolo
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #vado ad aggiornare i pesi con la regola vista nella teoria
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        #vado ad aggiornare i bias con la regola vista nella teoria
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    #BACKPROP ORIGINALE GIÀ COMMENTATA DA AUTORE ORIGINALE IN INGLESE, NON COMMENTO NEL CODICE MA QUI
    #FA LE COSE DELLA TEORIA, PRENDE INGRESSO CALCOLA LE VARIE ATTIVAZIONI E SOMME PESATE,
    #CALCOLA L'ERRORE ALLO STRATO DI USCITA
    #POI LO CALCOLA PER GLI STRATI PRECEDENTI TRANNE QUELLO DI INGRESSO (CHE NON È UN VERO STRATO DI NEURONI)
    #RITORNA LE MATRICI CON GLI ERRORI DEI BIAS E DEI PESI PER STRATO
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    #FUNZIONE CHE VALUTA QUANTI NUMERI LA RETE RIESCE A RICONOSCERE
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    #FUNZIONE CHE RITORNA LA DERIVATA DELLA FUNZIONE DI COSTO RISPETTO ALLE ATTIVAZIONI
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

              		
#### Miscellaneous functions
#FUNZIONE SIGMOIDE
def sigmoid(z):
    """The sigmoid function."""
    return (1.0/(1.0+np.exp(-z)))
#DERIVATA DELLA FUNZIONE SIGMOIDE
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
