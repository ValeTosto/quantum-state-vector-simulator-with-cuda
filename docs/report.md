# **Simulatore Quantistico State Vector in CUDA**
### Relazione progetto GPGPU - Valerio Tosto

## Abstract
Lo scopo della seguente relazione e' quello di documentare lo sviluppo di un simulatore quantistico State Vector
realizzato con l'idea di sfruttare una GPU per la parallelizzazione del codice, puntando alle migliori performance possibili.
Il simulatore e' stato sviluppato in CUDA C++, con hardware di riferimento NVIDIA GTX 1080.

## **Introduzione al Quantum Computing**

Il Quantum Computing e' una branca dell'informatica in rapida crescita, basata sull'idea di sfruttare la **meccanica quantistica** per ottenere vantaggi computazionali.  
Vi sono due principi fisici che caratterizzano la meccanica quantistica:  
- Il **principio di sovrapposizione**, per cui una particella quantistica puo' trovarsi in piu' stati fisici in contemporanea, ognuno con una certa probabilita';
- L'entanglement quantistico, ovvero la capacita' di due particelle quantistiche di trovarsi in correlazione, per cui un'informazione sullo stato di una automaticamente ci informa anche riguardo lo stato dell'altra.

Se la computazione classica lavora con il bit, ovvero informazioni di natura binaria, il QC lavora con l'informazione quantistica: il qubit.
Un qubit $\ket\psi$ e' un'unita' di informazione che vale `0` o `1`, ognuno con una certa ampiezza. Per eseguire operazioni sui qubit, servono degli operatori (unari, binari, ternari, ...) che ne alterino le ampiezze. Infine, su un qubit si puo' effettuare una **misurazione**, ovvero un'operazione che ne rompe la sovrapposizione e lo costringe a collassare in 0 o in 1, secondo le rispettive ampiezze.
La meccanica quantistica viene modellata perfettamente tramite l'algebra lineare:  
- Un qubit e' un vettore unitario di ampiezze complesse, combinazione lineare dei suoi possibili stati: $$\ket\psi = \alpha\ket0 + \beta\ket1$$, con $\alpha,\beta\in\mathbb{C}$. Un registro quantistico e' anch'esso un vettore, ottenibile calcolando il prodotto vettoriale dei singoli qubit che contiene: $n$ qubit vengono rappresentati tramite un vettore di dimensione $N=2^n$.
- Un operatore logico quantistico e' una matrice unitaria $U$. Un operatore $n$-ario e' una matrice quadrata di dimensioni $2^n\times2^n$. L'applicazione di un operatore sui rispettivi qubit viene calcolata tramite un semplice prodotto riga-colonna;
- La misurazione e' una proiezione su una base: in computazione quantistica essa corrisponde alla base computazionale { $\ket0, \ket1$}.

## **Il modello State Vector e il problema esponenziale**  

L'approccio *State Vector* per la simulazione quantistica consiste nel mantenere in memoria l'intero vettore di stato e aggiornarlo passo dopo passo applicando i vari gate quantistici del circuito.

Questo approccio si scontra rapidamente con i limiti hardware dei calcolatori classici. Poiché un registro di $n$ qubit richiede un vettore di $2^n$ ampiezze complesse, la memoria e il tempo di calcolo crescono in modo esponenziale. 
Utilizzando numeri in virgola mobile (4 byte per la parte reale e 4 byte per quella immaginaria):
- Un registro di **10 qubit** richiede $2^{10} \times 8$ B $\approx 8$ KB.
- Un registro di **20 qubit** richiede $2^{20} \times 8$ B $\approx 8$ MB.
- Un registro di **30 qubit** richiede $2^{30} \times 8$ B $\approx 8$ GB.

Oltre i 30 qubit, la simulazione State Vector diventa proibitiva per i normali calcolatori. La natura dell'approccio, che prevede la manipolazione di un unico pesante vettore, si dimostra subito adeguata all'accelerazione parallela tramite GPU.

# **2. Implementazione e Architettura CUDA**

## Rappresentazione dei Dati in Memoria
Per mappare il problema sull'architettura NVIDIA Pascal (GTX 1080), il vettore di stato è stato... *(qui spieghi come hai gestito i numeri complessi e l'allocazione)*

