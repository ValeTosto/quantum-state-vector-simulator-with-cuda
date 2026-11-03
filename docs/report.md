# **Simulatore Quantistico State Vector in CUDA**
### Relazione progetto GPGPU - Valerio Tosto

## Abstract
Lo scopo della seguente relazione e' quello di documentare lo sviluppo di un simulatore quantistico State Vector
realizzato con l'idea di sfruttare una GPU per la parallelizzazione del codice, puntando alle migliori performance possibili.
Il simulatore e' stato sviluppato in CUDA C++, con hardware di riferimento NVIDIA GTX 1080.

# **Introduzione al Quantum Computing**

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

# **Implementazione**

## **Il modello State Vector e il problema esponenziale**  

L'approccio *State Vector* per la simulazione quantistica consiste nel mantenere in memoria l'intero vettore di stato e aggiornarlo passo dopo passo applicando i vari gate quantistici del circuito.

Questo approccio si scontra rapidamente con i limiti hardware dei calcolatori classici. Poiché un registro di $n$ qubit richiede un vettore di $2^n$ ampiezze complesse, la memoria e il tempo di calcolo crescono in modo esponenziale. 
Utilizzando numeri in virgola mobile (4 byte per la parte reale e 4 byte per quella immaginaria):
- Un registro di **10 qubit** richiede $2^{10} \times 8$ B $\approx 8$ KB.
- Un registro di **20 qubit** richiede $2^{20} \times 8$ B $\approx 8$ MB.
- Un registro di **30 qubit** richiede $2^{30} \times 8$ B $\approx 8$ GB.

Oltre i 30 qubit, la simulazione State Vector diventa proibitiva per i normali calcolatori. La natura dell'approccio, che prevede la manipolazione di un unico pesante vettore, si dimostra subito adeguata all'accelerazione parallela tramite GPU.

## **Flusso di esecuzione**

Il simulatore consiste in una pipeline sequenziale in cui l'host costruisce il circuito, mentre il device esegue soltanto i calcoli sul vettore di stato. Per evitare il bottleneck relativo all'uso del bus di sistema, il vettore viene allocato e modificato sempre e solo nella VRAM del device. Il flusso di esecuzione del programma e' il seguente:  
1. **Inizializzazione**: viene allocato il vettore di 2$^N$ numeri complessi (coppie di float), con lo stato di partenza $\ket0^{\otimes n}$;  
2. **Applicazione dei gate**: corrisponde alla costruzione del circuito. La CPU chiama i kernel relativi all'applicazione di un gate specifico, mentre il device aggiorna in VRAM il vettore di stato opportunamente. L'idea e' quella di introdurre la possibilita' di implementare un set di gate universale, composto da (X, CNOT, Toffoli);
3. **Misurazione**: A questo punto i qubit del vettore di stato possono essere misurati e il risultato trasferito su host.

In aggiunta, come metodo di verifica della correttezza dei calcoli, in qualunque momento sara' possibile calcolare la norma del vettore di stato e verificare che sia costantemente unitaria.

# **Analisi dei kernel**

## **Inizializzazione**
Per inizializzare il vettore di stato, l'host chiama la funzione `cudaMalloc` e alloca $2^{QBITS}$ coppie di float in VRAM. La certezza assoluta che la grandezza della griglia di lancio sia una potenza di 2 non va sottovalutata: sarebbe tecnicamente possibile evitare la presenza di controlli relativi all'accesso out of bounds all'interno dei kernel.  
Il vettore di partenza $\ket0^{\otimes N}$ corrisponde al vettore di lunghezza $N$, dove ogni componente vale 0 tranne la prima che vale 1. Per l'implementazione sono stati valutati due approcci:  
1. Approccio a singolo Kernel: viene lasciato un unico kernel, dove ogni work item imposta a 0 l'elemento corrispondente ad eccezione per quello in cui `global_id==0`, che lo imposta ad 1;
2. Approccio tramite funzioni native: Si imposta l'intero vettore a 0 tramite la funzione nativa `cudaMemset`, ottimizzata per lo specifico hardware, seguita da un `cudaMemCpy` che imposta il primo elemento a 1.

### **Analisi delle performance**
1. Versione a singolo kernel su :

| Funzione | Tempo medio | Larghezza di Banda | Throughput |
| :--- | :--- | :--- | :--- |
| **Singolo Kernel** | 8.974 ms | 239.3 GB/s | 29.91 GE/s |
| **Funzioni Native** | 8.924 ms |  240.6 GB/s | 30.08 GE/s |

In conclusione, le due implementazioni portano a prestazioni identiche, per cui verra' mantenuta la seconda implementazione.  

## **Applicazione di gate unari**

L'operazione fondamentale del simulatore è l'applicazione di un gate unario su un singolo qubit bersaglio $k$. Questa operazione modifica l'intero vettore di stato, e il valore di ogni ampiezza e' indipendente dalle altre: si tratta di un'operazione imbarazzantemente parallelizzabile.

Dato che il gate unario lavora su un singolo qubit, il singolo work item lavora su una coppia di ampiezze di indice $p_0$ e $p_1$, identici a meno del $k$-esimo bit. 
Il kernel lancera' quindi $N/2$ work item per la parallelizzazione del calcolo.

Per passare dall'id globale alla coppia di id, la logica e' quella di "spezzare" in due l'id globale in corrispondenza del $k$-esimo bit e inserire tra le due meta' il valore 0 per $p_0$ e 1 per $p_1$. Questi calcoli sono stati implementati tramite logica bitwise per massimizzare le prestazioni.

Calcolati i due indici, bastera' trattare `state[p_0]` e `state[p_1]` come ampiezze del qubit da aggiornare, applicarvi il gate tramite prodotto riga-colonna, e infine aggiornare il vettore di stato con i due nuovi valori calcolati.

### Analisi delle Prestazioni

Il kernel `apply_u_gate` richiede la lettura e la scrittura dell'intero vettore di stato per ogni applicazione di un gate, movimentando un totale di `sizeof(complex)*N` byte, ovvero $16 \times N$ bytes.
