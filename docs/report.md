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
La VRAM da 8GB della GTX 1080 ammette un limite massimo di 29 qubit.  

## **Flusso di esecuzione**

Il simulatore consiste in una pipeline sequenziale in cui l'host costruisce il circuito, mentre il device esegue soltanto i calcoli sul vettore di stato. Per evitare il bottleneck relativo all'uso del bus di sistema, il vettore viene allocato e modificato sempre e solo nella VRAM del device. Il flusso di esecuzione del programma e' il seguente:  
1. **Inizializzazione**: viene allocato il vettore di 2$^N$ numeri complessi (coppie di float), con lo stato di partenza $\ket0^{\otimes n}$;  
2. **Applicazione dei gate**: corrisponde alla costruzione del circuito. La CPU chiama i kernel relativi all'applicazione di un gate specifico, mentre il device aggiorna in VRAM il vettore di stato opportunamente. L'idea e' quella di introdurre la possibilita' di implementare un set di gate universale, composto da (H, X, CNOT, Toffoli);
3. **Misurazione**: A questo punto i qubit del vettore di stato possono essere misurati e il risultato trasferito su host.

In aggiunta, come metodo di verifica della correttezza dei calcoli, in qualunque momento sara' possibile calcolare la norma del vettore di stato e verificare che sia costantemente unitaria.

# **Analisi dei kernel**

# **Inizializzazione**
Per inizializzare il vettore di stato, l'host chiama la funzione `cudaMalloc` e alloca $2^{QBITS}$ coppie di float in VRAM. La certezza assoluta che la grandezza della griglia di lancio sia una potenza di 2 non va sottovalutata: sarebbe tecnicamente possibile evitare la presenza di controlli relativi all'accesso out of bounds all'interno dei kernel.  
Il vettore di partenza $\ket0^{\otimes N}$ corrisponde al vettore di lunghezza $N$, dove ogni componente vale 0 tranne la prima che vale 1. Per l'implementazione sono stati valutati due approcci:  
1. Approccio a singolo Kernel: viene lasciato un unico kernel, dove ogni work item imposta a 0 l'elemento corrispondente ad eccezione per quello in cui `global_id==0`, che lo imposta ad 1;
2. Approccio tramite funzioni native: Si imposta l'intero vettore a 0 tramite la funzione nativa `cudaMemset`, ottimizzata per lo specifico hardware, seguita da un `cudaMemCpy` che imposta il primo elemento a 1.

### **Analisi delle performance**

-$LWS = 256$  
-$QBITS = 29$

| Funzione | Tempo | Bandwidth | Throughput |
| :--- | :--- | :--- | :--- |
| **Singolo Kernel** | 8.974 ms | 239.3 GB/s | 29.91 GE/s |
| **Funzioni Native** | 8.924 ms |  240.6 GB/s | 30.08 GE/s |

In conclusione, le due implementazioni portano a prestazioni identiche, per cui verra' mantenuta la seconda implementazione.  

# **Applicazione di gate unari**

L'operazione fondamentale del simulatore è l'applicazione di un gate unario su un singolo qubit bersaglio $k$. Questa operazione modifica l'intero vettore di stato, e il valore di ogni ampiezza e' indipendente dalle altre: si tratta di un'operazione imbarazzantemente parallelizzabile. Il kernel permettera' l'implementazione dei gate *Hadamard* ($H$) e *Pauli-x* ($X$).

Dato che il gate unario lavora su un singolo qubit, il singolo work item lavora su una coppia di ampiezze di indice $p_0$ e $p_1$, identici a meno del $k$-esimo bit. 
Il kernel lancera' quindi $N/2$ work item per la parallelizzazione del calcolo.

Per passare dall'id globale alla coppia di id, la logica e' quella di "spezzare" in due l'id globale in corrispondenza del $k$-esimo bit e inserire tra le due meta' il valore 0 per $p_0$ e 1 per $p_1$. Questo approccio e' particolarmente efficace perche' privo di branch condizionali e perche' permette la perfetta lettura dei dati dalla memoria in coalescenza.  

Calcolati i due indici, bastera' trattare `state[p_0]` e `state[p_1]` come ampiezze del qubit da aggiornare, applicarvi il gate tramite prodotto riga-colonna, e infine aggiornare il vettore di stato con i due nuovi valori calcolati.

### **Analisi delle Prestazioni**

Il kernel `apply_u_gate` richiede la lettura e la scrittura dell'intero vettore di stato per ogni applicazione, movimentando un totale di `2 * sizeof(complex) * N` byte, ovvero $16 \times N$ byte.  
In linea teorica, un incremento del target qubit $k$ aumenta esponenzialmente la distanza in memoria tra gli indici $p_0$ e $p_1$. Per testare il kernel sono state studiate quindi le performance al variare di $k$:

| $k$ | Tempo | Bandwidth | Throughput |
| :--- | :--- | :--- | :--- |
| 0 | 38.08 ms | 227.6 GB/s | 28.44 GE/s |
| 5 | 37.56 ms |  226.2 GB/s | 28.59 GE/s |
| 10 | 37.97 ms |  228.7 GB/s | 28.28 GE/s |
| 15 | 38.08 ms |  225.6 GB/s | 28.2 GE/s |
| 20 | 38.53 ms |  222.9 GB/s | 27.87 GE/s |
| 25 | 38.45 ms |  223.4 GB/s | 27.92 GE/s |
| 28 | 38.93 ms |  220.6 GB/s | 27.58 GE/s |

E' possibile notare un lieve calo di banda tra il caso $k=0$ e il caso $k=28$, fenomeno riconducibile al fatto che con $k=28$ i due indici $p_0$ e $p_1$ sono distanti 2GB in VRAM. Tuttavia le performance generali rimangono costanti e questo comportamento risulta trascurabile.  
## **Applicazione di gate binari**

Il CNOT (o NOT condizionato) e' un gate fondamentale per il quantum computing: inverte un qubit target se e solo se un secondo qubit di controllo assume valore 1. Questo gate permette di ottenere l'entanglement, proprieta' che genera la maggior parte della potenza del calcolo quantistico.  

L'implementazione delle porte logiche binarie e' basata sulla stessa logica sviluppata per i gate unari. Un gate binario agisce su una coppia di qubit, definiti `target_qubit` e `control_qubit`, modificandone le ampiezze attraverso una matrice unitaria $4 \times 4$.  
La sfida algoritmica principale consiste nell'estendere la logica di indicizzazione per ottenere non piu' due ma quattro indici da modificare. Il kernel lancia quindi $N/4$ work item che modificano quartetti di dati.  
Analogamente al caso unario, l'id globale viene modificato con l'aggiunta di bit nelle posizioni opportune, quindi due bit in posizioni `control_qubit` e `target_qubit`: per ottenere i quattro indici basta inserire tutte le configurazioni dei 2 bit.  

Calcolati i quattro indirizzi, il work item legge le relative ampiezze, moltiplica il vettore ottenuto con la matrice $4x4$ che rappresenta il gate, e sovrascrive i risultati. 

### **Analisi delle Prestazioni**

Anche in questo caso vengono mossi `2 * sizeof(complex) * N` Byte di dati.  
L'aspetto piu' interessante da analizzare sono ancora una volta i valori `target_qubit` e `control_qubit`, che se alti potrebbero generare un overhead significativo. 

| Coppia control-target | Tempo | Bandwidth | Throughput |
| :--- | :--- | :--- | :--- |
| 0-1 | 38.54 ms | 222.9 GB/s | 27.86 GE/s |
| 0-15 | 38.7 ms | 221.9 GB/s | 27.74 GE/s |
| 0-28 | 39.14 ms |  219.5 GB/s | 27.43 GE/s |
| 20-21 | 39.2 ms |  219.2 GB/s | 27.39 GE/s |
| 27-28 | 38.82 ms | 221.3 GB/s | 27.66 GE/s |

Le performance restano generalmente costanti.  

# **Applicazione di gate ternari**  

L'ultima tipologia di gate applicabile implementata nel simulatore e' quella dei gate ternari. Principalmente questo permette di utilizzare il *Toffoli Gate* (o CCNOT) ovvero un NOT condizionato con doppio controllo.

Anche questa volta la logica rimane la stessa, con l'aggiunta di un terzo bit e il lancio di $N/8$ work item che processano ottetti di ampiezze all'unisono.

### **Analisi delle Prestazioni**

I byte di dati movimentati dal kernel sono sempre $16\times N$, e le potenziali criticita' rimangono le stesse.

| Tripla control_1-control_2-target | Tempo | Bandwidth | Throughput |
| :--- | :--- | :--- | :--- |
| 0-1-2 |  43.66 ms | 196.7 GB/s | 24.59 GE/s|
| 0-1-28 |  43.11 ms | 199.3 GB/s  | 24.91 GE/s |
| 0-27-28 | 41.09 ms | 209.1 GB/s | 26.13 GE/s |
| 20-21-22 | 39.44 ms | 217.8 GB/s | 27.22 GE/s |
| 26-27-28 | 39.16 ms | 219.4 GB/s | 27.42 GE/s |

Questi risultati anomali mettono in luce un problema molto specifico: quando il gate lavora con qubit bassi, la coalescenza viene compromessa. Analizziamo cosa fanno i primi work item nel caso in cui stiamo lavorando con i qubit 0-1-2:
- `global_id = 0` -> inserisce i tre zeri all'inizio della rappresentazione binaria del global_id e accede all'elemento 0;
- `global_id = 1` -> accede all'elemento 8;
- `global_id = 2` -> accede all'elemento 16;
- ...

Con il crescere delle dimensioni dei gate applicati, lo stride cresce e questa anomalia si fa piu' evidente e frequente. Fortunatamente, il problema sorge in maniera significativa soltanto quando si lavora esattamente con tutti i primi qubit della stringa.  
Inoltre, si suppone che nei casi di gate unario e binario il problema sia stato mascherato dall'alta efficienza della cache, e che invece l'aggiunta del terzo qubit l'abbia portata al limite.

# **Calcolo della Norma**

Il calcolo della norma del vettore si suddivide in due fasi:
1. Il calcolo dei quadrati dei moduli delle ampiezze, ovvero delle probabilita' associate ad ogni possibile stato;
2. La somma di tutte le probabilita'.

Per il primo step basta implementare un kernel imbarazzantemente parallelo banale, che lancia $N$ work item, ognuno che calcola la norma del relativo elemento e la scrive in un vettore di `float` di appoggio.  

Il secondo step invece e' un classico esempio di riduzione, un'operazione non imbarazzantemente parallelizzabbile. Il simulatore utilizza un'implementazione con vettorizzazione a 16 elementi e appoggio di local memory (o shared memory per CUDA), dove un'ulteriore step di riduzione viene effettuato dall'intero work group. Sebbene la somma tra float goda della proprieta' commutativa, esso **non gode della proprieta' associativa** in generale a cause di possibili arrotondamenti. Per fortuna, gli arrotondamenti si verificano soltanto quando la somma avviene tra due valori distanti diversi ordini di grandezza, scenario reso improbabile a causa della struttura ad albero delle riduzioni successive.
Gli acessi in memoria vengono effettuati in coalescenza tramite Indirizzamento Sequenziale: ogni work item somma 16 elementi tutti a 16 elementi di distanza l'uno dall'altro.

### **Analisi delle Prestazioni**
Le prestazioni vengono misurate soltanto dal punto di vista del throughput, ovvero degli elementi elaborati al secondo.  
Per confrontare l'efficienza del kernel sono stati implementate anche le versioni naive del kernel di riduzione: la somma a 2 elementi e a 16 elementi senza appoggio di local memory.

| Kernel | Tempo | Throughput |
| :--- | :--- | :--- |
| sum_2 |  115 ms | 4.668 GE/s |
| sum_16 |  51.79 ms | 10.37 GE/s |
| sum_16_lmem | 8.873 ms | 60.51 GE/s |

# **Misurazione**

L'ultima funzionalita' necessaria per avere un simulatore utilizzabile e' la misurazione di un qubit. Secondo i postulati della meccanica quantistica, la misurazione provoca un collasso del vettore di stato, azzerando totalmente le ampiezze relative agli stati incompatibili con il risultato misurato e richiedendo una successiva rinormalizzazione delle ampiezze rimaste integre. Calcolare la probabilita' di collasso in $\ket0$ o in $\ket1$ per il singolo qubit bersaglio a partire dall'intero vettore di stato, bisogna sommare tutte le ampiezze di probabilita' per cui quel qubit assume valore 0. 

L'operazione è stata implementata con un approccio ibrido in tre fasi:
1. Calcolo delle probabilita' di collasso $p_0$ e $p_1$: si tratta di una somma di $N/2$ elementi (non imbarazzantemente parallelizzabile);
2. Decisione casuale del collasso: l'estrazione di un numero casuale e la determinazione dell'outcome vengono eseguite dall'host;
3. Azzeramento delle ampiezze incompatibili e rinormalizzazione del vettore di stato: il calcolo su ogni elemento e' indipendente dagli altri (imbarazzantemente parallelizzabile);

Per il primo passo, e' stata implementata una versione lievemente modificata del kernel di somma precedentemente analizzato, che esegue la somma soltanto per gli elementi in cui il qubit target assume valore 0.

Dopo aver estratto casualmente l'outcome, il risultato viene fornito al secondo kernel, che in parallelo aggiorna tutti gli elementi del vettore di stato:  
- se nell'indice globale il qubit bersaglio ha valore pari a quello misurato, l'ampiezza va moltiplicata per un fattore di normalizzazione;
- se nell'indice globale il qubit bersagio ha valore opposto a quello misurato, l'ampiezza viene completamente azzerata.

### **Analisi delle Prestazioni**  

Kernel di collasso:

| Qubit bersaglio | Tempo | Throughput |
| :--- | :--- | :--- |
| 0 | 18.45 ms | 29.1 GE/s | 
| 1 | 18.47 ms | 29.06 GE/s |
| 2 | 9.439 ms | 56.88 GE/s |
| 14 | 9.581 ms | 56.04 GE/s | 
| 28 | 10.63 ms | 50.53 GE/s | 

Kernel di rinormalizzazione:

| Qubit bersaglio | Tempo | Bandwidth| Throughput |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 36.76 ms | 175.3 GB/s | 14.61 GE/s |
| 1 | 36.75 ms | 175.3 GB/s | 14.61 GE/s |
| 2 | 28.37 ms | 227.1 GB/s | 18.93 GE/s | 
| 14 | 27.98 ms | 230.2 GB/s | 19.18 GE/s |
| 28 | 27.18 ms | 237 GB/s | 19.75 GE/s |

Quando si misurano esattamente i qubit 0 e 1, le performance peggiorano notevolmente, specialmente per quanto riguarda il kernel di collasso. Ancora una volta questo comportamento e' dovuto al pattern di posizionamento delle ampiezze: per valori cosi' bassi, gli elementi a cui i kernel accedono si alternano con frequenza molto alta, generando warp divergence. Se nel caso del secondo kernel si tratta di un comportamento inevitabile, potrebbe essere possibile rimuovere il branching dal kernel di collasso: invece di lanciare N thread, se ne lanciano $N/2$, e si modifica il global_id in modo tale che ogni thread acceda esclusivamente agli indirizzi corretti. Questo puo' essere facilmente ottenuto inserendo un bit 0 nell'id in posizione target, ma cio' va gestito in contemporanea alle logiche di accesso della vettorizzazione implementata.

# **Conclusioni**
I dati raccolti confermano che la simulazione quantistica State Vector è un problema computazionale memory bound a causa dell'aumento esponenziale dello spazio richiesto e dei tempi di esecuzione al crescere di $n$. Le alte capacita' di calcolo della GPU non sono sfruttabili per spingersi oltre i 29 qubit. Ad ogni modo, la simulazione quantistica si conferma un problema molto adatto alla parallelizzazione, viste le altissime performance misurate da ogni kernel della pipeline di calcolo.  




