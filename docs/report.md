# **Simulatore Quantistico State Vector in CUDA**
### Relazione progetto GPGPU - Valerio Tosto

## Abstract
Lo scopo della seguente relazione e' quello di documentare lo sviluppo di un simulatore quantistico State Vector
realizzato con l'idea di sfruttare una GPU per la parallelizzazione del codice, puntando alle migliori performance possibili.
Il simulatore e' stato sviluppato in CUDA C++, con hardware di riferimento NVIDIA GTX 1080.

# **Introduzione**  
## Introduzione al Quantum Computing

Il Quantum Computing e' una branca dell'informatica in rapida crescita, basata sull'idea di sfruttare la **meccanica quantistica** per ottenere vantaggi computazionali.  
Vi sono due principi fisici che caratterizzano la meccanica quantistica:  
- Il **principio di sovrapposizione**, per cui una particella quantistica puo' trovarsi in piu' stati fisici in contemporanea, ognuno con una certa probabilita';
- L'entanglement quantistico, ovvero la capacita' di due particelle quantistiche di trovarsi in correlazione, per cui un'informazione sullo stato di una automaticamente ci informa anche riguardo lo stato dell'altra.

Se la computazione classica lavora con il bit, ovvero informazioni di natura binaria, il QC lavora con l'informazione quantistica: il qubit.
Un qubit e' un'unita' di informazione che ha una certa probabilita' $\alpha$