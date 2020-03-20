#This program analyzes two corpus that contains: *two of Hilary Clinton's speeches named "Remarks in Miami on the Cuba embargo" and "Remarks at the Brookings Institution on the Iran deal", both dated Jan 31, 2016; *ten of Donald Trump's speeches, from October 21, 2016 to October 31, 2016. 

# -*- coding: utf-8 -*-

import re
import sys
import codecs
import nltk
import math
import collections
from nltk import word_tokenize, sent_tokenize
from nltk.probability import *


#questa funzione restituisce il testo tokenizzato
def TokenizzaTesto(frasi):
    tokensTOT=[]
    for frase in frasi:
        #divido la frase in token
        tokens = nltk.word_tokenize(frase) #prende in input una stringa e restituisce una lista di stringhe
        #concateno la frase appena tokenizzata con le tokenizzazioni precedenti
        tokensTOT = tokensTOT+tokens #questa variabile contiene tutto il file tokenizzato
    return tokensTOT

#questa funzione estrae le POS piu' frequenti nel corpus
def POSFrequenti(testoTokenizzato):
    POStag=nltk.pos_tag(testoTokenizzato)
    #creo una lista con tutte le POS osservate nel corpus
    listaTag=[]
    for (tok, pos) in POStag: 
           listaTag.append(pos)
    #conto quante volte occorre ciascuna pos osservata nel corpus
    frequenze=collections.Counter(listaTag)
    #restituisco le prime 10 piu' frequenti
    POSPiuFrequenti=frequenze.most_common(10)
    return POSPiuFrequenti

#questa funzione estrae i token piu' frequenti nel corpus
def TokenFrequenti(testoTokenizzato):
    SegniPunteggiatura=[".", ",", ":", ";", "!", "?", "...", "-"] #token da escludere
    #estraggo il testo senza punteggiatura
    testoSenzaPunteggiatura=[]
    for token in testoTokenizzato:
        if token not in SegniPunteggiatura:
            testoSenzaPunteggiatura.append(token)
    #calcolo la frequenza di ogni token del testo ottenuto
    freqToken=collections.Counter(testoSenzaPunteggiatura)
    #restituisco i primi 20 in ordine di frequenza decrescente
    tokenPiuFrequenti=freqToken.most_common(20)
    return tokenPiuFrequenti
    
#questa funzione estrae i bigrammi piu' frequenti nel corpus
def BigrammiFrequenti(testoTokenizzato):
    POStag=nltk.pos_tag(testoTokenizzato)
    bigrammi=nltk.bigrams(POStag)
    #lista dei tag per la classificazione di articoli, congiunzioni e  segni di punteggiatura usati nel PoS tagging (Penn Treebank)
    #da http://www.clips.ua.ac.be/pages/mbsp-tags
    tagArticoliCongiunzioniEPunteggiatura=["CC", "DT", "IN", "SYM", ".", ",", ":", "(", ")", "'", "''"] #POS da escludere
    #creo una lista di tutti i bigrammi le cui POS corrispondenti NON sono tra quelle da escludere
    listaBigrammi=[]
    for ((tok1, pos1), (tok2, pos2)) in bigrammi:
        if pos1 not in tagArticoliCongiunzioniEPunteggiatura:
            if pos2 not in tagArticoliCongiunzioniEPunteggiatura:
                bigramma= (tok1, tok2) 
                listaBigrammi.append(bigramma) 
    #calcolo la frequenza di ciascun bigramma
    freqBigrammi=collections.Counter(listaBigrammi)
    #restituisco i primi 20 in ordine di frequenza decrescente
    bigrammiPiuFrequenti=freqBigrammi.most_common(20)
    return bigrammiPiuFrequenti

#questa funzione estrae i trigrammi piu' frequenti nel corpus
def TrigrammiFrequenti(testoTokenizzato):
    POStag=nltk.pos_tag(testoTokenizzato)  
    trigrammi=nltk.trigrams(POStag)      
    #lista dei tag per la classificazione di articoli, congiunzioni e  segni di punteggiatura usati nel PoS tagging (Penn Treebank)
    #da http://www.clips.ua.ac.be/pages/mbsp-tags
    tagArticoliCongiunzioniEPunteggiatura=["CC", "DT", "IN", "SYM", ".", ",", ":", "(", ")", "'", "''"] #POS da escludere
    #creo una lista di tutti i trigrammi le cui POS corrispondenti NON sono tra quelle da escludere
    listaTrigrammi=[]
    for ((tok1, pos1), (tok2, pos2), (tok3, pos3)) in trigrammi:
        if pos1 not in tagArticoliCongiunzioniEPunteggiatura:
            if pos2 not in tagArticoliCongiunzioniEPunteggiatura:
                if pos3 not in tagArticoliCongiunzioniEPunteggiatura:
                    trigramma= (tok1, tok2, tok3) 
                    listaTrigrammi.append(trigramma) 
    #calcolo la frequenza di ciascun trigramma
    freqTrigrammi=collections.Counter(listaTrigrammi)
    #restituisco i primi 20 in ordine di frequenza decrescente
    trigrammiPiuFrequenti=freqTrigrammi.most_common(20)
    return trigrammiPiuFrequenti
 
#questa funzione estrae i bigrammi aggettivo-sostantivo in cui ogni token ha frequenza maggiore di 2
def BigrammiAggettivoSostantivo(testoTokenizzato):
    POStag=nltk.pos_tag(testoTokenizzato)
    #per ogni token calcolo quante volte occorre
    listaFreqToken=collections.Counter(testoTokenizzato)
    #creo una lista che contiene solo i token con frequenza maggiore di due
    listaToken=[]
    for tok in listaFreqToken:
        if listaFreqToken.values()>2:
            listaToken.append(tok)
    #estraggo i bigrammi
    bigrammi=nltk.bigrams(POStag)
    listaBigrammi=[]
    #lista dei tag per la classificazione di aggettivi e sostantivi usati nel PoS tagging (Penn Treebank)
    #da http://www.clips.ua.ac.be/pages/mbsp-tags
    tagAggettivi = ["JJ", "JJR", "JJS"]
    tagSostantivi = ["NN", "NNS", "NNP", "NNPS"]
    for ((tok1, pos1), (tok2, pos2)) in bigrammi:
        #se le POS sono rispettivamente un aggettivo e un nome
        if (pos1 in tagAggettivi) and (pos2 in tagSostantivi):
            #se i token dei bigrammi appartengono alla lista dei token con frequenza maggiore di 2
            if (tok1 in listaToken) and (tok2 in listaToken):
                bigramma= (tok1, tok2)
                #creo una lista di bigrammi che rispettano le condizioni, eliminando le POS che non mi servono per i calcoli successivi
                listaBigrammi.append(bigramma)
    return listaBigrammi

#questa funzione calcola la frequenza del primo elemento del bigramma (ovvero degli aggettivi) in tutto il corpus
def CalcolaFreqAggettivi(testoTokenizzato):
    POStag=nltk.pos_tag(testoTokenizzato)
    tagAggettivi = ["JJ", "JJR", "JJS"]
    #creo una lista con tutti gli aggettivi osservati nel corpus
    listaAggettivi=[]
    for (token, POS) in POStag:
        if POS in tagAggettivi:
           listaAggettivi.append(token)
    #calcolo la frequenza per ciascun aggettivo osservato
    freqAggettivo=collections.Counter(listaAggettivi)
    #trasformo l'oggetto Counter in una lista
    freqAggettivi=freqAggettivo.items() #contiene le frequenze degli aggettivi nel corpus
    return freqAggettivi

#questa funzione calcola la frequenza del secondo elemento del bigramma (ovvero dei sostantivi) in tutto il corpus
def CalcolaFreqSostantivi(testoTokenizzato):
    POStag=nltk.pos_tag(testoTokenizzato)
    tagSostantivi = ["NN", "NNS", "NNP", "NNPS"]
    #creo una lista con tutti i sostantivi osservati nel corpus
    listaSostantivi=[]
    for (token, POS) in POStag:
        if POS in tagSostantivi:
           listaSostantivi.append(token)
    #calcolo la frequenza per ciascun sostantivo osservato
    freqSostantivo=collections.Counter(listaSostantivi)
    #trasformo l'oggetto Counter in una lista
    freqSostantivi=freqSostantivo.items() #contiene le frequenze dei sostantivi nel corpus
    return freqSostantivi

#questa funzione calcola la prob condizionata di bigrammi agg-sost
def CalcoloProbCondizionata(testoTokenizzato):
    #estraggo i bigrammi che rispettano le condizioni
    listaBigrammi=BigrammiAggettivoSostantivo(testoTokenizzato)
    #per ogni bigramma agg-sost calcolo la sua frequenza
    freqBigramma=collections.Counter(listaBigrammi)
    #trasformo l'oggetto Counter in una lista
    freqBigrammi=freqBigramma.items() #contiene le frequenze dei bigrammi agg-sost nel corpus
    #calcolo la frequenza degli aggettivi
    freqAggettivi=CalcolaFreqAggettivi(testoTokenizzato)
    #creo una lista con i bigrammi e le relative probabilita' condizionate
    lista=[]
    for (agg, freqA) in freqAggettivi:
        for ((elem1, elem2), freqB) in freqBigrammi:
            if agg==elem1: #accoppio gli aggettivi con il bigramma in cui compaiono
               #calcolo la probabilita' condizionata
               probCondizionata=freqB*1.0 / freqA*1.0
               elemento= (elem1, elem2), probCondizionata
               lista.append(elemento)
    return lista

#questa funzione estrae i 20 bigrammi agg-sost con probabilita' condizionata massima
def BigrammiProbCondizionata(testoTokenizzato):
    #dalla funzione precedente estraggo una lista con tutte le prob condizionate per ciascun bigramma
    listaProbCondiz=CalcoloProbCondizionata(testoTokenizzato)
    #ordino la lista per probabilita' decrescenti
    listaOrdinata= sorted(listaProbCondiz, key = lambda a: -a[1], reverse=False)
    #restituisco i primi 20 bigrammi
    return listaOrdinata[:20]

#questa funzione calcola la prob congiunta di bigrammi agg-sost
def CalcoloProbCongiunta(testoTokenizzato):
    #Per calcolare la probabilita' congiunta e' necessario usare la probabilita' condizionata
    #estraggo una lista con tutte le prob condizionate per ciascun bigramma
    listaProbCondiz=CalcoloProbCondizionata(testoTokenizzato)
    #calcolo la frequenza degli aggettivi
    freqAggettivi=CalcolaFreqAggettivi(testoTokenizzato)
    lista=[]
    for (agg, freqA) in freqAggettivi:
        for ((elem1, elem2), probCondiz) in listaProbCondiz:
            if agg==elem1: #accoppio gli aggettivi con il bigramma in cui compaiono
                #calcolo la probabilita' congiunta
                probCongiunta=probCondiz*(freqA*1.0/len(testoTokenizzato)*1.0)
                elemento= (elem1, elem2), probCongiunta
                lista.append(elemento)
    return lista

#questa funzione estrae i 20 bigrammi agg-sost con probabilita' congiunta massima
def BigrammiProbCongiunta(testoTokenizzato):  
    #dalla funzione precedente estraggo una lista con tutte le prob congiunte per ciascun bigramma
    listaProbCong=CalcoloProbCongiunta(testoTokenizzato) 
    #ordino la lista per probabilita' decrescenti
    listaOrdinata= sorted(listaProbCong, key = lambda a: -a[1], reverse=False)
    #restituisco i primi 20 bigrammi
    return listaOrdinata[:20]

#questa funzione applica la formula della probabilita' (definizione frequentista)
def CalcoloProbabilita(frequenza, testoTokenizzato):
    probabilita=frequenza*1.0/len(testoTokenizzato)*1.0
    return probabilita

#questa funzione calcola il valore della Local Mutual Information per ciascun bigramma
def CalcoloLMI(testoTokenizzato):
    #calcolo la frequenza degli aggettivi
    freqAggettivi=CalcolaFreqAggettivi(testoTokenizzato)
    #calcolo la frequenza dei sostantivi
    freqSostantivi=CalcolaFreqSostantivi(testoTokenizzato)
    #estraggo i bigrammi che rispettano le condizioni
    listaBigrammi=BigrammiAggettivoSostantivo(testoTokenizzato)
    #per ogni bigramma agg-sost calcolo la sua frequenza
    freqBigramma=collections.Counter(listaBigrammi)
    #trasformo l'oggetto Counter in una lista
    freqBigrammi=freqBigramma.items() #contiene le frequenze dei bigrammi agg-sost nel corpus
    lista=[]
    for ((elem1, elem2), freqB) in freqBigrammi:
        for (agg, freqA) in freqAggettivi:
            for (sost, freqS) in freqSostantivi:
                #per i bigrammi agg-sost calcolo sia la prob del bigramma che quella dei singoli elementi che lo compognono
                if (elem1==agg) and (elem2==sost):
                   probabilitaAgg=CalcoloProbabilita(freqA, testoTokenizzato)
                   probabilitaSost=CalcoloProbabilita(freqS, testoTokenizzato)
                   probabilitaBigr=CalcoloProbabilita(freqB, testoTokenizzato)
                   #calcolo la MI memorizzando i valori intermedi del calcolo in una variabile
                   var=probabilitaBigr*1.0/(probabilitaAgg*probabilitaSost)*1.0
                   MI=math.log(var, 2)
                   #dalla MI ricavo la LMI
                   LMI=freqB*MI
                   elemento= (elem1, elem2), LMI
                   lista.append(elemento)
    return lista

#questa funzione estrae i 20 bigrammi agg-sost con forza associativa massima
def BigrammiLMI(testoTokenizzato):
    #dalla funzione precedente estraggo una lista con tutti i valori di LMI per ciascun bigramma
    listaBigrammiLMI=CalcoloLMI(testoTokenizzato)
    #ordino la lista per LMI decrescenti
    listaOrdinata= sorted(listaBigrammiLMI, key = lambda a: -a[1], reverse=False)
    #restituisco i primi 20 bigrammi
    return listaOrdinata[:20]

#questa funzione estrae la frase con probabilita' massima, calcolata con un modello markoviano di ordine 0
def MarkovOrdine0(testoTokenizzato, frasi):
    lunghezzaCorpus=len(testoTokenizzato) 
    #tokenizzo frase per frase
    listaFrasiTokenizzate=[]
    for frase in frasi:
        fraseTokenizzata=nltk.word_tokenize(frase)
        listaFrasiTokenizzate.append(fraseTokenizzata)
    #calcolo la freqenza di ciascun token osservato nella frase
    freqToken=collections.Counter(fraseTokenizzata)
    prob=1.0
    probMassima=0.0
    #calcolo la probabilita' di ciascun token
    for fraseTokenizzata in listaFrasiTokenizzate:
      for tok in fraseTokenizzata:
        probToken=CalcoloProbabilita(freqToken[tok], fraseTokenizzata)
        #nel modello di ordine 0 la probabilita' della frase equivale al prodotto delle probabilita' dei singoli token
        prob=prob*probToken
        #estraggo la probabilita' massima
        if prob > probMassima:
           probMassima=prob
    return fraseTokenizzata, "-------probabilita':", probMassima

#questa funzione estrae la frase con probabilita' massima, calcolata con un modello markoviano di ordine 1
def MarkovOrdine1(testoTokenizzato, frasi):
    lunghezzaCorpus=len(testoTokenizzato)
    freqToken=collections.Counter(testoTokenizzato) #lista di frequenze di ogni token nel testo
    bigrammi=nltk.bigrams(testoTokenizzato) #estraggo i bigrammi
    freqBigrammi=collections.Counter(bigrammi) #lista di frequenze di ogni bigramma nel testo
    probMassima=0.0
    fraseProbMax=""
    for frase in frasi:
        i=0.0
        #per ogni frase la tokenizzo ed estraggo i bigrammi
        tokens=nltk.word_tokenize(frase)
        frasiBigrammi=nltk.bigrams(tokens)
        #considero solo le frasi lunghe almeno 10 token e i token con frequenza maggiore di 2
        if len(tokens)>=10:
           for tok in tokens:
               if freqToken[tok]<2:
                  i=i+1
        if i==len(tokens): #quando ho letto tutta la frase
           #inizio calcolando la probabilita' semplice del primo token della frase
           probIntermedia=freqToken[tokens[0]]*1.0/lunghezzaCorpus*1.0
           for bigramma in frasiBigrammi:
               #nel modello di ordine 1 bisogna calcolare, per ogni bigramma, la prob condizionata(del secondo token del bigramma dato il primo)
               probBigr=freqBigrammi[bigramma]*1.0/freqToken[bigramma[0]]*1.0
               #faccio il prodotto di ogni valore ottenuto
               probIntermedia=probIntermedia*probBigr
           #confronto la probabilita' della frase (ottenuta dal prodotto) con la prob massima
           if probIntermedia>probMassima:
              probMassima=probIntermedia
              fraseProbMax=frase
    #restituisco la frase con la relativa probabilita' massima
    return fraseProbMax, "-------probabilita':", probMassima

#questa funzione restituisce i 20 nomi di persona piu' frequenti nel corpus
def NomiPropriPersona(testoTokenizzato):
    tutteLeNE=[]
    listaPersone=[]
    listaLuoghi=[]
    tokensPOS=nltk.pos_tag(testoTokenizzato) #lista di bigrammi (token, POS)
    analisi=nltk.ne_chunk(tokensPOS) #rappresentazione ad albero
    IOBformat=nltk.chunk.tree2conllstr(analisi) #trasformo in formato IOB
    for nodo in analisi: #ciclo l'albero scorrendo i nodi
        if hasattr(nodo, 'label'): #controlla se e' un nodo intermedio
           if nodo.label() =="PERSON":
              elementoP= nodo.leaves()
              #converte l'elemento in una tupla (utile per elementi composti da piu token che altrimenti verrebbero restituiti in sottoliste)
              listaPersone.append(tuple(elementoP)) 
    #calcolo le frequenze
    freqPersone=collections.Counter(listaPersone)
    #restituisco i primi 20
    persone=freqPersone.most_common(20)
    return persone
 
#questa funzione restituisce i 20 nomi di luogo piu' frequenti nel corpus       
def NomiPropriLuogo(testoTokenizzato):
    tutteLeNE=[]
    listaPersone=[]
    listaLuoghi=[]
    tokensPOS=nltk.pos_tag(testoTokenizzato) #lista di bigrammi (token, POS)
    analisi=nltk.ne_chunk(tokensPOS) #rappresentazione ad albero
    IOBformat=nltk.chunk.tree2conllstr(analisi) #trasformo in formato IOB
    for nodo in analisi: #ciclo l'albero scorrendo i nodi
        if hasattr(nodo, 'label'): #controlla se e' un nodo intermedio 
           if nodo.label() =="GPE":
              elementoL= nodo.leaves()
              #converte l'elemento in una tupla (utile per elementi composti da piu token che altrimenti verrebbero restituiti in sottoliste)
              listaLuoghi.append(tuple(elementoL))  
    #calcolo le frequenze
    freqLuoghi=collections.Counter(listaLuoghi)
    #restituisco i primi 20
    luoghi=freqLuoghi.most_common(20)
    return luoghi

def main(file1, file2):
  fileInput1 = codecs.open(file1, "r", "utf-8")
  fileInput2 = codecs.open(file2, "r", "utf-8")
  var1 = fileInput1.read()
  var2 = fileInput2.read()
  #carico il modello statistico utilizzato dalla funzione tokenize
  sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  #divido i testi dei due file in frasi 
  frasi1 = sent_tokenizer.tokenize(var1) 
  frasi2 = sent_tokenizer.tokenize(var2)
  #tokenizzo il file1 
  testoTokenizzato1=TokenizzaTesto(frasi1)
  #tokenizzo il file2
  testoTokenizzato2=TokenizzaTesto(frasi2)
  #memorizzo su due variabili i due testi annotati sulla base delle Part-Of-Speech
  POStag1=nltk.pos_tag(testoTokenizzato1)
  POStag2=nltk.pos_tag(testoTokenizzato2)
  
  print "\n\nPROGRAMMA #2\n"

  #le 10 POS piu' frequenti
  POSFrequenti1=POSFrequenti(testoTokenizzato1)
  POSFrequenti2=POSFrequenti(testoTokenizzato2)
  print "\nLe 10 POS piu' frequenti del file", file1, "(ordinate in senso descrescente), sono:\n"
  for POSFrequente1 in POSFrequenti1:
      print POSFrequente1
  print "\n\nLe 10 POS piu' frequenti del file", file2, "(ordinate in senso descrescente), sono:\n"
  for POSFrequente2 in POSFrequenti2:
      print POSFrequente2

  #i 20 token piu' frequenti
  tokenFrequenti1=TokenFrequenti(testoTokenizzato1)
  tokenFrequenti2=TokenFrequenti(testoTokenizzato2)
  print "\n\nI 20 token piu' frequenti nel file", file1, "sono:\n"
  for tokenFrequente1 in tokenFrequenti1:
      print tokenFrequente1
  print "\n\nI 20 token piu' frequenti nel file", file2, "sono:\n"
  for tokenFrequente2 in tokenFrequenti2:
      print tokenFrequente2
  
  #i 20 bigrammi piu' frequenti
  bigrammi1=BigrammiFrequenti(testoTokenizzato1)
  bigrammi2=BigrammiFrequenti(testoTokenizzato2)
  print "\n\nI 20 bigrammi piu' frequenti nel file", file1, "sono:\n"
  for bigramma1 in bigrammi1:
     print bigramma1
  print "\n\nI 20 bigrammi piu' frequenti nel file", file2, "sono:\n"
  for bigramma2 in bigrammi2:
     print bigramma2
  
  #i 20 trigrammi piu' frequenti
  trigrammi1=TrigrammiFrequenti(testoTokenizzato1)
  trigrammi2=TrigrammiFrequenti(testoTokenizzato2)
  print "\n\nI 20 trigrammi piu' frequenti nel file", file1, "sono:\n"
  for trigramma1 in trigrammi1:
     print trigramma1
  print "\n\nI 20 trigrammi piu' frequenti nel file", file2, "sono:\n"
  for trigramma2 in trigrammi2:
     print trigramma2

  #i 20 bigrammi aggettivo-sostantivo con probabilita' condizionata massima
  bigrammiProbCondizionata1=BigrammiProbCondizionata(testoTokenizzato1)
  bigrammiProbCondizionata2=BigrammiProbCondizionata(testoTokenizzato2)
  print "\n\nI 20 bigrammi aggettivo-sostantivo con probabilita' condizionata massima nel file", file1, "sono:\n"
  for bigrammaProbCondizionata1 in bigrammiProbCondizionata1:
      print bigrammaProbCondizionata1
  print "\n\nI 20 bigrammi aggettivo-sostantivo con probabilita' condizionata massima nel file", file2, "sono:\n"
  for bigrammaProbCondizionata2 in bigrammiProbCondizionata2:
      print bigrammaProbCondizionata2

  #i 20 bigrammi aggettivo-sostantivo con probabilita' congiunta massima
  bigrammiProbCongiunta1=BigrammiProbCongiunta(testoTokenizzato1)
  bigrammiProbCongiunta2=BigrammiProbCongiunta(testoTokenizzato2)
  print "\n\nI 20 bigrammi aggettivo-sostantivo con probabilita' congiunta massima nel file", file1, "sono:\n"
  for bigrammaProbCongiunta1 in bigrammiProbCongiunta1:
      print bigrammaProbCongiunta1
  print "\n\nI 20 bigrammi aggettivo-sostantivo con probabilita' congiunta massima nel file", file2, "sono:\n"
  for bigrammaProbCongiunta2 in bigrammiProbCongiunta2:
      print bigrammaProbCongiunta2

  #i 20 bigrammi aggettivo-sostantivo con forza associativa massima
  bigrammiLMI1=BigrammiLMI(testoTokenizzato1)
  bigrammiLMI2=BigrammiLMI(testoTokenizzato2)
  print "\n\nI 20 bigrammi aggettivo-sostantivo con forza associativa massima nel file", file1, "sono:\n"
  for bigrammaLMI1 in bigrammiLMI1:
      print bigrammaLMI1
  print "\n\nI 20 bigrammi aggettivo-sostantivo con forza associativa massima nel file", file2, "sono:\n"
  for bigrammaLMI2 in bigrammiLMI2:
      print bigrammaLMI2

  #Entita' Nominate
  persone1=NomiPropriPersona(testoTokenizzato1)
  persone2=NomiPropriPersona(testoTokenizzato2)
  luoghi1=NomiPropriLuogo(testoTokenizzato1)
  luoghi2=NomiPropriLuogo(testoTokenizzato2)
  print "\n\nNel file", file1, "i 20 nomi propri di persona piu' frequenti sono:\n" 
  for persona1 in persone1:
      print persona1
  print "\nNel file", file1, "i 20 nomi propri di luogo piu' frequenti sono:\n" 
  for luogo1 in luoghi1:
      print luogo1
  print "\n\nNel file", file2, "i 20 nomi propri di persona piu' frequenti sono:\n" 
  for persona2 in persone2:
      print persona2
  print "\nNel file", file2, "i 20 nomi propri di luogo piu' frequenti sono:\n" 
  for luogo2 in luoghi2:
      print luogo2
  
  #le due frasi con probabilita' piu' alta calcolate con modello di ordine 0
  probFrase01=MarkovOrdine0(testoTokenizzato1, frasi1)
  print "\n\nLa frase con probabilita' piu' alta, calcolata con catena di Markov di ordine 0, nel file", file1, " e':\n", probFrase01
  probFrase02=MarkovOrdine0(testoTokenizzato2, frasi2)
  print "\n\nLa frase con probabilita' piu' alta, calcolata con catena di Markov di ordine 0, nel file", file2, " e':\n", probFrase02

  #le due frasi con probabilita' piu' alta calcolate con modello di ordine 1
  probFrase11=MarkovOrdine1(testoTokenizzato1, frasi1)
  print "\n\nLa frase con probabilita' piu' alta, calcolata con catena di Markov di ordine 1, nel file", file1, " e':\n", probFrase11
  probFrase12=MarkovOrdine1(testoTokenizzato2, frasi2)
  print "\n\nLa frase con probabilita' piu' alta, calcolata con catena di Markov di ordine 1, nel file", file2, " e':\n", probFrase12  

main(sys.argv[1], sys.argv[2])
