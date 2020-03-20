#This program analyzes two corpus that contains: *two of Hilary Clinton's speeches named "Remarks in Miami on the Cuba embargo" and "Remarks at the Brookings Institution on the Iran deal", both dated Jan 31, 2016; *ten of Donald Trump's speeches, from October 21, 2016 to October 31, 2016. 

# -*- coding: utf-8 -*-

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

#questa funzione calcola il numero di token di un testo
def CalcolaLunghezza(frasi):
    lunghezzaTOT=0.0
    for frase in frasi:
        #divido la frase in token
        tokens=nltk.word_tokenize(frase)
        #calcolo la lunghezza totale 
        lunghezzaTOT=lunghezzaTOT+len(tokens)
    #restituisco il risultato
    return lunghezzaTOT

#questa funzione calcola la lunghezza media delle frasi si un testo
def LunghezzaMediaFrasi(frasi):
    lunghezzaFrasi=0.0
    numFrasi=0.0
    for frase in frasi:
        #divido la frase in token
        tokens=nltk.word_tokenize(frase)
        #calcolo la lunghezza di ciascuna frase e la sommo con la lunghezza delle altre ottenendo la lunghezza del testo
        lunghezzaFrasi=lunghezzaFrasi+len(tokens)
        #il contatore registra il numero delle frasi osservate con lo scorrimento del ciclo for
        numFrasi=numFrasi+1
    #restituisco la media matematica della lunghezza di ciascuna frase
    return lunghezzaFrasi/numFrasi

#lista delle porzioni incrementali dei corpora
listaPorzioni=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

#questa funzione calcola la grandezza del vocabolario per porzioni incrementali di 1000 token
def GrandezzaVocabolario(testoTokenizzato):  
    listaLunghezze=[] 
    for porzione in listaPorzioni:
        #estraggo il vocabolario della porzione di corpus corrispondente
        Vocab=set(testoTokenizzato[:int(porzione)]) #rendo intero il valore perche' e' l'estremo di un intervallo
        #calcolo la lunghezza del vocabolario
        lunghezzaVocab=len(Vocab) 
        #aggiungo ogni valore ottenuto ad una lista
        listaLunghezze.append(lunghezzaVocab) 
    #restituisco la lista contenente tutte le lunghezze corrispondenti a ciascuna porzione di corpus
    return listaLunghezze 

#questa funzione calcola la ricchezza lessicale per porzioni incrementali di 1000 token
def CalcolaTTR(testoTokenizzato):
    listaTTR=[]
    for porzione in listaPorzioni:
        #estraggo il vocabolario della porzione di corpus corrispondente
        Vocab=set(testoTokenizzato[:int(porzione)]) #rendo intero il valore perche' e' l'estremo di un intervallo
        #calcolo la lunghezza del vocabolario
        lunghezzaVocab=len(Vocab)
        #calcolo la Type-Token-Ratio come rapporto tra la cardinalita' del vocabolario (num. di tipi) per la cardinalita' del corpus (num. di token)
        TTR=lunghezzaVocab*1.0/porzione*1.0
        #aggiungo ogni valore ottenuto ad una lista
        listaTTR.append(TTR)
    #restituisco la lista contenente tutte le TTR corrispondenti a ciascuna porzione di corpus
    return listaTTR

#questa funzione calcola il rapporto tra sostantivi e verbi osservati nel corpus
def RapportoSostantiviVerbi(POStag):
    #lista dei tag usati nel PoS tagging (Penn Treebank), da http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tagSostantivi = ["NN", "NNS", "NNP", "NNPS"]
    tagVerbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    #per ogni coppia (token, PoS) memorizzo in due liste tutte le pos rispettivamente di sostantivi e di verbi osservate nel corpus
    listaTagSostantivi=[]
    listaTagVerbi=[]
    for (tok, pos) in POStag: 
        if pos in tagSostantivi: 
           listaTagSostantivi.append(pos)
        if pos in tagVerbi: 
           listaTagVerbi.append(pos)
    #conto quante volte occorre ciascuna pos osservata nel corpus
    occorrenzeSostantivi=collections.Counter(listaTagSostantivi) 
    occorrenzeVerbi=collections.Counter(listaTagVerbi)
    #faccio la somma totale di tutti i conteggi
    sostantivi=sum(occorrenzeSostantivi.values())
    verbi=sum(occorrenzeVerbi.values())
    #calcolo e restituisco il rapporto
    return sostantivi*1.0/verbi*1.0

#questa funzione calcola la densita' lessicale
def DensitaLessicale(testoTokenizzato):
    POStag=nltk.pos_tag(testoTokenizzato)
    lunghezzaTesto=CalcolaLunghezza(testoTokenizzato)
    #lista dei tag usati nel PoS tagging (Penn Treebank), da http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tagSostantivi = ["NN", "NNS", "NNP", "NNPS"]
    tagVerbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    tagAvverbiEAggettivi=["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"]  
    tagPunteggiatura=[".", ","]
    #memorizzo nelle liste tutte le pos osservate nel corpus
    listaTagSostantivi=[]
    listaTagVerbi=[]
    listaTagAvverbiEAggettivi=[]
    listaTagPunteggiatura=[]
    for (tok, pos) in POStag: 
        if pos in tagSostantivi: 
           listaTagSostantivi.append(pos)
        if pos in tagVerbi: 
           listaTagVerbi.append(pos)
        if pos in tagAvverbiEAggettivi:
           listaTagAvverbiEAggettivi.append(pos)
        if pos in tagPunteggiatura:
           listaTagPunteggiatura.append(pos)
    #conto quante volte occorre ciascuna pos osservata nel corpus
    occorrenzeSostantivi=collections.Counter(listaTagSostantivi) 
    occorrenzeVerbi=collections.Counter(listaTagVerbi)
    occorrenzeAvverbiEAggettivi=collections.Counter(listaTagAvverbiEAggettivi)
    occorrenzePunteggiatura=collections.Counter(listaTagPunteggiatura)
    #faccio la somma totale di tutti i conteggi
    sostantivi=sum(occorrenzeSostantivi.values())
    verbi=sum(occorrenzeVerbi.values())
    avverbiEAggettivi=sum(occorrenzeAvverbiEAggettivi.values())
    punteggiatura=sum(occorrenzePunteggiatura.values())
    #restituisco il risultato del calcolo
    return (sostantivi + verbi + avverbiEAggettivi) / (lunghezzaTesto - punteggiatura)


#questa funzione restituisce due liste contenenti i token dei due corpora e li confronta in base alla lunghezza
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
  
  #calcolo la lunghezza dei due testi 
  lunghezzaFile1 = CalcolaLunghezza(frasi1)
  lunghezzaFile2 = CalcolaLunghezza(frasi2)
 
  print "\n\nPROGRAMMA #1\n"

  #confronto i testi sulla base del numero di token e stampo i risultati
  print "\nIl file", file1, "e' lungo", lunghezzaFile1, "token"
  print "Il file", file2, "e' lungo", lunghezzaFile2, "token"
  if (lunghezzaFile1>lunghezzaFile2):
      print file1, "e' piu' lungo di", file2
  elif (lunghezzaFile1<lunghezzaFile2):
      print file2, "e' piu' lungo di", file1
  else:
      print "i due file hanno la stessa lunghezza"

  #calcolo la lunghezza media delle frasi nei due testi 
  lunghezzaMedia1 = LunghezzaMediaFrasi(frasi1)
  lunghezzaMedia2 = LunghezzaMediaFrasi(frasi2)

  #confronto i testi sulla base della lunghezza media delle frasi e stampo i risultati
  print "\nLe frasi del file", file1, "hanno lunghezza media pari a", lunghezzaMedia1, "token"
  print "Le frasi del file", file2, "hanno lunghezza media pari a", lunghezzaMedia2, "token"

  #calcolo le cardinalita' dei vocabolari dei due file
  lunghezzaVocabolario1=len(set(testoTokenizzato1))
  lunghezzaVocabolario2=len(set(testoTokenizzato2))
  #confronto i testi sulla base del vocabolario
  print "\nIl vocabolario del file", file1, "ha", lunghezzaVocabolario1, "tipi"
  print "Il vocabolario del file", file2, "ha", lunghezzaVocabolario2, "tipi"
  if (lunghezzaVocabolario1>lunghezzaVocabolario2):
      print file1, "ha un vocabolario piu' ricco di", file2
  elif (lunghezzaVocabolario2>lunghezzaVocabolario2):
      print file2, "ha un vocabiolario piu' ricco di", file1
  else:
      print "i vocabolari dei due file hanno la stessa lunghezza"

  #restituisco la lunghezza dei vocabolari dei due file per porzioni incrementali dei due corpora
  vocabolario1 = GrandezzaVocabolario(testoTokenizzato1)
  vocabolario2 = GrandezzaVocabolario(testoTokenizzato2)
  print "\nGrandezza del vocabolario del file", file1, "all'aumento del corpus per porzioni incrementali di 1000 token:"
  for voc1 in vocabolario1:
          print voc1
  print "\nGrandezza del vocabolario del file", file2, "all'aumento del corpus per porzioni incrementali di 1000 token:"
  for voc2 in vocabolario2:
          print voc2
  print "\nNotiamo dal risultato che il vocabolario tende a crescere sempre piu' lentamente."

  #restituisco la ricchezza lessicale dei due file per porzioni incrementali dei due corpora
  TTR1= CalcolaTTR(testoTokenizzato1)
  TTR2= CalcolaTTR(testoTokenizzato2)
  print "\nRicchezza lessicale del file", file1, "all'aumento del corpus per porzioni incrementali di 1000 token:"
  for t1 in TTR1:
      print t1
  print "\nRicchezza lessicale del file", file2, "all'aumento del corpus per porzioni incrementali di 1000 token:"
  for t2 in TTR2:
      print t2
  print "Piu' e' alto il valore della Type-Token-Ratio, maggiore e' la ricchezza del vocabolario"

  #memorizzo su due variabili i due testi annotati sulla base delle Part-Of-Speech
  POStag1=nltk.pos_tag(testoTokenizzato1)
  POStag2=nltk.pos_tag(testoTokenizzato2)

  #calcolo il rapporto sostantivi/verbi nei due testi
  rapportoSV1=RapportoSostantiviVerbi(POStag1)
  rapportoSV2=RapportoSostantiviVerbi(POStag2)
  print "\nIl rapporto sostantivi/verbi nel file", file1, "e': ", rapportoSV1
  print "Il rapporto sostantivi/verbi nel file", file2, "e': ", rapportoSV2
  
  #calcolo la densita' lessicale dei due testi
  densita1=DensitaLessicale(testoTokenizzato1)
  densita2=DensitaLessicale(testoTokenizzato2)
  print "\nLa densita' lessicale del file", file1, "e': ", densita1
  print "La densita' lessicale del file", file2, "e': ", densita2

main(sys.argv[1], sys.argv[2])
