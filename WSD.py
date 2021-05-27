import jieba
import IO, os
WORD = '辐射'

def findSentenceInCorpus(word):
    sentenceList = IO.ImportSentenceData2()
    resultList = []
    for sentence in sentenceList:
        if sentence.find(word) != -1:
            resultList.append(sentence)
    return resultList

def cutSentence(sentence):
    return jieba.lcut(sentence)

def WSD_Prepare():
    sentenceList = findSentenceInCorpus('辐射')
    for sentence in sentenceList:
        print(sentence)

def stat(LWord, LFind):
    cnt = 0
    for word in LWord:
        if word in LFind:
            cnt += 1
    return cnt

def WSD_Predict(inputStr):
    FuShe1 = IO.ReadTXT("Data/Processed/FuShe.txt").split()
    FuShe2 = IO.ReadTXT("Data/Processed/FuShe2.txt").split()
    wordList = cutSentence(inputStr)
    stat1 = stat(wordList, FuShe1)
    stat2 = stat(wordList, FuShe2)
    print(f"Freq: {stat1}, {stat2}")
    if stat1 > stat2:
        print("动词：对周边事物产生的一定影响")
    else:
        print("物理名词：指的是由场源发出的电磁能量中一部分脱离场源向远处传播，而后不再返回场源的现象")

def WSD_Main():
    WSD_Predict(input("Input WSD 辐射 Sentence:"))
