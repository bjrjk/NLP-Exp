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
        print("辐射：影响范围")
    else:
        print("辐射：核、电磁")

def WSD_Main():
    WSD_Predict(input("Input WSD 辐射 Sentence:"))
