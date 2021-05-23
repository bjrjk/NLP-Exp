from typing import List, Tuple
import os
import IO, config

def getSortedTokenFreqStat() -> List[Tuple[str, int]]:
    sentenceList = IO.ImportSentenceData()
    resultDict = {}
    for sentence in sentenceList:
        for word in sentence.split():
            if word in resultDict.keys():
                resultDict[word] += 1
            else:
                resultDict[word] = 1
    return sorted(resultDict.items(), key= lambda elem: elem[1], reverse= True)

def importSortedTokenFreqStat() -> List[Tuple[str, int]]:
    resultList = []
    path = os.path.join(config.PROCESSED_DATA_PATH, config.TOKENFREQ_FILENAME)
    content = IO.ReadTXT(path)
    for line in content.splitlines():
        l = line.split()
        resultList.append((l[0], int(l[1]), ))
    return resultList

def writeSortedTokenFreqStat() -> None:
    tokenList = getSortedTokenFreqStat()
    content = ''
    for item in tokenList:
        content += f"{item[0]} {item[1]}\n"
    path = os.path.join(config.PROCESSED_DATA_PATH, config.TOKENFREQ_FILENAME)
    IO.WriteTXT(path, content)