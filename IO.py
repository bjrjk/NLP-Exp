from typing import List
import os
import config

def ReadTXT(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def WriteTXT(path: str, content: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def ProcessTXT2Lines(content: str) -> List[str]:
    originList = content.splitlines()
    resultList = []
    for elem in originList:
        if len(elem) != 0 and elem[0] != '<':
            resultList.append(elem)
    return resultList

def ImportSentenceData() -> List[str]:
    resultList = []
    for root, dirs, files in os.walk(config.SEGMENTED_DATA_PATH):
        for f in files:
            path = os.path.join(root, f)
            content = ReadTXT(path)
            resultList.extend(ProcessTXT2Lines(content))
    return resultList

def ImportSentenceData2() -> List[str]:
    resultList = []
    path = os.path.join(config.PROCESSED_DATA_PATH, config.SENTENCE_FILENAME)
    content = ReadTXT(path)
    resultList.extend(ProcessTXT2Lines(content))
    return resultList

def WriteSentenceData() -> None:
    resultList = ImportSentenceData()
    path = os.path.join(config.PROCESSED_DATA_PATH, config.SENTENCE_FILENAME)
    with open(path, 'w', encoding='utf-8') as f:
        for elem in resultList:
            f.write(elem)
            f.write('\n')