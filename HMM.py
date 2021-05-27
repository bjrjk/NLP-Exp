from typing import Tuple, List
import numpy as np
import pickle, os, re
import IO, config

relation = {  # Character represents:
    'B': 0,  # Begin Character of Word
    'M': 1,  # Middle Characters of Word
    'E': 2,  # End Character of Word
    'S': 3   # The only Character in Word
}
revRelation = "BMES"
DEFAULT_RATE = 0.000001
observeCounter = 0
observeRelation = {}

initStatus = [0.0, 0.0, 0.0, 0.0]  # Pi
statusTransMat = None  # A
observeMat = None  # B

def saveModel() -> None:  # 保存模型
    global observeCounter, observeRelation
    global initStatus, statusTransMat, observeMat
    path = os.path.join(config.PROCESSED_DATA_PATH, config.SEGMODEL_FILENAME)
    model = [observeCounter, observeRelation, initStatus, statusTransMat, observeMat]
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print("Model Saved!")


def loadModel() -> None:  # 加载模型
    global observeCounter, observeRelation
    global initStatus, statusTransMat, observeMat
    path = os.path.join(config.PROCESSED_DATA_PATH, config.SEGMODEL_FILENAME)
    with open(path, 'rb') as f:
        model = pickle.load(f)
    observeCounter = model[0]
    observeRelation = model[1]
    initStatus = model[2]
    statusTransMat = model[3]
    observeMat = model[4]
    print("Model Loaded!")

# Get the Status and Output Sequence of a sentence
def getSentenceSOSeq(sentence: List[str]) -> Tuple[List[str], List[str]]:
    charList = []
    statusList = []
    for word in sentence:
        if len(word) == 1:
            statusList.append('S')
            charList.append(word)
        elif len(word) > 1:
            statusList.append('B')
            for k in range(len(word) - 2):
                statusList.append('M')
            statusList.append('E')
            for c in word:
                charList.append(c)
    return charList, statusList


def train() -> None:  # 训练模型
    global relation, observeCounter, observeRelation
    global initStatus, statusTransMat, observeMat
    print("Start to train...")

    sentenceList = IO.ImportSentenceData2()  # Load Sentences
    sentenceArrayList = []
    for i in range(len(sentenceList)):
        sentenceArrayList.append(sentenceList[i].split())  # Turn sentence string to word list

    # Train initStatus
    print("Training initStatus...")
    tmpCnter = 0
    for sentence in sentenceArrayList:
        if tmpCnter % 1000 == 0:
            print(f"initStatus: {tmpCnter} / {len(sentenceArrayList)}")
        if len(sentence) == 0:
            continue
        length = len(sentence[0])
        # If length of first word in sentence is 1, the initial Markov status should be S
        if length == 1:
            initStatus[relation['S']] += 1
        # If length of first word in sentence large than 1, the initial Markov status should be B
        else:
            initStatus[relation['B']] += 1
        tmpCnter += 1
    for i in range(len(initStatus)):
        initStatus[i] /= len(sentenceArrayList)

    # Train statusTransMat, Prepare Train for observeMat
    print("Training statusTransMat...")
    tmpCnter = 0
    statusTransMat = np.zeros((4, 4))
    SOList = []
    for sentence in sentenceArrayList:
        if tmpCnter % 1000 == 0:
            print(f"statusTransMat: {tmpCnter} / {len(sentenceArrayList)}")
        charList, statusList = getSentenceSOSeq(sentence)
        SOList.append({"charList": charList, "statusList": statusList})
        # Index Character in charList
        for c in charList:
            if c not in observeRelation.keys():
                observeRelation[c] = observeCounter
                observeCounter += 1
        # Sum statusTransMat
        for i in range(len(statusList) - 1):
            statusTransMat[relation[statusList[i]]][relation[statusList[i + 1]]] += 1
        tmpCnter += 1
    sumsMat = np.array([np.sum(statusTransMat, axis=1)]).T
    statusTransMat = statusTransMat / sumsMat

    # Train observeMat
    print("Training observeMat...")
    tmpCnter = 0
    observeMat = np.zeros((4, observeCounter))
    for SO in SOList:
        if tmpCnter % 1000 == 0:
            print(f"observeMat: {tmpCnter} / {len(SOList)}")
        charList = SO['charList']
        statusList = SO['statusList']
        for i in range(len(charList)):
            observeMat[relation[statusList[i]]][observeRelation[charList[i]]] += 1
        tmpCnter += 1
    sumsMat = np.array([np.sum(observeMat, axis=1)]).T
    observeMat = observeMat / sumsMat

    saveModel()

def getObserveMatWithDefault(j: int, c:str) -> float:
    global observeRelation
    global observeMat
    global DEFAULT_RATE
    if c not in observeRelation.keys():
        return DEFAULT_RATE
    elif observeMat[j][observeRelation[c]] == 0:
        return DEFAULT_RATE
    return observeMat[j][observeRelation[c]]

def getInitStatusWithDefault(i: int) -> float:
    global initStatus
    global DEFAULT_RATE
    if initStatus[i] == 0:
        return DEFAULT_RATE
    return initStatus[i]

def getStatusTransMatWithDefault(i: int, j: int) -> float:
    global statusTransMat
    global DEFAULT_RATE
    if statusTransMat[i][j] == 0:
        return DEFAULT_RATE
    return statusTransMat[i][j]

def predict(sentence: str) -> str:  # Viterbi算法
    global observeCounter, observeRelation
    global initStatus, statusTransMat, observeMat
    global revRelation

    length = len(sentence)
    f = np.zeros((length, 4))
    path = np.zeros((length, 4), dtype=int)

    # Initialization
    for i in range(4):
        f[0][i] = getInitStatusWithDefault(i) * getObserveMatWithDefault(i, sentence[0])
        path[0][i] = i

    # Viterbi
    for t in range(1, length):
        for j in range(4):
            for i in range(4):
                if f[t-1][i] * getStatusTransMatWithDefault(i, j) > f[t][j]:
                    f[t][j] = f[t-1][i] * getStatusTransMatWithDefault(i, j)
                    path[t][j] = i
            f[t][j] *= getObserveMatWithDefault(j, sentence[t])

    # Find Max Result
    k = 0
    for i in range(4):
        if f[length - 1][i] > f[length - 1][k]:
            k = i
    pathSeq = []
    for i in range(length - 1, -1, -1):
        pathSeq.append(k)
        k = path[i][k]
    pathSeq.reverse()

    result = ""
    for p in pathSeq:
        result += revRelation[p]

    sentenceResult = ""
    p = re.compile(r'BM*E|S')
    for i in p.finditer(result):
        start, end = i.span()
        word = sentence[start:end]
        sentenceResult += word + " "

    return result, sentenceResult