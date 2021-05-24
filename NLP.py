import IO, tokenizer, entropy, HMM

def Prepare():
    IO.WriteSentenceData()
    tokenizer.writeSortedTokenFreqStat()
    HMM.train()

def CalculateEntropy():
    tokenFreq = tokenizer.importSortedTokenFreqStat()
    print(entropy.CalculateEntropy(tokenFreq))

def predict():
    HMM.loadModel()
    while True:
        txt = input(">>")
        mdlTXT, resultTXT = HMM.predict(txt)
        print(f"Model Result: {mdlTXT}")
        print(f"Sentence Result: {resultTXT}")

def Main():
    predict()

if __name__ == '__main__':
    Main()