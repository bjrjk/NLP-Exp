import IO, tokenizer, entropy, HMM, WSD

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

def func_WSD():
    WSD.WSD_Main()

def Main():
    func_WSD()

if __name__ == '__main__':
    Main()