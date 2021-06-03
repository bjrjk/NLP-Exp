import IO, tokenizer, entropy, HMM, WSD
import jieba

def Prepare():
    IO.WriteSentenceData()
    tokenizer.writeSortedTokenFreqStat()
    HMM.train()

def CalculateEntropy():
    tokenFreq = tokenizer.importSortedTokenFreqStat()
    print(entropy.CalculateEntropy(tokenFreq))

def calculateNorm(hmmResult: str, jiebaResult: str):
    hmmList = hmmResult.split()
    jiebaList = jiebaResult.split()
    intersection = list(set(hmmList).intersection(set(jiebaList)))
    print(f"n: {len(intersection)}")
    print(f"N: {len(hmmList)}")
    print(f"M: {len(jiebaList克里姆林宫新闻局5月25日发布消息表示，俄罗斯总统普京与美国总统拜登将于6月16日在日内瓦举行会谈。)}")

def predict():
    HMM.loadModel()
    while True:
        txt = input(">>")
        mdlTXT, resultTXT = HMM.predict(txt)
        print(f"Model Result: {mdlTXT}")
        print(f"Sentence Result: {resultTXT}")
        jiebaResult = " ".join(jieba.cut(txt))
        print(f"Jieba Result: {jiebaResult}")
        calculateNorm(resultTXT, jiebaResult)

def func_WSD():
    while True:
        WSD.WSD_Main()

def Main():
    predict()
    # func_WSD()
    # CalculateEntropy()

if __name__ == '__main__':
    Main()