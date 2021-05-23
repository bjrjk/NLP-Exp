import IO, tokenizer, entropy

def Prepare():
    IO.WriteSentenceData()
    tokenizer.writeSortedTokenFreqStat()

def Main():
    tokenFreq = tokenizer.importSortedTokenFreqStat()
    print(entropy.CalculateEntropy(tokenFreq))

if __name__ == '__main__':
    Main()