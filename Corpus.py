import Config
import re
from random import shuffle
import numpy as np
import math

class Vocabulary:

    def __init__(self):
        self.word2ID = {}
        self.id2Word = []

    def loadDict(self, dictf):
        f = open(dictf, encoding="utf8")
        line = f.readline()
        wid = 0
        while line:
            key = line.split("\t")[0]
            self.word2ID[key] = wid
            wid = wid + 1
            self.id2Word.append(key)
            line = f.readline()
        f.close()

    @staticmethod
    def buildDict(file, dictf):
        print("building vocabulary for " + file)
        f = open(file)
        line = f.readline()
        wordCounter = {}
        while line:
            line = line.strip()+ " </s>"
            words = re.split("\s+", line)
            for word in words:
                if(word in wordCounter):
                    wordCounter[word] = wordCounter[word] + 1
                else:
                    wordCounter[word] = 1
            line = f.readline()
        f.close()
        wordCounter["</s>"] = 100000000
        wordCounter["<unk>"] = 90000000
        f = open(dictf, "w")
        for word in sorted(wordCounter, key = wordCounter.get, reverse=True):
            f.write(word + "\t" + str(wordCounter[word]) + "\n")
        f.close()

    def getID(self, word):
        if word in self.word2ID:
            return self.word2ID[word]
        else:
            return self.word2ID["<unk>"]

    def getWord(self, wordID):
        if wordID < len(self.id2Word):
            return self.id2Word[wordID]
        else:
            return "<unk>"

    def getWordList(self, wordIDs):
        return [self.getWord(id) for id in wordIDs]

    def getIDList(self, words):
        return [self.getID(word) for word in words]

    def getEndId(self):
        return self.getID("</s>")

class MonoCorpus:

    def __init__(self, vocabF, file, shuffle = False):
        self.vocab = Vocabulary()
        self.sentences = []
        self.batchPool = []
        self.batchId = 0
        self.curSent = 0
        self.vocab.loadDict(vocabF)
        self.needShuffle = shuffle
        self.loadData(file)

    def loadData(self, file):
        print("Loading data " + file)
        sentences = []
        f=open(file)
        line = f.readline()
        while line:
            line = line.strip() + " </s>"
            words = re.split("\s+", line)
            wordIDs = self.vocab.getIDList(words)
            if(len(wordIDs) < 5):
                self.sentences.append(wordIDs)
            line = f.readline()
        f.close()

    def buildBatchPool(self):
        batchPool = []
        sents = self.getSentences(Config.BatchSize * 100)
        sents = sorted(sents, key=lambda sent : len(sent))
        self.batchPool = [sents[x:x + Config.BatchSize] for x in range(0, len(sents), Config.BatchSize)]
        #shuffle(self.batchPool)
        self.batchId = 0

    def reset(self):
        self.curSent = 0
        if (self.needShuffle): shuffle(self.sentences)

    def getSentences(self, num):
        sentences = []
        for i in range(0, num, 1):
            if (self.curSent >= len(self.sentences)): self.reset()
            sentence = self.sentences[self.curSent]
            sentences.append(sentence)
            self.curSent += 1
        return sentences

    def getTrainBatch(self):
        if(self.batchId >= len(self.batchPool)): self.buildBatchPool()
        rBatch = self.batchPool[self.batchId]
        self.batchId += 1
        return rBatch

    def getValBatch(self, num=Config.BatchSize):
        if (self.curSent >= len(self.sentences)):
            self.curSent = 0
            return None
        sentences = []
        for i in range(0, num, 1):
            if(self.curSent>= len(self.sentences)): break
            sentence = self.sentences[self.curSent ]
            sentences.append(sentence)
            self.curSent += 1
        return sentences

    def getEndId(self):
        return self.vocab.getEndId()

    @staticmethod
    def buildInputMono(sentences, vocabSize, maxLength, endID):
        batch = np.zeros((maxLength, Config.BatchSize), dtype=np.int32)
        for i in range(0, maxLength, 1):
            for j in range(0, Config.BatchSize, 1):
                if (j < len(sentences) and i < len(sentences[j])):
                    batch[i][j] = sentences[j][i]
                else:
                    batch[i][j] = endID

        batchMask = np.zeros((maxLength, Config.BatchSize), dtype=np.float32)
        for i in range(0, len(sentences), 1):
            sentence = sentences[i]
            lastIndex = len(sentence) if len(sentence) < maxLength else maxLength
            batchMask[0:lastIndex, i] = 1

        return (batch, batchMask)

    @staticmethod
    def getBucket(length, bucketGap, maxLength):
        bucketID = int(math.ceil(length / float(bucketGap)) * bucketGap)
        bucketID = bucketID if bucketID <= maxLength else maxLength
        return bucketID

class BiCorpus:

    def __init__(self, srcVocab, trgVocab, srcF, trgF, shuffle = False):
        self.srcVocab = srcVocab
        self.trgVocab = trgVocab
        self.sentencePairs = []
        self.batchPool = []
        self.batchId = 0
        self.curSent = 0
        self.needShuffle = shuffle
        self.loadData(srcF, trgF)

    def loadData(self, fileSrc, fileTrg):
        print("Loading data " + fileSrc + "-->" + fileTrg)
        sentences = []
        fsrc=open(fileSrc, encoding="utf8")
        ftrg = open(fileTrg, encoding="utf8")
        line = fsrc.readline()
        while line:
            line = line.strip()
            words = re.split("\s+", line)
            srcS = self.srcVocab.getIDList(words)
            line = ftrg.readline()
            line = line.strip() + " </s>"
            words = re.split("\s+", line)
            trgS = self.trgVocab.getIDList(words)
            self.sentencePairs.append((srcS, trgS))
            line = fsrc.readline()
        fsrc.close()
        ftrg.close()

    @staticmethod
    def getBucketIDs(srcLength, trgLength, bucketGap, maxLength):
        srcBucket = MonoCorpus.getBucket(srcLength, bucketGap, maxLength)
        trgBucket = MonoCorpus.getBucket(trgLength, bucketGap, maxLength)
        return (srcBucket, trgBucket)

    def getBuckets(self):
        buckets = {}
        for (srcSent, trgSent) in self.sentencePairs:
            srcLength = len(srcSent)
            trgLength = len(trgSent)
            (srcBucket, trgBucket) = BiCorpus.getBucketIDs(srcLength, trgLength, Config.BucketGap, Config.MaxLength)
            buckets[(srcBucket, trgBucket)] = 1
        return buckets

    def buildBatchPool(self):
        batchPool = []
        sentences = self.getSentences(Config.BatchSize * 1)
        sentences = sorted(sentences, key=lambda sent : len(sent[1]))
        self.batchPool = [sentences[x:x + Config.BatchSize] for x in range(0, len(sentences), Config.BatchSize)]
        shuffle(self.batchPool)
        self.batchId = 0

    def reset(self):
        self.curSent = 0
        if (self.needShuffle): shuffle(self.sentencePairs)

    def getSentences(self, num):
        sentences = []
        for i in range(0, num, 1):
            if(self.curSent >= len(self.sentencePairs)): self.reset()
            sentence = self.sentencePairs[self.curSent]
            sentences.append(sentence)
            self.curSent += 1
        return sentences

    def iD2SentPairs(self, idsPair):
        srcSent = self.srcVocab.getWordList(idsPair[0])
        trgSent = self.trgVocab.getWordList(idsPair[1])
        return (srcSent, trgSent)

    def iD2Sent(self, ids, src=True):
        return self.srcVocab.getWordList(ids) if src else self.trgVocab.getWordList(ids)

    def getTrainBatch(self):
        if(self.batchId >= len(self.batchPool)): self.buildBatchPool()
        rBatch = self.batchPool[self.batchId]
        self.batchId += 1
        return rBatch

    def getValBatch(self, num=Config.BatchSize):
        if (self.curSent >= len(self.sentencePairs)):
            self.curSent = 0
            return None
        sentences = []
        for i in range(0, num, 1):
            if(self.curSent + i >= len(self.sentencePairs)): break
            sentence = self.sentencePairs[self.curSent + i]
            sentences.append(sentence)
        self.curSent += len(sentences)
        return sentences

    def getEndId(self, source=True):
        if(source):
            return self.srcVocab.getEndId()
        else:
            return self.trgVocab.getEndId()

    def buildInput(self, sentences):
        (batchSrc, batchSrcMask) = MonoCorpus.buildInputMono([pair[0] for pair in sentences], Config.VocabSize, Config.MaxLength, self.getEndId(True))
        (batchTrg, batchTrgMask) = MonoCorpus.buildInputMono([pair[1] for pair in sentences], Config.VocabSize, Config.MaxLength, self.getEndId(False))
        return (batchSrc, batchTrg, batchSrcMask, batchTrgMask)

class ValCorpus:

    def __init__(self, srcVocab, trgVocab, valFile, nRef):
        self.srcVocab = srcVocab
        self.trgVocab = trgVocab
        self.sentencePairs = []
        self.nRef = nRef
        self.curSent = 0
        self.needShuffle = shuffle
        self.loadData(valFile)

    def loadData(self, valFile):
        print("Loading data " + valFile)
        sentences = []
        f = open(valFile)
        line = f.readline()
        while line:
            line = line.split("||||")[0]
            line = line.strip()
            words = re.split("\s+", line)
            src = self.srcVocab.getIDList(words)
            line = f.readline()
            trgs = []
            for refi in range(0, self.nRef, 1):
                line = f.readline()
                line = line.strip() + " </s>"
                words = re.split("\s+", line)
                trgS = self.trgVocab.getIDList(words)
                trgs.append(trgS)
            self.sentencePairs.append((src, trgs))
            line = f.readline()
        f.close()

    def getValBatch(self, num=Config.BatchSize):
        if (self.curSent >= len(self.sentencePairs)):
            self.curSent = 0
            return None
        sentences = []
        for i in range(0, num, 1):
            if(self.curSent + i >= len(self.sentencePairs)): break
            sentence = self.sentencePairs[self.curSent + i]
            sentences.append(sentence)
        self.curSent += len(sentences)
        return sentences

    @staticmethod
    def splitFile(valFile, srcFile, trgFile, nRef):
        sentences = []
        f = open(valFile)
        fSrc = open(srcFile, "w")
        fTrg = open(trgFile, "w")
        line = f.readline()
        while line:
            line = line.split("||||")[0]
            line = line.strip()
            fSrc.write(line+ "\n")
            line = f.readline()
            trgs = []
            for refi in range(0, nRef, 1):
                line = f.readline()
                line = line.strip()
                if(refi ==0 ):
                    fTrg.write(line+ "\n")
            line = f.readline()
        f.close()
        fSrc.close()
        fTrg.close()
