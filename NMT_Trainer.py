from __future__ import print_function
import os
import tensorflow as tf
import math
import Config
import NMT_Model
import Corpus
import nltk
import numpy

class NMT_Trainer:

    def __init__(self):
        self.model = NMT_Model.NMT_Model()
        self.srcVocab = Corpus.Vocabulary()
        self.trgVocab = Corpus.Vocabulary()
        self.srcVocab.loadDict(Config.srcVocabF)
        self.trgVocab.loadDict(Config.trgVocabF)
        self.trainData = Corpus.BiCorpus(self.srcVocab, self.trgVocab, Config.trainSrcF, Config.trainTrgF)
        self.valData = Corpus.BiCorpus(self.srcVocab, self.trgVocab, Config.valSrcF, Config.valTrgF)
        self.buckets = self.trainData.getBuckets()
        self.networkBucket = {}
        self.bestValCE = 999999
        self.bestBleu = 0
        self.badValCount = 0
        self.maxBadVal = 5
        self.learningRate = Config.LearningRate
        self.inputSrc = tf.placeholder(tf.int32, shape=[Config.MaxLength, Config.BatchSize],
                                       name='srcInput')
        self.maskSrc = tf.placeholder(tf.float32, shape=[Config.MaxLength, Config.BatchSize], name='srcMask')
        self.inputTrg = tf.placeholder(tf.int32, shape=[Config.MaxLength, Config.BatchSize],
                                       name='trgInput')
        self.maskTrg = tf.placeholder(tf.float32, shape=[Config.MaxLength, Config.BatchSize], name='trgMask')
        self.optimizer = tf.train.AdamOptimizer()
        self.createBucketNetworks()

    def createBucketNetworks(self):
        for (srcLength, trgLength) in self.buckets:
            self.getNetwork(srcLength, trgLength)

    def getNetwork(self, srcBucket, trgBucket):
        if((srcBucket, trgBucket) not in self.networkBucket):
            print("Creating network (" + str(srcBucket) + "," + str(trgBucket) + ")", end="\r")
            self.networkBucket[(srcBucket, trgBucket)] = self.model.createTrainingNetwork(
                self.inputSrc, self.maskSrc, self.inputTrg, self.maskTrg, srcBucket, trgBucket, self.optimizer)
            print("Bucket contains " + str(len(self.networkBucket)) +" networks ", end="")
            for key in self.networkBucket: print("(" + str(key[0]) + "," + str(key[1]) + ")", end=" ")
            print()
        return self.networkBucket[(srcBucket, trgBucket)]

    def train(self):
        bestValScore = 10000
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(0, 100000000, 1):
            print("Training with batch " + str(i), end="\r")

            trainBatch = self.trainData.getTrainBatch()
            maxSrcLength = max(len(x[0]) for x in trainBatch)
            maxTrgLength = max(len(x[1]) for x in trainBatch)
            (srcBucketID, trgBucketID) = Corpus.BiCorpus.getBucketIDs(maxSrcLength, maxTrgLength,
                                                                      Config.BucketGap, Config.MaxLength)
            min_loss, loss = self.getNetwork(srcBucketID, trgBucketID)
            (batchSrc, batchTrg, srcMask, trgMask) = self.trainData.buildInput(trainBatch)
            train_dict = {self.inputSrc:batchSrc, self.maskSrc:srcMask, self.inputTrg: batchTrg, self.maskTrg: trgMask}
            _, cePerWord = sess.run([min_loss, loss], feed_dict=train_dict)
            if (i % 10 == 0):
                print(str(cePerWord / math.log(2.0)))


    def validateAndSaveModel(self, i, bestValScore):
        if (i % Config.ValiditionPerBatch == 0):
            valScore = self.validate()
            if (valScore < bestValScore):
                self.model.saveModel(Config.modelF + "." + str(i))
                bestValScore = valScore
                self.badValCount = 0
            else:
                self.badValCount += 1
                if(self.badValCount >= self.maxBadVal):
                    self.learningRate /=2
                    self.badValCount = 0
        return bestValScore

    def validate(self):
        valBatch = self.valData.getValBatch()
        countAll = 0
        ceAll = 0
        print("Validation ...", end="\r")
        while(valBatch):
            count = sum(len(s[1]) for s in valBatch)
            countAll += count
            maxSrcLength = max(len(x[0]) for x in valBatch)
            maxTrgLength = max(len(x[1]) for x in valBatch)
            network = self.getNetwork(maxSrcLength, maxTrgLength)

            (batchSrc, batchTrg, batchSrcMask, batchTrgMask) = self.valData.buildInput(valBatch)
            ce = network.eval({self.model.inputMatrixSrc: batchSrc,
                                      self.model.inputMatrixTrg:batchTrg,
                                      self.model.maskMatrixSrc:batchSrcMask,
                                      self.model.maskMatrixTrg: batchTrgMask})
            ceAll += ce
            valBatch = self.valData.getValBatch()
        cePerWord = ceAll/countAll
        print("Validation Log Likelihood :" + str(cePerWord/math.log(2.0)) + " with LR="+ str(self.learningRate)+"\n")
        return cePerWord



    def computeBleu(self, trans, golden):
        return nltk.translate.bleu_score.corpus_bleu(golden, trans)


if __name__ == '__main__':

    nmtTrainer = NMT_Trainer()
    nmtTrainer.train()


