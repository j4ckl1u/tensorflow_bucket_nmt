import tensorflow as tf
import ZoneOutLSTM
import Config
import  numpy as np

class NMT_Model:

    def __init__(self):
        scope = "NMT_Model"
        with tf.variable_scope(scope):
            self.EmbSrc = tf.get_variable("SrcEmbedding", shape=[Config.VocabSize, Config.EmbeddingSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.EmbTrg = tf.get_variable("TrgEmbedding", shape=[Config.VocabSize, Config.EmbeddingSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.EncoderL2R = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize, hidden_size=Config.HiddenSize,
                                                          scope=scope+"_EncoderL2R")
            self.EncoderR2L = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize, hidden_size=Config.HiddenSize,
                                                          scope=scope+"_EncoderR2L")
            self.Decoder = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize+2*Config.HiddenSize,
                                                       hidden_size=Config.HiddenSize, scope=scope+"_Decoder")

            self.Wt = tf.get_variable("ReadoutWeight", shape=[Config.HiddenSize + Config.EmbeddingSize, Config.VocabSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.Wtb = tf.get_variable("ReadoutBias", shape=Config.VocabSize, initializer=tf.constant_initializer(0.0))

            self.WI = tf.get_variable("DecoderInitWeight",shape=(Config.HiddenSize, Config.HiddenSize*2),
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.WIb = tf.get_variable("DecoderInitBias", shape=Config.HiddenSize*2, initializer=tf.constant_initializer(0.0))

            self.Was = tf.get_variable("AttentionWeightS", shape=(Config.HiddenSize*2, Config.HiddenSize),
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.Wat = tf.get_variable("AttentionWeightT", shape=(Config.HiddenSize, Config.HiddenSize),
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.Wav = tf.get_variable("AttentionWeightV", shape=(Config.HiddenSize, 1),
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))

        self.Parameters = [self.EmbSrc, self.EmbTrg, self.Wt, self.Wtb, self.WI, self.WIb, self.Was, self.Wat, self.Wav]
        self.Parameters.extend(self.EncoderL2R.Parameters)
        self.Parameters.extend(self.EncoderR2L.Parameters)
        self.Parameters.extend(self.Decoder.Parameters)

        self.firstHidden = tf.constant(0.0, shape=[Config.BatchSize, Config.HiddenSize])
        self.initTrgEmb = tf.constant(0.0, shape=[Config.BatchSize, Config.EmbeddingSize])

    def createEncoderNetwork(self, inputSrc, maskSrc, srcLength):
        networkHiddenSrcL2R = {}
        networkHiddenSrcR2L = {}
        networkMemSrcL2R = {}
        networkMemSrcR2L = {}

        for i in range(0, srcLength, 1):

            embedI = tf.nn.embedding_lookup(self.EmbSrc, inputSrc[i])
            preL2RHidden = self.firstHidden if i == 0 else networkHiddenSrcL2R[i-1] *\
                                                           tf.reshape(maskSrc[i-1], shape=[Config.BatchSize, 1])
            preL2RMem = self.firstHidden if i == 0 else networkMemSrcL2R[i - 1] * \
                                                           tf.reshape(maskSrc[i-1], shape=[Config.BatchSize, 1])
            networkHiddenSrcL2R[i], networkMemSrcL2R[i] = self.EncoderL2R.createNetwork(embedI, preL2RHidden, preL2RMem)

            embedRI = tf.nn.embedding_lookup(self.EmbSrc, inputSrc[srcLength-i-1])
            preR2LHidden = self.firstHidden if i == 0 else networkHiddenSrcR2L[srcLength-i]* \
                                                           tf.reshape(maskSrc[srcLength-i], shape=[Config.BatchSize, 1])
            preR2LMem = self.firstHidden if i == 0 else networkMemSrcR2L[srcLength-i] * \
                                                           tf.reshape(maskSrc[srcLength-i], shape=[Config.BatchSize, 1])
            networkHiddenSrcR2L[srcLength-i-1], networkMemSrcR2L[srcLength-i-1] = self.EncoderR2L.createNetwork(embedRI, preR2LHidden, preR2LMem)

        networkHiddenSrc = []
        for i in range(0, srcLength, 1):
            networkHiddenSrc.append(tf.concat([networkHiddenSrcL2R[i], networkHiddenSrcR2L[i]], axis=-1))

        if(srcLength > 1):
            sourceHidden = tf.concat([networkHiddenSrc[0], networkHiddenSrc[1]], axis=0)
            for i in range(2, srcLength, 1):
                sourceHidden = tf.concat([sourceHidden, networkHiddenSrc[i]], axis=0)
        else:
            sourceHidden = tf.reshape(networkHiddenSrc[0], shape=(1, Config.BatchSize, Config.HiddenSize*2))
        return sourceHidden

    def createDecoderInitNetwork(self, srcSentEmb):
        WIS = tf.matmul(srcSentEmb, self.WI) + self.WIb
        initHiddenMem = tf.tanh(WIS)
        initHiddden, initMem = tf.split(initHiddenMem, 2, -1)
        return initHiddden, initMem

    def createAttentionNet(self, hiddenSrc, maskSrc, curHiddenTrg, srcLength):
        srcHiddenSize = Config.HiddenSize*2
        hsw = tf.matmul(hiddenSrc, self.Was)
        htw = tf.matmul(curHiddenTrg, self.Wat)
        hst = tf.reshape(hsw, shape=(srcLength, Config.BatchSize * Config.HiddenSize)) + \
              tf.reshape(htw, shape=(1, Config.BatchSize * Config.HiddenSize))
        hstT = tf.reshape(tf.tanh(hst), shape=(srcLength * Config.BatchSize, Config.HiddenSize))
        attScore = tf.reshape(tf.matmul(hstT, self.Wav), shape=(srcLength, Config.BatchSize))
        maskOut = (maskSrc - tf.ones(tf.shape(maskSrc)))*99999999
        nAttScore = attScore + tf.slice(maskOut, [0,0], [srcLength, Config.BatchSize])
        attProb = tf.reshape(tf.nn.softmax(nAttScore, dim=0), shape=(srcLength, Config.BatchSize, 1))
        attVector =tf.reshape(hiddenSrc, shape=(srcLength, Config.BatchSize, srcHiddenSize))*attProb
        contextVector =tf.reduce_sum(tf.reshape(attVector, shape=(srcLength, Config.BatchSize * srcHiddenSize)), axis=0)
        contextVector= tf.reshape(contextVector, shape=(Config.BatchSize, srcHiddenSize))

        return (contextVector, attProb)

    def createDecoderRNNNetwork(self, srcHiddenStates, maskSrc, preTrgEmb, preHidden, preMem, srcLength):
        (contextVect, attProb) = self.createAttentionNet(srcHiddenStates, maskSrc, preHidden, srcLength)
        curInput = tf.concat([contextVect, preTrgEmb], axis=-1)
        networkHiddenTrg, networkMemTrg = self.Decoder.createNetwork(curInput, preHidden, preMem)
        return (networkHiddenTrg, networkMemTrg, attProb)

    def createDecoderNetwork(self, networkHiddenSrc, inputTrg, maskSrc, maskTrg, srcLength, trgLength, optimizer):
        timeZeroHidden = tf.slice(networkHiddenSrc, [0, 0], [Config.BatchSize, Config.HiddenSize*2])
        srcSentEmb = tf.slice(timeZeroHidden, [0, Config.HiddenSize], [Config.BatchSize, Config.HiddenSize])
        networkHiddenTrg = {}
        networkMemTrg = {}
        attProbAll=[]
        tce = 0
        for i in range(0, trgLength, 1):
            
            preTrgEmb = self.initTrgEmb if i==0 else tf.nn.embedding_lookup(self.EmbTrg, inputTrg[i-1])
            
            if (i == 0):
                networkHiddenTrg[i], networkMemTrg[i] = self.createDecoderInitNetwork(srcSentEmb)
            else:
                (networkHiddenTrg[i], networkMemTrg[i], attProb) = self.createDecoderRNNNetwork(
                    networkHiddenSrc, maskSrc, preTrgEmb, networkHiddenTrg[i - 1], networkMemTrg[i-1], srcLength)
                attProbAll = attProb if i == 1 else tf.concat([attProbAll, attProb], axis=0)

            logits_out = self.createReadOutNetwork(networkHiddenTrg[i],  preTrgEmb)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=inputTrg[i])
            tce += tf.reduce_sum(ce * maskTrg[i])

        totalCount = tf.reduce_sum(maskTrg)
        tce = tce / totalCount
        min_loss = optimizer.minimize(tce)
        return min_loss, tce

    def createReadOutNetwork(self, decoderHidden, preTrgEmb):
        readOut = tf.concat([decoderHidden, preTrgEmb], axis=-1)
        preSoftmax = tf.matmul(readOut, self.Wt) + self.Wtb
        return preSoftmax

    def createTrainingNetwork(self, inputSrc, maskSrc, inputTrg, maskTrg, srcLength, trgLength, optimizer):
        encoderNet = self.createEncoderNetwork(inputSrc, maskSrc, srcLength)
        decoderNet = self.createDecoderNetwork(encoderNet, inputTrg, maskSrc, maskTrg, srcLength, trgLength, optimizer)
        return decoderNet

    def createPredictionNetwork(self, preSoftmax):
        nextWordProb = tf.softmax(preSoftmax)
        bestTrans = tf.reshape(tf.argmax(nextWordProb, -1), shape=Config.BatchSize)
        return bestTrans

    def createDecodingInitNetwork(self, srcSentEmb):
        decoderInitHidden = self.createDecoderInitNetwork(srcSentEmb)
        preSoftmax = self.createReadOutNetwork(decoderInitHidden, self.initTrgEmb)
        decoderInitPredict = self.createPredictionNetwork(preSoftmax)
        decoderInitPredictNet= tf.group(decoderInitHidden, decoderInitPredict)
        return (decoderInitPredictNet, [decoderInitHidden.output, decoderInitPredict.output])

    def createDecodingNetworks(self, srcHiddenStates, trgPreWord, trgPreHidden, trgPreMem, srcLength):
        preTrgEmb = self.EmbTrg(trgPreWord)
        (decoderHidden, attProb) = self.createDecoderRNNNetwork(
            tf.slice(srcHiddenStates, [0, 0], [srcLength, Config.HiddenSize*2]), preTrgEmb, trgPreHidden, trgPreMem, srcLength)
        preSoftmax = self.createReadOutNetwork(decoderHidden, preTrgEmb)
        decoderPredict = self.createPredictionNetwork(preSoftmax)
        decoderPredictNet=tf.group(decoderHidden, decoderPredict)
        return (decoderPredictNet, [decoderHidden.output, decoderPredict.output])
