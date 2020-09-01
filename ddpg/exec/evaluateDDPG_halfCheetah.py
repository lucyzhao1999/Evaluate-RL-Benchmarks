import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import gym
from functionTools.loadSaveModel import loadFromPickle, restoreVariables, saveToPickle
import json
import ddpg.src.ddpg_Lucy as LucyDDPG
import ddpg.src.ddpg_martin as MartinDDPG
import ddpg.src.ddpg_Phil as PhilDDPG
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class EvaluateDDPG:
    def __init__(self, hyperparamDict, env, fileName):
        self.hyperparamDict = hyperparamDict
        self.fileName = fileName
        self.env = env

    def __call__(self, df):
        person = df.index.get_level_values('person')[0]
        seed = df.index.get_level_values('seed')[0]
        epsID = df.index.get_level_values('index')[0]
        tf.reset_default_graph()

        modelDir = os.path.join(dirName, '..', 'results', 'models')

        if person == 'Lucy':
            modelPath = os.path.join(modelDir, self.fileName + '_Lucy_seed' + str(seed) + '_') + str(epsID) + "eps"
            meanTrajReward, seTrajReward = LucyDDPG.getModelEvalResult(self.env, self.hyperparamDict, modelPath)
        elif person == 'Martin':
            modelPath = os.path.join(modelDir, self.fileName + '_Martin_seed' + str(seed) + '_') + str(epsID) + "eps"
            meanTrajReward, seTrajReward = MartinDDPG.getModelEvalResult(self.env, self.hyperparamDict, modelPath)
        else:
            modelPath = os.path.join(modelDir, self.fileName + '_Phil_seed' + str(seed) + '_') + str(epsID) + "eps"
            meanTrajReward, seTrajReward = PhilDDPG.getModelEvalResult(self.env, self.hyperparamDict, modelPath)

        return pd.Series({'meanReward': meanTrajReward})


def main():
    env_name = 'HalfCheetah-v2'
    fileName = 'HalfCheetah'

    hyperparamDict = dict()
    hyperparamDict['actorHiddenLayersWidths'] = [256, 256] #[400, 300]
    hyperparamDict['actorActivFunction'] = [tf.nn.relu]* len(hyperparamDict['actorHiddenLayersWidths'])+ [tf.nn.tanh]
    hyperparamDict['actorHiddenLayersWeightInit'] = [tf.random_normal_initializer(0., 0.1) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorHiddenLayersBiasInit'] = [tf.constant_initializer(0.1) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorOutputWeightInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['actorOutputBiasInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['actorLR'] = 1e-4

    hyperparamDict['criticHiddenLayersWidths'] = [256, 256] #[400, 300]
    hyperparamDict['criticActivFunction'] = [tf.nn.relu]* len(hyperparamDict['criticHiddenLayersWidths'])+ [None]
    hyperparamDict['criticHiddenLayersWeightInit'] = [tf.random_normal_initializer(0., 0.1) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticHiddenLayersBiasInit'] = [tf.constant_initializer(0.1) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticOutputWeightInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['criticOutputBiasInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['criticLR'] = 1e-3

    hyperparamDict['tau'] = 0.001
    hyperparamDict['gamma'] = 0.99
    hyperparamDict['minibatchSize'] = 128

    hyperparamDict['gradNormClipValue'] = None
    hyperparamDict['maxEpisode'] = 1000
    hyperparamDict['maxTimeStep'] = 1000
    hyperparamDict['bufferSize'] = 1e6

    hyperparamDict['noiseInitVariance'] = 2
    hyperparamDict['varianceDiscount'] = 1e-5
    hyperparamDict['noiseDecayStartStep'] = 10000
    hyperparamDict['minVar'] = .1
    hyperparamDict['normalizeEnv'] = False
    hyperparamDict['modelSaveRate'] = 10 #eps

    env = gym.make(env_name)

    independentVariables = dict()
    independentVariables['person'] = ['Lucy', 'Martin']
    independentVariables['seed'] = [1, 2, 3, 4, 5]
    independentVariables['timeStep'] = list(range(10, 1010, 10))

    evaluate = EvaluateDDPG(hyperparamDict, env, fileName)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluate)

    evalResultDir = os.path.join(dirName, '..', 'results', 'evalAll')
    if not os.path.exists(evalResultDir):
        os.makedirs(evalResultDir)
    resultLoc = os.path.join(evalResultDir, fileName + '.pkl')
    saveToPickle(resultDF, resultLoc)

    # resultDF = loadFromPickle(resultLoc)
    print(resultDF)

    ax = sns.lineplot(x="timeStep", y="meanReward", hue="person", style="person", ci='sd', data=resultDF.reset_index())
    plt.savefig(os.path.join(evalResultDir, fileName))
    plt.show()


if __name__ == '__main__':
    main()