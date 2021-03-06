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
import pandas as pd

from functionTools.loadSaveModel import loadFromPickle
from ddpg.src.ddpg_Lucy import *
from ddpg.src.ddpg_martin import *
from ddpg.src.ddpg_Phil import *

class EvaluateDDPG:
    def __init__(self, hyperparamDict):
        self.hyperparamDict = hyperparamDict

    def __call__(self, df):
        person = df.index.get_level_values('person')[0]
        # env_name = 'HalfCheetah-v2'
        # env = gym.make(env_name)
        # tf.reset_default_graph()

        if person == 'Lucy':
            meanRewardList = loadFromPickle(self.hyperparamDict['rewardSavePathLucy'])
        elif person == 'Martin':
            meanRewardList = loadFromPickle(self.hyperparamDict['rewardSavePathMartin'])
        else:
            meanRewardList = loadFromPickle(self.hyperparamDict['rewardSavePathPhil'])

        print(len(meanRewardList))
        timeStep = list(range(len(meanRewardList)))
        resultSe = pd.Series({time: reward for time, reward in zip(timeStep, meanRewardList)})

        return resultSe


def main():
    fileName = 'mountainCarContinuous'

    hyperparamDict = dict()
    hyperparamDict['actorHiddenLayersWidths'] = [256] #[400, 300]
    hyperparamDict['actorActivFunction'] = [tf.nn.relu]* len(hyperparamDict['actorHiddenLayersWidths'])+ [tf.nn.tanh]
    hyperparamDict['actorHiddenLayersWeightInit'] = [tf.random_normal_initializer(0., 0.1) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorHiddenLayersBiasInit'] = [tf.constant_initializer(0.1) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorOutputWeightInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['actorOutputBiasInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['actorLR'] = 1e-4

    hyperparamDict['criticHiddenLayersWidths'] = [256] #[400, 300]
    hyperparamDict['criticActivFunction'] = [tf.nn.relu]* len(hyperparamDict['criticHiddenLayersWidths'])+ [None]
    hyperparamDict['criticHiddenLayersWeightInit'] = [tf.random_normal_initializer(0., 0.1) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticHiddenLayersBiasInit'] = [tf.constant_initializer(0.1) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticOutputWeightInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['criticOutputBiasInit'] = tf.random_normal_initializer(0., 0.1)
    hyperparamDict['criticLR'] = 1e-3

    hyperparamDict['tau'] = 0.001
    hyperparamDict['gamma'] = 0.99
    hyperparamDict['minibatchSize'] = 64

    hyperparamDict['gradNormClipValue'] = None
    hyperparamDict['maxEpisode'] = 2000
    hyperparamDict['maxTimeStep'] = 1000
    hyperparamDict['bufferSize'] = 500000

    hyperparamDict['noiseInitVariance'] = 2
    hyperparamDict['varianceDiscount'] = 1e-5
    hyperparamDict['noiseDecayStartStep'] = hyperparamDict['bufferSize']
    hyperparamDict['minVar'] = .001
    hyperparamDict['normalizeEnv'] = False

    modelDir = os.path.join(dirName, '..', 'results', 'models')
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    hyperparamDict['modelSavePathLucy'] = os.path.join(modelDir, fileName + '_Lucy')
    hyperparamDict['modelSavePathPhil'] = os.path.join(modelDir, fileName + '_Phil')
    hyperparamDict['modelSavePathMartin'] = os.path.join(modelDir, fileName + '_Martin')

    rewardDir = os.path.join(dirName, '..', 'results', 'rewards')
    if not os.path.exists(rewardDir):
        os.makedirs(rewardDir)
    hyperparamDict['rewardSavePathLucy'] = os.path.join(rewardDir, fileName + '_Lucy')
    hyperparamDict['rewardSavePathPhil'] = os.path.join(rewardDir, fileName + '_Phil')
    hyperparamDict['rewardSavePathMartin'] = os.path.join(rewardDir, fileName + '_Martin')

    independentVariables = dict()
    independentVariables['person'] = ['Lucy', 'Phil', 'Martin']
    evaluateWolfSheepTrain = EvaluateDDPG(hyperparamDict)

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(evaluateWolfSheepTrain)

    evalResultDir = os.path.join(dirName, '..', 'results', 'eval')
    if not os.path.exists(evalResultDir):
        os.makedirs(evalResultDir)
    resultLoc = os.path.join(evalResultDir, fileName + '.pkl')
    saveToPickle(resultDF, resultLoc)

    # resultDF = loadFromPickle(resultLoc)
    print(resultDF)

    resultDF.T.plot.line()
    plt.savefig(os.path.join(evalResultDir, fileName))
    plt.show()


if __name__ == '__main__':
    main()