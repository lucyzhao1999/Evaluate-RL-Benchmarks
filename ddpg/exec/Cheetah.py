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
from ddpg.src.ddpg_Lucy import *


def main():
    env_name = 'HalfCheetah-v2'
    env = gym.make(env_name)

    hyperparamDict = dict()
    hyperparamDict['actorHiddenLayersWidths'] = [256] #[400, 300]
    hyperparamDict['actorActivFunction'] = [tf.nn.relu]* len(hyperparamDict['actorHiddenLayersWidths'])+ [tf.nn.tanh]
    hyperparamDict['actorHiddenLayersWeightInit'] = [tf.random_normal_initializer(0., 0.1) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorHiddenLayersBiasInit'] = [tf.constant_initializer(0.1) for units in hyperparamDict['actorHiddenLayersWidths']]
    hyperparamDict['actorOutputWeightInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    hyperparamDict['actorOutputBiasInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    hyperparamDict['actorLR'] = 1e-4

    hyperparamDict['criticHiddenLayersWidths'] = [256] #[400, 300]
    hyperparamDict['criticActivFunction'] = [tf.nn.relu]* len(hyperparamDict['criticHiddenLayersWidths'])+ [None]
    hyperparamDict['criticHiddenLayersWeightInit'] = [tf.random_normal_initializer(0., 0.1) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticHiddenLayersBiasInit'] = [tf.constant_initializer(0.1) for units in hyperparamDict['criticHiddenLayersWidths']]
    hyperparamDict['criticOutputWeightInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    hyperparamDict['criticOutputBiasInit'] = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
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

    fileName = 'HalfCheetah'
    modelDir = os.path.join(dirName, '..', 'results', 'models')
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    hyperparamDict['modelSavePathLucy'] = os.path.join(modelDir, fileName + 'Lucy')
    hyperparamDict['modelSavePathPhil'] = os.path.join(modelDir, fileName + 'Phil')
    hyperparamDict['modelSavePathMartin'] = os.path.join(modelDir, fileName + 'Martin')

    rewardDir = os.path.join(dirName, '..', 'results', 'rewards')
    if not os.path.exists(rewardDir):
        os.makedirs(rewardDir)
    hyperparamDict['rewardSavePathLucy'] = os.path.join(rewardDir, fileName + 'Lucy')
    hyperparamDict['rewardSavePathPhil'] = os.path.join(rewardDir, fileName + 'Phil')
    hyperparamDict['rewardSavePathMartin'] = os.path.join(rewardDir, fileName + 'Martin')

    lucyDDPG = LucyDDPG(hyperparamDict)
    lucyDDPG(env)



if __name__ == '__main__':
    main()