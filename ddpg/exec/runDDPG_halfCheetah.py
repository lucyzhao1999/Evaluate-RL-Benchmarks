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
import json

def main():
    debug = 1
    if debug:
        person = 'Phil'
        seed = 4
    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        person = condition['person']
        seed = int(condition['seed'])

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

    modelDir = os.path.join(dirName, '..', 'results', 'models')
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    hyperparamDict['modelSavePathLucy'] = os.path.join(modelDir, fileName + '_Lucy_seed'+ str(seed) + '_')
    hyperparamDict['modelSavePathPhil'] = os.path.join(modelDir, fileName + '_Phil_seed'+ str(seed)+ '_')
    hyperparamDict['modelSavePathMartin'] = os.path.join(modelDir, fileName + '_Martin_seed'+ str(seed)+ '_')

    rewardDir = os.path.join(dirName, '..', 'results', 'rewards')
    if not os.path.exists(rewardDir):
        os.makedirs(rewardDir)
    hyperparamDict['rewardSavePathLucy'] = os.path.join(rewardDir, fileName + '_Lucy'+ str(seed))
    hyperparamDict['rewardSavePathPhil'] = os.path.join(rewardDir, fileName + '_Phil'+ str(seed))
    hyperparamDict['rewardSavePathMartin'] = os.path.join(rewardDir, fileName + '_Martin'+ str(seed))

    env = gym.make(env_name)

    if person == 'Lucy':
        ddpgModel = LucyDDPG(hyperparamDict)
    elif person == 'Martin':
        ddpgModel = MartinDDPG(hyperparamDict)
    else:
        ddpgModel = PhilDDPG(hyperparamDict)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    ddpgModel(env)


if __name__ == '__main__':
    main()