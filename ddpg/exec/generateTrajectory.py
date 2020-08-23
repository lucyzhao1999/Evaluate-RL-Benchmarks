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
from functionTools.loadSaveModel import saveVariables
import tensorflow as tf

from ddpg.src.ddpg_Lucy import GetActorNetwork, Actor, GetCriticNetwork, Critic, SaveModel
from functionTools.loadSaveModel import restoreVariables

def main():
    env_name = 'Ant-v2'
    env = gym.make(env_name)

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

    actionHigh = env.action_space.high
    actionLow = env.action_space.low
    actionBound = (actionHigh - actionLow) / 2
    tf.reset_default_graph()

    session = tf.Session()
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    getActorNetwork = GetActorNetwork(hyperparamDict, batchNorm=True)
    actor = Actor(getActorNetwork, stateDim, actionDim, session, hyperparamDict, agentID=None, actionRange=actionBound)

    getCriticNetwork = GetCriticNetwork(hyperparamDict, addActionToLastLayer=True, batchNorm=True)
    critic = Critic(getCriticNetwork, stateDim, actionDim, session, hyperparamDict)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    session.run(tf.global_variables_initializer())

    fileName = 'Ant_'
    modelPath = os.path.join(dirName, '..', 'results', 'models', fileName+'Lucy')

    restoreVariables(session, modelPath)

    for i in range(50):
        state = env.reset()
        epsReward = 0
        for timestep in range(hyperparamDict['maxTimeStep']):
            env.render()
            state = state.reshape(1, -1)
            action = actor.actByTrain(state)
            nextState, reward, done, info = env.step(action)
            epsReward+= reward
            if done:
                break
            print(epsReward)
            state = nextState



if __name__ == '__main__':
    main()