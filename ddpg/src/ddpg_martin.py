"""
Created on Tue Jun 16 06:25:29 2020

@author: martin
"""
import tensorflow as tf
import numpy as np 
import random
import gym
import matplotlib.pyplot as plt
from collections import deque
from functionTools.loadSaveModel import saveToPickle, saveVariables

def samplebuffer(replaybuffer,batchsize):
    minibatch = [list(Batch) for Batch in zip(*random.sample(replaybuffer, batchsize))]
    return minibatch

def memory(replayBuffer, states, action, nextStates, reward):
    replayBuffer.append((states, action ,nextStates,reward))
    return replayBuffer

def fillbuffer(initbuffer,bufferfill,env,replaybuffer,state):
    while bufferfill < initbuffer:
        action = env.action_space.sample()
        next_state, reward,done,_  = env.step(action)
        if done:
            state = env.reset()
        else:
            bufferfill += 1
            memory(replaybuffer,state,action,next_state,reward)
            state = next_state
    return replaybuffer
            


def ReplaceParameters(model):
    modelgraph = model.graph
    replaceParam_ = modelgraph.get_collection_ref("ReplaceTargetParam_")[0]
    model.run(replaceParam_)
    print('replace parameter')
    return model



class MartinDDPG:
    def __init__(self,fixedParameters):   
        self.fixedParameters = fixedParameters
        self.bufferfill = 0
        self.runstep = 0

    def __call__(self,env):
        env = env_norm(env) if self.fixedParameters['normalizeEnv'] else env
        actionHigh = env.action_space.high
        actionLow = env.action_space.low
        actionBound = (actionHigh - actionLow) / 2
        stateDim = env.observation_space.shape[0]
        actionDim = env.action_space.shape[0]
        meanreward = []
        trajectory = []
        totalreward = []
        totalrewards = []
        replaybuffer = deque(maxlen=int(self.fixedParameters['bufferSize']))
        buildActorModel = BuildActorModel(stateDim, actionDim,actionBound ,
                                          self.fixedParameters['actorHiddenLayersWeightInit'],self.fixedParameters['actorHiddenLayersBiasInit'],
                                          self.fixedParameters['actorOutputWeightInit'], self.fixedParameters['actorOutputBiasInit'],self.fixedParameters['actorActivFunction'])
        actorModel = buildActorModel(self.fixedParameters['actorHiddenLayersWidths'])
    
        buildCriticModel = BuildCriticModel(stateDim, actionDim,
                                            self.fixedParameters['criticHiddenLayersWeightInit'],self.fixedParameters['criticHiddenLayersBiasInit'],
                                            self.fixedParameters['criticOutputWeightInit'], self.fixedParameters['criticOutputBiasInit'],
                                            self.fixedParameters['criticActivFunction'])
        criticModel = buildCriticModel(self.fixedParameters['criticHiddenLayersWidths'])
        
        trainCritic = TrainCritic(self.fixedParameters['criticlearningRate'], self.fixedParameters['gamma'])
        trainActor = TrainActor(self.fixedParameters['actorlearningRate'])
        updateParameters = UpdateParameters(self.fixedParameters['tau'])
        trainddpgModels = TrainDDPGModels(updateParameters, trainActor, trainCritic, actorModel,criticModel)
    
        getnoise = GetNoise(self.fixedParameters['varianceDiscount'],self.fixedParameters['minVar'],self.fixedParameters['noiseDecayStartStep'],self.fixedParameters['initnoisevar'])
        getnoiseaction = GetNoiseAction(actorModel,actionLow, actionHigh)
        learn = Learn(self.fixedParameters['buffersize'],self.fixedParameters['minibatchSize'],trainddpgModels)
        actorModel= ReplaceParameters(actorModel)
        criticModel= ReplaceParameters(criticModel)
        state  = env.reset()
        replaybuffer = fillbuffer(self.fixedParameters['initbuffer'],self.bufferfill,env,replaybuffer,state)
        
        for episode in range(1,self.fixedParameters['EPISODE']+1):
            state  = env.reset()
            rewards = 0
            for j in range(self.fixedParameters['maxTimeStep']):
                env.render()
                noise = getnoise(self.runstep)
                noiseaction = getnoiseaction(state,noise)
                nextstate,reward,done,info = env.step(noiseaction)
                learn(replaybuffer,state, noiseaction, nextstate,reward)
                trajectory.append((state, noiseaction, nextstate,reward))
                rewards += reward
                state = nextstate
                self.runstep += 1
                if j == self.fixedParameters['maxTimeStep']-1:
                    totalrewards.append(rewards)
                    totalreward.append(rewards)
                    print('episode: ',episode,'reward:',rewards,'runstep',self.runstep)
            if episode % 100 == 0:
                meanreward.append(np.mean(totalreward))
                print('episode: ',episode,'meanreward:',np.mean(totalreward))
                self.totalreward = []
        episodes = 100*(np.arange(len(meanreward)))

        with actorModel.as_default():
            saveVariables(actorModel, self.fixedParameters['modelSavePathMartin'])
        with criticModel.as_default():
            saveVariables(criticModel, self.fixedParameters['modelSavePathMartin'])
        saveToPickle(meanreward, self.fixedParameters['rewardSavePathMartin'])
        plt.plot(episodes,meanreward)
        plt.xlabel('episode')
        plt.ylabel('rewards')

        return meanreward
        


                
        
class UpdateParameters:
    def __init__(self, tau):
        self.tau = tau
    def __call__(self, model):
        modelgraph = model.graph
        updateParam_ = modelgraph.get_collection_ref("updateParam_")[0]
        tau_ = modelgraph.get_collection_ref("tau_")[0]
        model.run(updateParam_,feed_dict={tau_: self.tau})
        return model
    

class BuildActorModel ():
    def __init__(self, statedim, actiondim,actionbound,actorHiddenLayersInitinitweight,actorHiddenLayersinitbias,actorOutputInitinitweight,actorOutputinitbias,actorActivFunction):
        self.stateDim = statedim
        self.actionDim = actiondim
        self.actionbound = actionbound
        self.actorHiddenLayersInitinitweight = actorHiddenLayersInitinitweight
        self.actorHiddenLayersinitbias = actorHiddenLayersinitbias
        self.actorOutputInitinitweight = actorOutputInitinitweight
        self.actorOutputinitbias = actorOutputinitbias
        self.actorActivFunction = actorActivFunction
        
    def __call__(self, numberlayers):
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.stateDim], 'states_')
                nextstates_ = tf.placeholder(tf.float32, [None, self.stateDim],'nextstates_')
                actionGradients_ = tf.placeholder(tf.float32, [None, self.actionDim], 'actionGradients_')
                
                tf.add_to_collection('states_', states_)
                tf.add_to_collection('nextstates_', nextstates_)
                tf.add_to_collection("actionGradients_", actionGradients_)

                
            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)
                
            with tf.variable_scope('evalnet'):
                activation_ = states_
                for unit in range(len(numberlayers)):
                    actorhiddenActivFunction = self.actorActivFunction[unit]
                    activation_ = tf.layers.dense(activation_, units = numberlayers[unit], kernel_initializer=self.actorHiddenLayersInitinitweight[unit],activation = actorhiddenActivFunction, 
                                         bias_initializer=self.actorHiddenLayersinitbias[unit], name="hidden" + str(unit + 1),trainable = True)
                    activation_ = tf.layers.batch_normalization(activation_)
                actoroutputActivFunction = self.actorActivFunction[-1]
                Qevalvalue_ = tf.layers.dense(activation_, self.actionDim,actoroutputActivFunction, kernel_initializer=self.actorOutputInitinitweight,
                                      bias_initializer=self.actorOutputinitbias, name='e2',trainable = True)
                
                tf.add_to_collection('Qevalvalue_', Qevalvalue_)
                    
            with tf.variable_scope('targetnet'):
                activation_ = nextstates_
                for unit in range(len(numberlayers)):
                    actorhiddenActivFunction = self.actorActivFunction[unit]
                    activation_ = tf.layers.dense(activation_, units = numberlayers[unit], kernel_initializer=self.actorHiddenLayersInitinitweight[unit],activation = actorhiddenActivFunction, 
                                         bias_initializer=self.actorHiddenLayersinitbias[unit], name="hidden" + str(unit + 1),trainable = False)
                    activation_ = tf.layers.batch_normalization(activation_)
                actoroutputActivFunction = self.actorActivFunction[-1]
                Qnext = tf.layers.dense(activation_, self.actionDim, actoroutputActivFunction,kernel_initializer=self.actorOutputInitinitweight,
                                      bias_initializer=self.actorOutputinitbias, name='t2',trainable = False)
                
                tf.add_to_collection('Qnext', Qnext)   
                
            with tf.name_scope("replaceParameters"):
                evalParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evalnet')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetnet')
                updateParam_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * evalParams_[i]) for i in range(len(targetParams_))]
                ReplaceTargetParam_ = [tf.assign(targetParams_, evalParams_) for targetParams_, evalParams_ in zip(targetParams_, evalParams_)]
                tf.add_to_collection("evalParams_", evalParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_) 
                tf.add_to_collection("ReplaceTargetParam_", ReplaceTargetParam_) 
            
            with tf.variable_scope('Qaction'):
                evalAction_ = tf.multiply(Qevalvalue_, self.actionbound, name='evalAction_')
                targetAction_ = tf.multiply(Qnext, self.actionbound, name='targetAction_')
                policyGradient_ = tf.gradients(evalAction_, evalParams_, actionGradients_)
                tf.add_to_collection("evalAction_", evalAction_)
                tf.add_to_collection("targetAction_", targetAction_)
                tf.add_to_collection("policyGradient_", policyGradient_)
                
            with tf.variable_scope('train'):
                optimizer = tf.train.AdamOptimizer(-learningRate_, name='adamOptimizer')
                trainOpt_ = optimizer.apply_gradients(zip(policyGradient_, evalParams_))
                tf.add_to_collection("trainOpt_", trainOpt_)
                
            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)
            
            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)
            
            model = tf.Session(graph = graph)
            model.run(tf.global_variables_initializer())

        return model

class BuildCriticModel ():
    def __init__(self, statedim, actiondim,criticHiddenLayersInitinitweight,criticHiddenLayersinitbias,criticOutputInitinitweight,criticOutputinitbias,criticActivFunction):
        self.stateDim = statedim
        self.actionDim = actiondim
        self.criticHiddenLayersInitinitweight = criticHiddenLayersInitinitweight
        self.criticHiddenLayersinitbias = criticHiddenLayersinitbias
        self.criticOutputInitinitweight = criticOutputInitinitweight
        self.criticOutputinitbias = criticOutputinitbias
        self.criticActivFunction = criticActivFunction
    def __call__(self, numberlayers):
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.stateDim])
                action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.actionDim]))
                nextstates_ = tf.placeholder(tf.float32, [None, self.stateDim])
                reward_ = tf.placeholder(tf.float32, [None, 1], 'reward_')
                targetaction = tf.placeholder(tf.float32, [None, self.actionDim])
                Qtarget_ = tf.placeholder(tf.float32, [None, 1], name="Qtarget_")
                tf.add_to_collection('states_', states_)
                tf.add_to_collection('nextstates_', nextstates_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("action_", action_)
                tf.add_to_collection("targetaction", targetaction)
                tf.add_to_collection("Qtarget_", Qtarget_)
                
            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                tau_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)
                
                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("tau_", tau_)
                tf.add_to_collection("gamma_", gamma_)
                
            with tf.variable_scope('evalnet'):
                activation_ = states_
                for unit in range(len(numberlayers)):
                    critichiddenActivFunction = self.criticActivFunction[unit]
                    activation_ = tf.layers.dense(activation_, units = numberlayers[unit], kernel_initializer=self.criticHiddenLayersInitinitweight[unit],activation = critichiddenActivFunction, 
                                         bias_initializer=self.criticHiddenLayersinitbias[unit], name="hidden" + str(unit + 1),trainable = True)
                    activation_ = tf.layers.batch_normalization(activation_)
                criticoutputActivFunction = self.criticActivFunction[-2]
                secondnumlayers = numberlayers[-2] if len(numberlayers) >= 2 else self.stateDim
                evalstateweight_ = tf.get_variable('evalstateweight_', [secondnumlayers, numberlayers[-1]])
                evalactionweight_ = tf.get_variable('evalactionweight_', [self.actionDim, numberlayers[-1]])
                evalbias1 = tf.get_variable(name='evalbias1', shape=[numberlayers[-1]], initializer=self.criticHiddenLayersinitbias[-1])
                evalactivation = criticoutputActivFunction(tf.matmul(activation_, evalstateweight_) + tf.matmul(action_, evalactionweight_) + evalbias1)
                Qevalvalue_ = tf.layers.dense(evalactivation,1, activation = self.criticActivFunction[-1], kernel_initializer=self.criticOutputInitinitweight, bias_initializer=self.criticOutputinitbias,  trainable = True)

                tf.add_to_collection('evalstateweight_', evalstateweight_) 
                tf.add_to_collection('evalactionweight_', evalactionweight_) 
                tf.add_to_collection('evalbias1', evalbias1) 
                tf.add_to_collection('evalactivation', evalactivation) 
                tf.add_to_collection('Qevalvalue_', Qevalvalue_) 
                
            with tf.variable_scope('targetnet'):
                activation_ = nextstates_
                for unit in range(len(numberlayers)):
                    critichiddenActivFunction = self.criticActivFunction[unit]
                    activation_ = tf.layers.dense(activation_, units = numberlayers[unit], kernel_initializer=self.criticHiddenLayersInitinitweight[unit],activation = critichiddenActivFunction, 
                                         bias_initializer=self.criticHiddenLayersinitbias[unit], name="hidden" + str(unit + 1),trainable = True)
                    activation_ = tf.layers.batch_normalization(activation_)
                criticoutputActivFunction = self.criticActivFunction[-2]
                secondnumlayers = numberlayers[-2] if len(numberlayers) >= 2 else self.stateDim
                targetstateweight_ = tf.get_variable('targetstateweight_', [secondnumlayers, numberlayers[-1]])
                targetactionweight_ = tf.get_variable('targetactionweight_', [self.actionDim, numberlayers[-1]])
                targetbias1 = tf.get_variable(name='targetbias1', shape=[numberlayers[-1]], initializer=self.criticHiddenLayersinitbias[-1])
                targetactivation = criticoutputActivFunction(tf.matmul(activation_, targetstateweight_) + tf.matmul(targetaction, targetactionweight_) + targetbias1)
                Qnextvalue_ = tf.layers.dense(targetactivation,1,activation = self.criticActivFunction[-1], kernel_initializer=self.criticOutputInitinitweight, bias_initializer=self.criticOutputinitbias, trainable = False)
                
                tf.add_to_collection('targetstateweight_', targetstateweight_) 
                tf.add_to_collection('targetactionweight_', targetactionweight_) 
                tf.add_to_collection('targetbias1', targetbias1) 
                tf.add_to_collection('targetactivation', targetactivation) 
                tf.add_to_collection('Qnextvalue_', Qnextvalue_) 
                
                
            with tf.name_scope("replaceParameters"):
                evalParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evalnet')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetnet')
                updateParam_ = [targetParams_[i].assign((1 - tau_) * targetParams_[i] + tau_ * evalParams_[i]) for i in range(len(targetParams_))]
                ReplaceTargetParam_ = [tf.assign(targetParams_, evalParams_) for targetParams_, evalParams_ in zip(targetParams_, evalParams_)]
                tf.add_to_collection("evalParams_", evalParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("updateParam_", updateParam_) 
                tf.add_to_collection("ReplaceTargetParam_", ReplaceTargetParam_) 
            
            with tf.name_scope("actionGradients"):
                actionGradients_ = tf.gradients(Qevalvalue_, action_)[0]
                tf.add_to_collection("actionGradients_", actionGradients_)
            
            with tf.name_scope("loss"):
                yi_ = reward_ + gamma_ * Qtarget_
                loss_ = tf.losses.mean_squared_error(labels=yi_, predictions=tf.multiply(Qevalvalue_, 1))
                tf.add_to_collection("loss_", loss_)
            
            with tf.variable_scope('train'):
                trainopt_ = tf.train.AdamOptimizer(learningRate_, name='AdamOptimizer').minimize(loss_, var_list=evalParams_)
                tf.add_to_collection("trainopt_", trainopt_)
                
            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)
            
            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)
            
            model = tf.Session(graph = graph)
            model.run(tf.global_variables_initializer())

        return model
    
def getevalaction(actorModel, stateBatch):
    modelgraph = actorModel.graph
    states_ = modelgraph.get_collection_ref("states_")[0]
    evalAction_ = modelgraph.get_collection_ref("evalAction_")[0]
    evalAction_ = actorModel.run(evalAction_, feed_dict={states_: stateBatch})
    return evalAction_

def gettargetaction(actorModel, nextStatesBatch):
    modelgraph = actorModel.graph
    nextstates_ = modelgraph.get_collection_ref("nextstates_")[0]
    targetAction_ = modelgraph.get_collection_ref("targetAction_")[0]
    targetAction_ = actorModel.run(targetAction_, feed_dict={nextstates_: nextStatesBatch})
    return targetAction_

def getQtarget(nextStatesBatch, targetactionsBatch, criticModel):
    modelgraph = criticModel.graph
    Qnextvalue_ = modelgraph.get_collection_ref('Qnextvalue_')[0]
    nextstates_ = modelgraph.get_collection_ref("nextstates_")[0]
    targetaction_ = modelgraph.get_collection_ref("targetaction")[0]
    Qtargetvalue_ = criticModel.run(Qnextvalue_, feed_dict={nextstates_: nextStatesBatch,targetaction_: targetactionsBatch})
    return Qtargetvalue_


def getActionGradients(criticModel, stateBatch, actionsBatch):
    criticGraph = criticModel.graph
    states_ = criticGraph.get_collection_ref("states_")[0]
    action_ = criticGraph.get_collection_ref("action_")[0]
    actionGradients_ = criticGraph.get_collection_ref("actionGradients_")[0]
    actionGradients = criticModel.run(actionGradients_, feed_dict={states_: stateBatch,
                                                                   action_: actionsBatch})
    return actionGradients

class TrainCritic:
    def __init__(self, criticlearningRate, gamma):
        self.criticlearningRate = criticlearningRate
        self.gamma = gamma

    def __call__(self, actormodel,criticmodel, minibatch,batchSize):
        modelGraph = criticmodel.graph
        states_ = modelGraph.get_collection_ref("states_")[0]
        action_ = modelGraph.get_collection_ref("action_")[0]
        nextstates_ = modelGraph.get_collection_ref("nextstates_")[0]
        reward_ = modelGraph.get_collection_ref("reward_")[0]
        Qtarget_ = modelGraph.get_collection_ref("Qtarget_")[0]
        
        learningRate_ = modelGraph.get_collection_ref("learningRate_")[0]
        gamma_ = modelGraph.get_collection_ref("gamma_")[0]

        loss_ =  modelGraph.get_collection_ref("loss_")[0]
        trainopt_ = modelGraph.get_collection_ref("trainopt_")[0]
        
        states, actions, nextStates, rewards = minibatch
        statebatch = np.asarray(states).reshape(batchSize, -1)
        actionbatch = np.asarray(actions).reshape(batchSize, -1)
        nextStatebatch = np.asarray(nextStates).reshape(batchSize, -1)
        rewardbatch = np.asarray(rewards).reshape(batchSize, -1)
        

        targetActionBatch = gettargetaction(actormodel, nextStatebatch)
        QtargetBatch = getQtarget(nextStatebatch, targetActionBatch, criticmodel)
        
        criticmodel.run([trainopt_,loss_],feed_dict={states_: statebatch,  action_: actionbatch,nextstates_: nextStatebatch,reward_:rewardbatch,
                                 learningRate_: self.criticlearningRate, gamma_:self.gamma, Qtarget_: QtargetBatch})
        
        summary = tf.Summary()
        summary.value.add(tag='reward', simple_value=float(np.mean(rewardbatch)))

        return criticmodel
    
class TrainActor:
    def __init__(self, actorLearningRate):
        self.actorLearningRate = actorLearningRate

    def __call__(self, actorModel ,criticModel, minibatch,batchSize):
        actorGraph = actorModel.graph
        states_ = actorGraph.get_collection_ref("states_")[0]
        
        actionGradients_ = actorGraph.get_collection_ref("actionGradients_")[0]
        learningRate_ = actorGraph.get_collection_ref("learningRate_")[0]
        
        states, actions, nextStates, rewards = minibatch
        statebatch = np.asarray(states).reshape(batchSize, -1)
        
        evalactionbatch = getevalaction(actorModel,statebatch)
        actionGradients = getActionGradients(criticModel, statebatch, evalactionbatch)
        
        trainOpt_ = actorGraph.get_collection_ref("trainOpt_")[0]
        actorModel.run([trainOpt_], feed_dict={states_: statebatch,actionGradients_: actionGradients,
                                                        learningRate_: self.actorLearningRate})
        return actorModel


class TrainDDPGModels:
    def __init__(self, updateParameters, trainActor, trainCritic, actorModel, criticModel):
        self.updateParameters = updateParameters
        self.trainActor = trainActor
        self.trainCritic = trainCritic
        self.actorModel = actorModel
        self.criticModel = criticModel
        
    def __call__(self, miniBatch,batchsize):
        self.actorModel = self.trainActor(self.actorModel, self.criticModel, miniBatch,batchsize)
        self.criticModel = self.trainCritic(self.actorModel, self.criticModel, miniBatch,batchsize)
        
        self.actorModel = self.updateParameters(self.actorModel)
        self.criticModel = self.updateParameters(self.criticModel)

    
class Learn():
    def __init__(self,buffersize,batchsize,trainDDPGmodel):
        self.buffersize = buffersize
        self.batchsize = batchsize
        self.trainDDPGmodel = trainDDPGmodel
        
        
    def __call__(self,replaybuffer,state, action, nextstate, rewards):
        replaybuffer = memory(replaybuffer, state, action, nextstate, rewards)
        if len(replaybuffer) > self.batchsize:
            minibatch = samplebuffer(replaybuffer,self.batchsize)
            self.trainDDPGmodel(minibatch,self.batchsize)                          


class GetNoise():
    def __init__(self,noiseDecay,minVar,noiseDacayStep,initnoisevar):
        self.noiseDecay = noiseDecay
        self.minVar = minVar
        self.noiseDacayStep = noiseDacayStep
        self.initnoisevar = initnoisevar
    def __call__(self,runtime):
        if runtime > self.noiseDacayStep:
            self.initnoisevar = self.initnoisevar-self.noiseDecay if self.initnoisevar > self.minVar else self.minVar 
        noise = np.random.normal(0, self.initnoisevar)
        if runtime % 10000 == 0:
            print('noise Variance', self.initnoisevar)
        return noise


            
class GetNoiseAction():
    def __init__(self,actorModel,actionLow, actionHigh):
        self.actorModel = actorModel
        self.actionLow = actionLow
        self.actionHigh = actionHigh
    def __call__(self,state,noise):
        state = np.asarray(state).reshape(1, -1)
        action = getevalaction(self.actorModel, state)[0]
        noisyaction = np.clip(noise + action,self.actionLow,self.actionHigh)
        return noisyaction
    

            
def env_norm(env):
    '''Normalize states (observations) and actions to [-1, 1]'''
    action_space = env.action_space
    state_space = env.observation_space

    env_type = type(env)

    class EnvNormalization(env_type):
        def __init__(self):
            self.__dict__.update(env.__dict__)
            # state (observation - o to match Gym environment class)
            if np.any(state_space.high < 1e10):
                h = state_space.high
                l = state_space.low
                self.o_c = (h+l)/2.
                self.o_sc = (h-l)/2.
            else:
                self.o_c = np.zeros_like(state_space.high)
                self.o_sc = np.ones_like(state_space.high)

            # action
            h = action_space.high
            l = action_space.low
            self.a_c = (h+l)/2.
            self.a_sc = (h-l)/2.

            # reward
            self.r_sc = 0.1
            self.r_c = 0.

            self.observation_space = gym.spaces.Box(self.filter_observation(state_space.low), self.filter_observation(state_space.high))

        def filter_observation(self, o):
            return (o - self.o_c)/self.o_sc

        def filter_action(self, a):
            return self.a_sc*a + self.a_c

        def filter_reward(self, r):
            return self.r_sc*r + self.r_c

        def step(self, a):
            ac_f = np.clip(self.filter_action(a), self.action_space.low, self.action_space.high)
            o, r, done, info = env_type.step(self, ac_f)
            o_f = self.filter_observation(o)

            return o_f, r, done, info
    fenv = EnvNormalization()
    return fenv
    

