import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from functionTools.loadSaveModel import loadFromPickle
import os
dirName = os.path.dirname(__file__)
import matplotlib.pyplot as plt

def main():
    fileName = 'HalfCheetah1'

    rewardDir = os.path.join(dirName, '..', 'results', 'rewards')
    rewardSavePathMartin = os.path.join(rewardDir, fileName + '_Martin')
    rewardSavePathLucy = os.path.join(rewardDir, fileName + '_Lucy')

    rewLucy = loadFromPickle(rewardSavePathLucy)
    rewMartin = loadFromPickle(rewardSavePathMartin)
    plt.plot(range(len(rewLucy)), rewLucy)
    plt.plot(range(len(rewMartin)), rewMartin)
    plt.show()

#ddpg/results/rewards/HalfCheetah_Lucy

if __name__ == '__main__':
    main()