import time
import sys
import os
DIRNAME = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

from subprocess import Popen, PIPE
import json
import math
import numpy as np
import itertools as it

class ExcuteCodeOnConditionsParallel:
    def __init__(self, numSample, numCmdList):
        self.numSample = numSample
        self.numCmdList = numCmdList

    def __call__(self, conditions, fileNameList):
        assert self.numCmdList >= len(conditions), "condition number > cmd number, use more cores or less conditions"
        numCmdListPerCondition = math.floor(self.numCmdList / len(conditions))
        if self.numSample:
            startSampleIndexes = np.arange(0, self.numSample, math.ceil(self.numSample / numCmdListPerCondition))
            endSampleIndexes = np.concatenate([startSampleIndexes[1:], [self.numSample]])
            startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)
            conditionStartEndIndexesPair = list(it.product(conditions, startEndIndexesPair))
            cmdList = [['python3', fileName, json.dumps(condition), str(startEndSampleIndex[0]), str(startEndSampleIndex[1])]
                       for condition, startEndSampleIndex in conditionStartEndIndexesPair for fileName in fileNameList]
        else:
            cmdList = [['python3', fileName, json.dumps(condition)] for condition in conditions for fileName in fileNameList]
        print(cmdList)
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.communicate()
            # proc.wait()
        return cmdList

def main():
    startTime = time.time()
    fileNameList = ['evalDDPG_halfCheetah.py']
    numSample = None
    numCpuToUse = int(0.8 * os.cpu_count())
    excuteCodeParallel = ExcuteCodeOnConditionsParallel(numSample, numCpuToUse)
    print("start")

    people = ['Lucy', 'Phil', 'Martin']
    seedList = [1, 2, 3, 4, 5]

    conditionLevels = [(person, seed) for person in people for seed in seedList]

    conditions = []
    for condition in conditionLevels:
        person, seed = condition
        parameters = {'person': person, 'seed': seed}
        conditions.append(parameters)

    cmdList = excuteCodeParallel(conditions, fileNameList)
    print(cmdList)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))

if __name__ == '__main__':
    main()
