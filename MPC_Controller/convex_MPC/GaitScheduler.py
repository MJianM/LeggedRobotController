import numpy as np

class GaitScheduler():     
    def __init__(self, nSegment:int, offset:np.ndarray, durations:np.ndarray, name:str):

        # offset in segments
        self.offsets = offset.flatten()
        # duration of stance
        self.durations = durations.flatten()
        # offsets in phase (0 to 1)
        self.offsetsFloat = offset / nSegment
        # durations in phase (0 to 1)
        self.durationsFloat = durations / nSegment
        # period of gait
        self.nSegment = nSegment
        # self.__nIterations = nSegment
        self.name = name

        # self.__stance = durations[0]
        # self.__swing = nSegment - durations[0]

        assert self.offsets.shape[0] == self.durations.shape[0]
        self.legNum = self.offsets.shape[0]
        # MPC 向前规划一个步态周期
        self.__mpc_table = [0 for _ in range(nSegment*self.legNum)]

        self.stanceState = np.ones(self.legNum)
        self.swingState = np.zeros(self.legNum)
        self.contactFlag = np.ones(self.legNum)
        self.iteration = 0  # (0 to nSegment)
        self.phase = 0.0 # (0 to 1)

    def update(self, currentIteration:int, iterationsPerSegment:int):
        '''
            update stance and swing state : (legNum,)
        '''
        self.iteration = (currentIteration / iterationsPerSegment) % self.nSegment
        self.phase = float(currentIteration % (iterationsPerSegment * self.nSegment)) / float(iterationsPerSegment * self.nSegment)

        progress = self.phase - self.offsetsFloat
        swingBeginFlag:list = [0 for _ in range(self.legNum)]
        for leg in range(self.legNum):
            if progress[leg] < 0.0:
                progress[leg] += 1.0

            if progress[leg] >= self.durationsFloat[leg]:
                # swing phase
                if self.contactFlag[leg] == 1:
                    swingBeginFlag[leg] = 1
                    self.contactFlag[leg] = 0
                self.stanceState[leg] = 0
                self.swingState[leg] = (progress[leg] - self.durationsFloat[leg])/(1.0 - self.durationsFloat[leg])
            else:
                # stance phase
                if self.contactFlag[leg] == 0:
                    self.contactFlag[leg] = 1
                self.stanceState[leg] = progress[leg] / self.durationsFloat[leg]
                self.swingState[leg] = 0

        return swingBeginFlag

    def getMpcTable(self):

        for i in range(self.nSegment):

            iter = (self.iteration + i + 1) % self.nSegment
            progress = iter - self.offsets
            for leg in range(self.legNum):
                if progress[leg] < 0:
                    progress[leg] += self.nSegment

                if progress[leg] > self.durations[leg]:
                    self.__mpc_table[self.legNum*i + leg] = 0
                else:
                    self.__mpc_table[self.legNum*i + leg] = 1
        return self.__mpc_table
            
    def getStanceTime(self,leg:int):
        return self.durations[leg]
    def getContactFlag(self):
        return self.contactFlag