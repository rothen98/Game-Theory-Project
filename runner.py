
from math import sqrt
import numpy as np

from agent import Agent, Error
from game_creator import calculateMatrixGame
from stationaryOpponent import StationaryOpponent


def calculateRealError(trueAtt, trueBel, estimatedAtt, estimatedBel):
    return sqrt((trueAtt-estimatedAtt)**2 + (trueBel-estimatedBel)**2)



def run():
    gameSize = 16
    allRewards = []
    allEstimatedErrors = []
    allErrors = []
    for _ in range(10):
        agent: Agent = Agent(200, 0.1, np.array([0, .066, .13, .2, .26, .33, .4, .46, .53, .6, .66, .73, .8, .86, .93, 1]), gameSize)
        opponent = StationaryOpponent(0.5, 0.5, 1)
        estimatedErrors = []
        errors = []
        rewards = []
        for _ in range(1000):
            (agentPayoff, opponentPayoff) = calculateMatrixGame(gameSize)
            agent.observeGame(agentPayoff, opponentPayoff)
            act = agent.pickMove()
            oppAct = opponent.pickMove(agentPayoff, opponentPayoff)
            rewards.append(agentPayoff[act,oppAct])
            agent.learn(oppAct)

            estimatedErrors.append(agent.err)
            errors.append(calculateRealError(opponent.attitude, agent.opponent.attitude, opponent.belief, agent.opponent.belief))
            

        print("Average reward:",sum(rewards)/len(rewards))
        allRewards.append(rewards)
        allEstimatedErrors.append(estimatedErrors)
        allErrors.append(errors)

    import matplotlib.pyplot as plt
    plt.plot(np.mean(allRewards, axis=0), label="Reward")
    plt.plot(np.mean(allEstimatedErrors, axis=0), label="Estimated Error")
    plt.plot(np.mean(allErrors, axis=0), label="True Error")
    plt.ylabel('some numbers')
    plt.legend()
    plt.show()
run()