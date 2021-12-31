
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
    allEstimated_prob_difference = []
    timesteps = 1000
    for _ in range(100):
        agent: Agent = Agent(200, 0.1, np.array([0.1, .2, .4, .6, .8, 0.9]), gameSize)
        opponent = StationaryOpponent(0.5, 0.5, 1)
        estimatedErrors = []
        errors = []
        rewards = []
        estimated_prob_difference = []
        
        for _ in range(timesteps):
            (agentPayoff, opponentPayoff) = calculateMatrixGame(gameSize)
            agent.observeGame(agentPayoff, opponentPayoff)
            act = agent.pickMove()
            (oppAct, withProb) = opponent.pickMove(agentPayoff, opponentPayoff)
            rewards.append(agentPayoff[act,oppAct])
            probOfOpponentAction = agent.learn(oppAct)
            estimated_prob_difference.append(1-(abs(withProb-probOfOpponentAction)))
            estimatedErrors.append(agent.err)
            errors.append(calculateRealError(opponent.attitude, agent.opponent.attitude, opponent.belief, agent.opponent.belief))
            

        print("Average reward:",sum(rewards)/len(rewards))
        allRewards.append(rewards)
        allEstimatedErrors.append(estimatedErrors)
        allErrors.append(errors)
        allEstimated_prob_difference.append(estimated_prob_difference)

    import matplotlib.pyplot as plt
    #plt.plot(np.mean(allRewards, axis=0), label="Reward")
    plt.plot(np.mean(allEstimatedErrors, axis=0), label="Estimated Error")
    plt.plot(np.mean(allErrors, axis=0), label="True Error")
    plt.scatter(y=np.mean(allEstimated_prob_difference,axis=0), x=list(range(timesteps)), label="Predictive Accuracy", s=1)

        
    plt.ylabel('some numbers')
    plt.legend()
    plt.show()
run()