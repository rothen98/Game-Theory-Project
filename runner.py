
import numpy as np

from agent import Agent

def calculateMatrixGame(size):
    return (np.random.rand(size,size), np.random.rand(size, size))



def run():
    gameSize = 16
    agent: Agent = Agent(10, 0.1, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), gameSize)
    for _ in range(100):
        (agentPayoff, opponentPayoff) = calculateMatrixGame(gameSize)
        agent.observeGame(agentPayoff, opponentPayoff)
        act = agent.pickMove()
        oppAct = 1 #opponentPayoff.sum(axis=0).argmax()
        print("Reward: ", agentPayoff[act,oppAct])
        agent.learn(oppAct)


run()