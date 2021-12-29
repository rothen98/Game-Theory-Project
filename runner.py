
import numpy as np

from agent import Agent
from game_creator import calculateMatrixGame





def run():
    gameSize = 4
    agent: Agent = Agent(10, 0.1, np.array([0, 0.2, 0.4,  0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8]), gameSize)
    for _ in range(100):
        (agentPayoff, opponentPayoff) = calculateMatrixGame(gameSize)
        agent.observeGame(agentPayoff, opponentPayoff)
        act = agent.pickMove()
        oppAct = 1 #opponentPayoff.sum(axis=0).argmax()
        print("Reward: ", agentPayoff[act,oppAct])
        agent.learn(oppAct)


run()