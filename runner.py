from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from game_creator import calculateMatrixGame
from stationaryOpponent import StationaryOpponent


def calculateRealError(trueAtt, estimatedAtt, trueBel, estimatedBel):
    attError = (trueAtt - estimatedAtt)**2
    belError = (trueBel - estimatedBel)**2
    return sqrt(attError + belError)

def cooperation(att,bel):
    return (att + bel) / (sqrt(att**2 + 1) * sqrt(bel**2 + 1))

# Run two cooperating agents against each other. Currently not working.
def runSelfplay():
    gameSize = 16
    allRewards = []
    allCooperation = []
    timesteps = 1000

    for iteration in range(2):
        print("Iteration:", iteration+1)
        print("Initializing agents...")

        agent: Agent = Agent(200, 0.1, np.array([0.05,0.1, .2, .3, .4, .5, .6, .7, .8, 0.9]), gameSize)
        opponent = Agent(200, 0.1, np.array([0.05,0.1, .2, .3, .4, .5, .6, .7, .8, 0.9]), gameSize)
        rewards = []
        cooperations = []
        
        for i in range(1, timesteps + 1):
            agentPayoff, opponentPayoff = calculateMatrixGame(gameSize)
            agent.observeGame(agentPayoff, opponentPayoff)
            opponent.observeGame(opponentPayoff, agentPayoff)
            act = agent.pickMove()
            oppAct = opponent.pickMove()
            rewards.append((agentPayoff[act,oppAct] + opponentPayoff[oppAct,act])/2)
            cooperations.append(cooperation(agent.attitudeAgent, opponent.attitudeAgent))

            if(i % 100 == 0):
                print("Agent attitude:", agent.attitudeAgent, "Belief:",agent.opponent.attitude)
                print("Opponent attitude:", opponent.attitudeAgent, "Belief:",opponent.opponent.attitude)
                print("---- Iteration:",i,"Estimated Cooperation: ", cooperations[-1],"----")
            
            agent.learn(oppAct)
            opponent.learn(act)
        
        allRewards.append(rewards)
        allCooperation.append(cooperations)
    
    plt.plot(np.mean(allCooperation,axis=0), label="Cooperation level")
    plt.scatter(y=np.mean(allRewards, axis=0), x=list(range(timesteps)), label="Average Payoff", s=1)
    plt.xlabel('Time')
    plt.legend()
    plt.show()

# Run a cooperating agent against a stationary agent
def run():
    gameSize = 16
    allEstimatedErrors = []
    allErrors = []
    allEstimated_prob_difference = []
    timesteps = 1000

    for iteration in range(100):
        agent: Agent = Agent(200, 0.1, np.array([0.05, 0.1, .2, .3, .4, .5, .6, .7, .8, 0.9]), gameSize)
        opponent = StationaryOpponent(np.random.uniform(-1,1), np.random.uniform(-1, 1),
                                      np.random.randint(0, gameSize-1))

        print("Opponent attitude:", opponent.attitude, "belief:", opponent.belief)

        estimatedErrors = []
        errors = []
        estimated_prob_difference = []

        print("Iteration:", iteration+1)

        for i in range(1, timesteps+1):
            agentPayoff, opponentPayoff = calculateMatrixGame(gameSize)
            agent.observeGame(agentPayoff, opponentPayoff)
            act = agent.pickMove()
            oppAct, withProb = opponent.pickMove(agentPayoff, opponentPayoff)
            probOfOpponentAction = agent.learn(oppAct)
            estimated_prob_difference.append(1 - abs(withProb - probOfOpponentAction))
            estimatedErrors.append(agent.err)
            errors.append(calculateRealError(opponent.attitude, agent.opponent.attitude, opponent.belief,
                                             agent.opponent.belief))
            if(i % 100 == 0):
                print("Timestep:", i, "Estimated attitude: ", agent.opponent.attitude, "belief:", agent.opponent.belief)
            
        print("Final estimated error is:", estimatedErrors[-1])
        print("Final true error is:", errors[-1])

        allEstimatedErrors.append(estimatedErrors)
        allErrors.append(errors)
        allEstimated_prob_difference.append(estimated_prob_difference)
    
    plt.plot(np.mean(allEstimatedErrors, axis=0), label="Estimated Error")
    plt.plot(np.mean(allErrors, axis=0), label="True Error")
    plt.scatter(y=np.mean(allEstimated_prob_difference,axis=0), x=list(range(timesteps)), label="Predictive Accuracy", s=1)
    plt.xlabel('Time')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
