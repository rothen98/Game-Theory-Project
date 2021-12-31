import numpy as np
from quantecon.game_theory import lemke_howson, NormalFormGame, Player
class StationaryOpponent:
    def __init__(self, attitude, belief, nash):
        self.attitude = attitude
        self.belief = belief
        self.nash = nash

    def pickMove(self, agentMatrix, myMatrix):
        self.opponentPayoffMatrix = agentMatrix
        self.agentPayoffMatrix = myMatrix
        #Choose move from nash equilibrium of modified game
        (myNashEq, _) = self._calculateNashEqOfModifiedGame(self.attitude, self.belief, self.nash)
        print("Opponent real nash eq:", myNashEq)
        return np.random.choice(np.arange(len(myNashEq)), 1, p=myNashEq)


    # Creates a modified game according to equation given in paper
    def _createModifiedGame(self, attitudeAgent, attitudeOpponent):
        agentModifiedPayOffMatrix = self.agentPayoffMatrix + attitudeAgent*self.opponentPayoffMatrix
        opponentModifiedPayOffMatrix = self.opponentPayoffMatrix + attitudeOpponent*self.agentPayoffMatrix

        return (agentModifiedPayOffMatrix, opponentModifiedPayOffMatrix)

    # Calculates the nash equilibrium of the modified game created with the given parameters
    def _calculateNashEqOfModifiedGame(self, attitudeAgent, attitudeOpponent, nashParameter):

        (agentModified, opponentModified) = self._createModifiedGame(attitudeAgent, attitudeOpponent)

        (nashOne, nashTwo) = lemke_howson(NormalFormGame((Player(agentModified),Player(opponentModified))), nashParameter)

        return (nashOne,nashTwo)    
