import numpy as np
from quantecon.game_theory import lemke_howson, NormalFormGame, Player
class StationaryOpponent:
    def __init__(self, attitude, belief, nash):
        self.attitude = attitude
        self.belief = belief
        self.nash = nash

    def pickMove(self, agentMatrix, myMatrix):
        self.agentPayoffMatrix = agentMatrix
        self.opponentPayoffMatrix = myMatrix
        #Choose move from nash equilibrium of modified game
        (_, myNashEq) = self._calculateNashEqOfModifiedGame(self.belief, self.attitude, self.nash)
        print("Opponent real nash eq:", myNashEq)
        move = np.random.choice(np.arange(len(myNashEq)), 1, p=myNashEq)
        return move, myNashEq[move]


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
