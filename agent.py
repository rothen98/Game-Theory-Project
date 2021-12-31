import numpy as np
from collections import Counter
from game_creator import calculateMatrixGame
from opponent import Opponent
from particle import Particle
from math import sqrt
#import nashpy as nash
from quantecon.game_theory import lemke_howson, NormalFormGame, Player

class Error(Exception):
    """Base class for other exceptions"""
    pass

class Agent:
    def __init__(self,numberOfParticles, reciprocationLevel,errorLevels, gamesize,pertubationFactor=0.2, pertubationFactorNash=0.1):
        self.numberOfParticles = numberOfParticles
        self.reciprocationLevel = reciprocationLevel
        self.pertubationFactor = pertubationFactor
        self.pertubationFactorNash = pertubationFactorNash
        self.errorLevels = errorLevels
        self.gameSize = gamesize

        #Errro distribution is uniform?
        self.distributionOverErrorLevels = np.array([1.0/len(errorLevels)]*len(errorLevels))

      
        self.errorLevels = errorLevels
        self.probabilityBins = np.array([0, .066, .13, .2, .26, .33, .4, .46, .53, .6, .66, .73, .8, .86, .93, 1])
        self.cooperationBins = np.linspace(-0.9, 0.9, num=10)
        self.lookupTable = self._generateLookupTable()
    

        #The particles used
        self.particles = [Particle(np.random.normal(), np.random.normal(), np.random.randint(low=0, high=self.gameSize-1)) for _ in range(0, numberOfParticles)]

        print("Initialized")

    def _generateLookupTable(self):
        # We populated a lookup table by creating a large number of games, true attitude/belief pairs, 
        # and estimated attitude/belief pairs, and observing the frequency with which moves with various predicted probabilities 
        # were observed.
        table = np.ones((len(self.cooperationBins)+1,len(self.errorLevels) +1, len(self.probabilityBins)+1))
        for _ in range(100000):
            #Generate game
            (self.agentPayoffMatrix, self.opponentPayoffMatrix) = calculateMatrixGame(self.gameSize)
            # Generate true attitude and belief
            oppAttitudeTrue = np.random.uniform(-1,1)
            oppBeliefTrue = np.random.uniform(-1,1)
            # Generate estimated attitude belief
            oppAttitudeEstimated = np.random.uniform(-1,1)
            oppBeliefEstimated = np.random.uniform(-1,1)

            # Generate opponent nash
            oppNash = np.random.randint(0, self.gameSize)
            

            # Calculate cooperation level and error level
            # Should I calculate estimated or true cooperation level?
            coop = (oppAttitudeEstimated + oppBeliefEstimated)/(sqrt((oppAttitudeEstimated**2)+1)*sqrt((oppBeliefEstimated**2) + 1))
            cooperationDigitized = np.digitize([coop], self.cooperationBins)[0]

            err = sqrt((oppAttitudeTrue-oppAttitudeEstimated)**2 + (oppBeliefTrue-oppBeliefEstimated)**2)
            errorDigitized = np.digitize([err], self.errorLevels)[0]


            #Observe a move
            (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(oppBeliefTrue, oppAttitudeTrue, oppNash)
            move = np.random.choice(np.arange(len(nashEqOpponent)), 1, p=nashEqOpponent)

            #Estimate a probability for the move
            (_, nashOpp) = self._calculateNashEqOfModifiedGame(oppBeliefEstimated,  oppAttitudeEstimated, oppNash)
            probability = nashOpp[move]
            probabilityDigitized = np.digitize([probability], self.probabilityBins)[0]

            table[cooperationDigitized, errorDigitized, probabilityDigitized] += 1
            
        print(table)
        print(table.shape)
        for i,errorLevels in enumerate(table):
            for j,probabilities in enumerate(errorLevels):
                table[i,j] = probabilities / probabilities.sum(axis=0)
        print(table)
        return table    

                
    #------------------------------------------------2. Observe Game--------------------------------------------------------
    #Observe a new game
    def observeGame(self, agentPayoffMatrix, opponentPayoffMatrix):
        self.agentPayoffMatrix = agentPayoffMatrix 
        self.opponentPayoffMatrix = opponentPayoffMatrix
        print("------------------- NEW GAME OBSERVED -----------------------------")


    #------------------------------------------------3. Pick Move----------------------------------------------------------
    #Pick a move
    def pickMove(self):
        #Estimate opponent attitude and belief and nash eq method
        self.opponent = self._estimateOpponent()
        print("Estimated opponent:", self.opponent.readable())
        #Update my attitude
        self.attitudeAgent = min(self.opponent.attitude + self.reciprocationLevel, 1)
        print("Updated my attitude:", self.attitudeAgent)
        #Choose move from nash equilibrium of modified game
        (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(self.attitudeAgent, self.opponent.attitude, self.opponent.nashParameter)
        

        move = np.random.choice(np.arange(len(nashEqAgent)), 1, p=nashEqAgent)
        print("My nash equilibrium:", nashEqAgent,"Chosen move is:",move)
        return move

    # Estimates the opponents attitude, belief and nash eq parameter
    def _estimateOpponent(self):
        opponentAttitude = sum([p.pAtt for p in self.particles]) / self.numberOfParticles
        opponentBelief = sum([p.pBel for p in self.particles]) / self.numberOfParticles
        c = Counter([p.pNash for p in self.particles])
        m = c.most_common(1)
        opponentNash = m[0][0]
        return Opponent(opponentAttitude, opponentBelief, opponentNash)

    # Creates a modified game according to equation given in paper
    def _createModifiedGame(self, attitudeAgent, attitudeOpponent):
        agentModifiedPayOffMatrix = self.agentPayoffMatrix + attitudeAgent*self.opponentPayoffMatrix
        opponentModifiedPayOffMatrix = self.opponentPayoffMatrix + attitudeOpponent*self.agentPayoffMatrix

        #print("Modified agent payoff", agentModifiedPayOffMatrix)
        #print("Modified opponent payoff", opponentModifiedPayOffMatrix)
        #return nash.Game(agentModifiedPayOffMatrix, opponentModifiedPayOffMatrix)
        return (agentModifiedPayOffMatrix, opponentModifiedPayOffMatrix)

    # Calculates the nash equilibrium of the modified game created with the given parameters
    def _calculateNashEqOfModifiedGame(self, attitudeAgent, attitudeOpponent, nashParameter):
        #nashEquilibrias = list(self._createModifiedGame(attitudeAgent, attitudeOpponent).support_enumeration())
        #if(nashEquilibrias.length < self.gameSize):
        #    return nashEquilibrias[0]
        #else: return nashEquilibrias[nashParameter] 
        # q = NormalFormGame((Player(A), Player(B)))

        (agentModified, opponentModified) = self._createModifiedGame(attitudeAgent, attitudeOpponent)
        #(nashOne, nashTwo) = perform_lemke_howson(agentModified,opponentModified, nashParameter)

        (nashOne, nashTwo) = lemke_howson(NormalFormGame((Player(agentModified),Player(opponentModified))), nashParameter)

        return (nashOne,nashTwo)    


    
    #---------------------------------------4. Observe Opponent and 5. Update Model------------------------------------------------------

    # Observe opponent action and update model
    def learn(self, opponentAction):

        #The error estimate is the euclidian between the true attitude and belief of opponent and the
        #estimated attitude and belief.
        self.err = self._updateErrorEstimate(opponentAction)
        print("The new estimated error is", self.err)

        #Resample the particles to get better estimates
        self._resampleParticles(opponentAction)

        #Perturb particles to avoid a concentration of all the probability mass into a single particle
        self._perturbParticles(self.err)
    
    #Update the error estimate.
    #Returns the new error estimate
    def _updateErrorEstimate(self, opponentAction):
        self.attitudeAgent = self.opponent.belief
        (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(self.attitudeAgent, self.opponent.attitude, self.opponent.nashParameter)
        
        print("Opponent chosen action:", opponentAction, "Our estimated probabilities:", nashEqOpponent)
    
 
        j = nashEqOpponent[opponentAction]
        jDigitized = np.digitize([j], self.probabilityBins)[0]
        k = (self.opponent.attitude + self.opponent.belief)/(sqrt((self.opponent.attitude**2)+1)*sqrt((self.opponent.belief**2) + 1))
        kDigitized = np.digitize([k], self.cooperationBins)[0]

        #print("j-value", j,  "jDigitized", jDigitized, "probability levels", self.probabilityBins)
        #print("k-value", k,  "kDigitized", kDigitized, "cooperation levels", self.cooperationBins)
        #print("Current error level", np.argmax(self.distributionOverErrorLevels), "All error levels:", self.distributionOverErrorLevels)
        

        #updating error level distribution
        for level in range(0,len(self.errorLevels)):
            lookupValue  = self.lookupTable[kDigitized,level,jDigitized]
            self.distributionOverErrorLevels[level] *= lookupValue
        
        #Normalizing error level distribution
        self.distributionOverErrorLevels /= self.distributionOverErrorLevels.sum()

        #Calculating error estimation
        return sum([self.errorLevels[level]*self.distributionOverErrorLevels[level] for level in range(0, len(self.errorLevels))])
        
    # Resamples the particles. 
    # Each particle is given a weight before resampling. The weight is equal to the probability the particle assigned
    # to the opponents move
    def _resampleParticles(self, opponentAction):
        weights = np.zeros(len(self.particles))
        for index, particle in enumerate(self.particles):
            (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(particle.pBel,particle.pAtt, particle.pNash)
            #print("Attitude:",particle.pAtt,"Belief:", particle.pBel, "Nash:", particle.pNash, "Weight:", nashEqOpponent[opponentAction])
            weights[index] =  nashEqOpponent[opponentAction]

        weightsSum = weights.sum()
        print("Weight sum", weightsSum)
        if(weightsSum == 0):
            # If all weights are zero, set equal probability for all particles
            weights = np.ones(len(weights))/len(weights)
        else:
            #Make sure sum is one
            weights = weights / weightsSum
      
        #Choose best particles
        self.particles = np.random.choice(self.particles, self.numberOfParticles, p=weights)

    #Perturbs the particles slightly
    def _perturbParticles(self, err):
        newNashMethod = np.random.uniform() < err*self.pertubationFactorNash
        for i in range(len(self.particles)):
            particle = self.particles[i]
            att =  np.clip(np.random.normal(particle.pAtt, err*self.pertubationFactor),-1,1)
            bel =  np.clip(np.random.normal(particle.pBel, err*self.pertubationFactor),-1,1)
            nash = particle.pNash
            if(newNashMethod):
                nash = np.random.randint(0, self.gameSize-1)
            self.particles[i] = Particle(att,bel,nash)    
          


    
        

    
