import numpy as np
from collections import Counter
from game_creator import calculateMatrixGame
from opponent import Opponent
from particle import Particle
from math import sqrt
import nashpy as nash

class Agent:
    def __init__(self,numberOfParticles, reciprocationLevel,errorLevels, gamesize,pertubationFactor=0.1, pertubationFactorNash=0.1):
        self.numberOfParticles = numberOfParticles
        self.reciprocationLevel = reciprocationLevel
        self.pertubationFactor = pertubationFactor
        self.pertubationFactorNash = pertubationFactorNash
        self.errorLevels = errorLevels
        self.gameSize = gamesize

        #Should the distribution just be uniform in beginning?
        self.distributionOverErrorLevels = np.array([1.0/len(errorLevels)]*len(errorLevels))

        #Creating the lookup table. We use frequencies instead of probabilities and calculate probabilities when necessary 
        # TODO Not sure how to do this
        self.errorLevels = errorLevels
        self.probabilityBins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9,1])
        self.cooperationBins = np.linspace(-0.9, 1, num=20)
        self.lookupTable = self._generateLookupTable()
        # np.ones((len(self.probabilityBins)+1,len(self.cooperationBins)+1,len(errorLevels)))

        #The particles used
        self.particles = [Particle(np.random.uniform(), np.random.uniform(), np.random.randint(low=0, high=self.gameSize-1)) for _ in range(0, numberOfParticles)]

        print("Initialized")

    def _generateLookupTable(self):
        # We populated a lookup table by creating a large number of games, true attitude/belief pairs, 
        # and estimated attitude/belief pairs, and observing the frequency with which moves with various predicted probabilities 
        # were observed
        table = np.zeros((len(self.probabilityBins)+1,len(self.cooperationBins)+1,len(self.errorLevels) +1))
        for _ in range(0,1000):
            #Generate game
            (self.agentPayoffMatrix, self.opponentPayoffMatrix) = calculateMatrixGame(self.gameSize)
            # Generate true attitude and belief
            oppAttitudeTrue = np.random.uniform()
            oppBeliefTrue = np.random.uniform()
            # Generate estimated attitude belief
            oppAttitudeEstimated = np.random.uniform()
            oppBeliefEstimated = np.random.uniform()

            # Generate opponent nash
            oppNash = np.random.randint(0, self.gameSize)

            # Generate agent attitude
            agentAttitude = np.random.uniform()

            # Calculate cooperation level and error level
            # Should I calculate estimated or true cooperation level?
            coop = (oppAttitudeEstimated + oppBeliefEstimated)/(sqrt((oppAttitudeEstimated**2)+1)*sqrt((oppBeliefEstimated**2) + 1))
            cooperationDigitized = np.digitize([coop], self.cooperationBins)[0]

            err = sqrt((oppAttitudeTrue-oppAttitudeEstimated)**2 + (oppBeliefTrue-oppBeliefEstimated)**2)
            errorDigitized = np.digitize([err], self.errorLevels)[0]


            move = np.random.randint(0, self.gameSize)

            print("Gamesize:", self.gameSize)
            (_, nashOpp) = self._calculateNashEqOfModifiedGame(agentAttitude,  oppAttitudeEstimated, oppNash)
            print("Nash length: ", len(nashOpp))
            print("Nash: ", nashOpp)
            probability = nashOpp[move]

            probabilityDigitized = np.digitize([probability], self.probabilityBins)[0]

            table[probabilityDigitized, cooperationDigitized, errorDigitized] += 1
            # Observe a move
            # How should I calculate this move? Just select one randomly? 
            # Further, to calculate the probabilities for different moves I need to generate an attitude for the agent 
            # and a nash parameter for the opponent. 
        print(table)
        return table    

                
    #------------------------------------------------2. Observe Game--------------------------------------------------------
    #Observe a new game
    def observeGame(self, agentPayoffMatrix, opponentPayoffMatrix):
        self.agentPayoffMatrix = agentPayoffMatrix 
        self.opponentPayoffMatrix = opponentPayoffMatrix
        print("------------------- NEW GAME OBSERVED -----------------------------")
        #print("I have observed a new game with agent payoff")
        #print(self.agentPayoffMatrix)
        #print("and opponent payoff")
        #print(self.opponentPayoffMatrix)


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
        opponentNash = Counter([p.pNash for p in self.particles]).most_common(1)[0][0]
        return Opponent(opponentAttitude, opponentBelief, opponentNash)

    # Creates a modified game according to equation given in paper
    def _createModifiedGame(self, attitudeAgent, attitudeOpponent):
        agentModifiedPayOffMatrix = self.agentPayoffMatrix + attitudeAgent*self.opponentPayoffMatrix
        opponentModifiedPayOffMatrix = self.opponentPayoffMatrix + attitudeOpponent*self.agentPayoffMatrix

        
        #print("Modified agent payoff", agentModifiedPayOffMatrix)
        #print("Modified opponent payoff", opponentModifiedPayOffMatrix)
        return nash.Game(agentModifiedPayOffMatrix, opponentModifiedPayOffMatrix)

    # Calculates the nash equilibrium of the modified game created with the given parameters
    def _calculateNashEqOfModifiedGame(self, attitudeAgent, attitudeOpponent, nashParameter):

        return self._createModifiedGame(attitudeAgent, attitudeOpponent).lemke_howson(initial_dropped_label=nashParameter)
    
    #---------------------------------------4. Observe Opponent and 5. Update Model------------------------------------------------------

    # Observe opponent action and update model
    def learn(self, opponentAction):

        #The error estimate is the euclidian between the true attitude and belief of opponent and the
        #estimated attitude and belief.
        err = self._updateErrorEstimate(opponentAction)
        print("The new estimated error is", err)

        #Resample the particles to get better estimates
        self._resampleParticles(opponentAction)

        #Perturb particles to avoid a concentration of all the probability mass into a single particle
        self._perturbParticles(err)
    
    #Update the error estimate.
    #Returns the new error estimate
    def _updateErrorEstimate(self, opponentAction):
        self.attitudeAgent = self.opponent.belief
        (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(self.attitudeAgent, self.opponent.attitude, self.opponent.nashParameter)
        
        print("Opponent chosen action:", opponentAction, "Our estimated probabilities:", nashEqOpponent)
    
        # TODO Unsure about how to use the lookup table. 
        j = nashEqOpponent[opponentAction]
        jDigitized = np.digitize([j], self.probabilityBins)[0]
        k = (self.opponent.attitude + self.opponent.belief)/(sqrt(sqrt((self.opponent.attitude**2)+1)*(self.opponent.belief**2) + 1))
        kDigitized = np.digitize([k], self.cooperationBins)[0]

        #print("j-value", j,  "jDigitized", jDigitized, "probability levels", self.probabilityBins)
        #print("k-value", k,  "kDigitized", kDigitized, "cooperation levels", self.cooperationBins)
        #print("Current error level", np.argmax(self.distributionOverErrorLevels), "All error levels:", self.distributionOverErrorLevels)


        #updating error level distribution
        for level in range(0,len(self.errorLevels)):
            self.distributionOverErrorLevels[level] *= self.lookupTable[jDigitized,kDigitized,level]/sum(self.lookupTable[:,kDigitized, level])
        
        #Normalizing error level distribution
        self.distributionOverErrorLevels /= np.linalg.norm(self.distributionOverErrorLevels)    

        #Calculating error estimation
        return sum([self.errorLevels[level]*self.distributionOverErrorLevels[level] for level in range(0, len(self.errorLevels))])
        
    # Resamples the particles. 
    # Each particle is given a weight before resampling. The weight is equal to the probability the particle assigned
    # to the opponents move
    def _resampleParticles(self, opponentAction):
        weights = np.zeros(len(self.particles))
        for index, particle in enumerate(self.particles):
            (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(particle.pAtt, particle.pBel, particle.pNash)
            weights[index] =  nashEqOpponent[opponentAction]

        weightsSum = weights.sum()
        print("Weight sum", weightsSum)
        if(weightsSum == 0):
            # If all weights are zero, set equal probability for all particles
            weights = np.ones(len(weights))/len(weights)
        else:
            #Make sure sum is one
            weights = weights / weightsSum
      
        #Add noise
        self.particles = np.random.choice(self.particles, self.numberOfParticles, p=weights)

    #Perturbs the particles slightly
    def _perturbParticles(self, err):
        newNashMethod = np.random.uniform() < err*self.pertubationFactorNash
        for particle in self.particles:
            particle.pAtt =  np.clip(np.random.normal(particle.pAtt, err*self.pertubationFactor), -1, 1)
            particle.pBel =  np.clip(np.random.normal(particle.pBel, err*self.pertubationFactor), -1, 1)
            if(newNashMethod):
                particle.pNash = np.random.randint(0, self.gameSize-1)


    
        

    
