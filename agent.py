import numpy as np
from collections import Counter
from opponent import Opponent
from particle import Particle
from math import sqrt
import nashpy as nash

class Agent:
    def __init__(self,numberOfParticles, reciprocationLevel,errorLevels, pertubationFactor=10, pertubationFactorNash=10,):
        self.numberOfParticles = numberOfParticles
        self.reciprocationLevel =reciprocationLevel
        self.pertubationFactor = pertubationFactor
        self.pertubationFactorNash = pertubationFactorNash
        self.errorLevels = errorLevels

        #Should the distribution just be uniform in beginning
        self.distributionOverErrorLevels = np.array([1.0/len(errorLevels)]*len(errorLevels))

        #Creating the lookup table 
        # TODO Not sure how to do this
        self.probabilityBins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9,1])
        self.cooperationBins = np.linspace(-0.9, 1, num=20)
        self.errorEstimation = np.zeros((len(self.probabilityBins),len(self.cooperationBins),len(errorLevels)))

        #The particles used
        self.particles = [Particle(np.random.uniform(), np.random.uniform(), np.random.randint(0, 16-1)) for _ in range(0, numberOfParticles)]

    #------------------------------------------------2. Observe Game--------------------------------------------------------
    #Observe a new game
    def observeGame(self, agentPayoffMatrix, opponentPayoffMatrix):
        self.agentPayoffMatrix = agentPayoffMatrix 
        self.opponentPayoffMatrix = opponentPayoffMatrix

    #------------------------------------------------3. Pick Move----------------------------------------------------------
    #Pick a move
    def pickMove(self):
        #Estimate opponent attitude and belief and nash eq method
        self.opponent = self._estimateOpponent()
        #Update my attitude
        self.attitudeAgent = self.opponent.attitude + self.reciprocationLevel
        #Choose move from nash equilibrium of modified game
        (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(self.attitudeAgent, self.opponent.attitude, self.opponent.nashParameter)
        return  np.random.choice(np.arange(len(nashEqAgent)), 1, p=nashEqAgent)

    # Estimates the opponents attitude, belief and nash eq parameter
    def _estimateOpponent(self):
        opponentAttitude = sum([p.pAtt for p in self.particles]) / self.numberOfParticles
        opponentBelief = sum([p.pBel for p in self.particles]) / self.numberOfParticles
        opponentNash = Counter([p.pBel for p in self.particles]).mostCommon(1)[0][0]
        return Opponent(opponentAttitude, opponentBelief, opponentNash)

    # Creates a modified game according to equation given in paper
    def _createModifiedGame(self, attitudeAgent, attitudeOpponent):
        agentModifiedPayOffMatrix = self.agentPayoffMatrix + attitudeAgent*self.opponentPayoffMatrix
        opponentModifiedPayOffMatrix = self.opponentPayoffMatrix + attitudeOpponent*self.agentPayoffMatrix
        return nash.Game(agentModifiedPayOffMatrix, opponentModifiedPayOffMatrix)

    # Calculates the nash equilibrium of the modified game created with the given parameters
    def _calculateNashEqOfModifiedGame(self, attitudeAgent, attitudeOpponent, nashParameter):
        return self._createModifiedGame(attitudeAgent, attitudeOpponent).lemke_howson(initial_dropped_label=nashParameter)
    
    #---------------------------------------4. Observe Opponent and 5. Update Model------------------------------------------------------

    # Observe opponent action and update model
    def learn(self, opponentAction):
        
        # TODO Should the lookup table be updated here?
        # From paper: 
        # We populated a lookup table by creating a large number of games, true attitude/belief 
        # pairs, and estimated attitude/belief pairs, and observing the frequency with which moves with various 
        # predicted probabilities were observed

        #The error estimate is the euclidian between the true attitude and belief of opponent and the
        #estimated attitude and belief.
        err = self._updateErrorEstimate(opponentAction)

        #Resample the particles to get better estimates
        self._resampleParticles()

        #Perturb particles to avoid a concentration of all the probability mass into a single particle
        self._perturbParticles(err)
    
    #Update the error estimate.
    #Returns the new error estimate
    def _updateErrorEstimate(self, opponentAction):
        self.attitudeAgent = self.opponent.belief
        (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(self.attitudeAgent, self.opponent.attitude, self.opponent.nashParameter)
        

        # TODO Unsure about how to use the lookup table. Shouldn't the lookup table be updated somewhere?
        j = nashEqOpponent[opponentAction]
        jDigitized = np.digitize([j], self.probabilityBins)[0]

        k = (self.attitudeAgent + self.opponent.attitude)/(sqrt((self.attitudeAgent**2) + 1)* sqrt((self.opponent.attitude**2)+1))
        kDigitized = np.digitize([k], self.cooperationBins)[0]


        for level in self.errorLevels:
            self.distributionOverErrorLevels[level] *= self.errorEstimation[jDigitized,kDigitized,level]

        self.distributionOverErrorLevels /= np.linalg.norm(self.distributionOverErrorLevels)    

        return sum([level*self.distributionOverErrorLevels[level] for level in self.errorLevels])
        
    # Resamples the particles. 
    # Each particle is given a weight before resampling. The weight is equal to the probability the particle assigned
    # to the opponents move
    def _resampleParticles(self, opponentAction):
        weights = np.zeros(len(self.particles))
        for index, particle in enumerate(self.particles):
            (nashEqAgent, nashEqOpponent) = self._calculateNashEqOfModifiedGame(particle.pAtt, particle.pBel, particle.pBel)
            weights[index] =  nashEqOpponent[opponentAction]

        #Add noise
        self.particles = np.random.choice(self.particles, self.numberOfParticles, p=weights)

    #Perturbs the particles slightly
    def _perturbParticles(self, err):
        newNashMethod = np.random.uniform() < err*self.pertubationFactorNash
        for particle in self.particles:
            particle.pAtt =  np.random.normal(particle.pAtt, err*self.pertubationFactor)
            particle.pBel =  np.random.normal(particle.pBel, err*self.pertubationFactor)
            if(newNashMethod):
                particle.pNash = np.random.randint(0, 16-1)


    
        

    
