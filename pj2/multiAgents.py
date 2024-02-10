# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)   # ???
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()   # (x, y)
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()    # use its asList method.
        newGhostStates = successorGameState.getGhostStates()   # try newGhostStats[0].getPosition()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]   # [0]
        foodDist = []
        for food in oldFood.asList():
            foodDist.append(util.manhattanDistance(newPos, food))
        ghostDist = []
        for ghost in newGhostStates:
            if ghost.scaredTimer > 0:
                # when ghost are scared, we can attach less significance...
                ghostDist.append(util.manhattanDistance(newPos, ghost.getPosition()) * 2)
            else:
                ghostDist.append(util.manhattanDistance(newPos, ghost.getPosition()))

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        ### conservative stategy: only consider ghost ###
        # return min(ghostDist)
        # have to be more conservative.

        # definitely do not step on a ghost!
        if min(ghostDist) <= 2:
            # ultra danger! step out of here!
            return -max(foodDist)
        return min(ghostDist) - min(foodDist) * 2

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        node = {'state': gameState, 'actions': [], 'value': -float('inf'), 'depth': 0}
        node['value'] = self.evaluationFunction(gameState)
        resNode = self.minimax(node, self.depth, 0, gameState.getNumAgents())        
        return resNode['actions'][0]
    
    def minimax(
        self,
        node: dict,     # initial node.
        depth: int,     # depth of search
        agentIdx: int,  # current agent index.
        numAgents: int  # number of agents.
    ) -> dict:
        """ minimax implementation"""
        # source of psuedo code:
        # https://en.wikipedia.org/wiki/Minimax#Minimax_algorithm_with_alternate_moves
        if depth == 0 or node['state'].isWin() or node['state'].isLose():
            node['value'] = self.evaluationFunction(node['state'])
            return node
        if agentIdx == 0:
            # pacman's turn.
            maxVal = -float('inf')
            prevState = node['state']
            resNode = None
            for action in prevState.getLegalActions(0):
                newState = prevState.generateSuccessor(0, action)
                newVal = self.evaluationFunction(newState)
                newNode = {
                    'state': newState,
                    'actions': node['actions'] + [action],
                    'value': newVal,
                    'depth': node['depth'] + 1
                }
                newNode = self.minimax(newNode, depth, 1, numAgents)
                if newNode['value'] > maxVal:
                    maxVal = newNode['value']
                    resNode = newNode 
            return resNode
        else:
            # ghosts' turn.
            minVal = float('inf')
            prevState = node['state']
            resNode = None
            for action in prevState.getLegalActions(agentIdx):
                newState = prevState.generateSuccessor(agentIdx, action)
                newVal = self.evaluationFunction(newState)
                newNode = {
                    'state': newState,
                    'actions': node['actions'],
                    'value': newVal,
                    'depth': node['depth']
                }
                if agentIdx == numAgents - 1:
                    # turn to pacman.
                    newNode = self.minimax(newNode, depth - 1, 0, numAgents)
                else:
                    # turn to next ghost.
                    newNode = self.minimax(newNode, depth, agentIdx + 1, numAgents)
                if newNode['value'] < minVal:
                    minVal = newNode['value']
                    resNode = newNode
            return resNode
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        node = {'state': gameState, 'actions': [], 'value': -float('inf'), 'depth': 0}
        node['value'] = self.evaluationFunction(gameState)
#        expandRes = self.expandNode(node)
#        maxVal = -float('inf') 
#        candidate = node
#        for item in expandRes:
#            if item['state'].isWin():
#                return item['actions'][0]
#            if item['value'] > maxVal:
#                candidate = item
#                maxVal = item['value']
#        return candidate['actions'][0]
        resNode = self.alphabeta(node, self.depth, 0, gameState.getNumAgents(), -float('inf'), float('inf'))
        return resNode['actions'][0]

    def alphabeta(
        self,
        node: dict,     # initial node.
        depth: int,     # depth of search
        agentIdx: int,  # current agent index.
        numAgents: int, # number of agents.
        alpha: float,   # initial call: -infty
        beta: float     # initial call: +infty
    ) -> dict:
        """ alpha-beta implementation"""
        # goto https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning for more info.
        # Be quiet. I borrowed pseudo code there and made neccessary changes.
        if depth == 0 or node['state'].isWin() or node['state'].isLose():
            node['value'] = self.evaluationFunction(node['state'])
            return node
        if agentIdx == 0:
            # pacman's turn.
            maxVal = -float('inf')
            prevState = node['state']
            resNode = None
            for action in prevState.getLegalActions(0):
                newState = prevState.generateSuccessor(0, action)
                newVal = self.evaluationFunction(newState)
                newNode = {
                    'state': newState,
                    'actions': node['actions'] + [action],
                    'value': newVal,
                    'depth': node['depth'] + 1
                }
                newNode = self.alphabeta(newNode, depth, 1, numAgents, alpha, beta)
                # value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
                if newNode['value'] > maxVal:
                    maxVal = newNode['value']
                    resNode = newNode 
                if float(maxVal) > beta:
                    # beta cutoff
                    break
                # alpha := max([alpha, value])
                if maxVal > alpha:
                    alpha = maxVal
            return resNode
        else:
            # ghosts' turn.
            minVal = float('inf')
            prevState = node['state']
            resNode = None
            for action in prevState.getLegalActions(agentIdx):
                newState = prevState.generateSuccessor(agentIdx, action)
                newVal = self.evaluationFunction(newState)
                newNode = {
                    'state': newState,
                    'actions': node['actions'],
                    'value': newVal,
                    'depth': node['depth']
                }
                if agentIdx == numAgents - 1:
                    # turn to pacman.
                    newNode = self.alphabeta(newNode, depth - 1, 0, numAgents, alpha, beta)
                else:
                    # turn to next ghost.
                    newNode = self.alphabeta(newNode, depth, agentIdx + 1, numAgents, alpha, beta)
                # value := min(value, alphabeta(child, depth − 1, alpha, beta, TRUE))
                if newNode['value'] < minVal:
                    minVal = newNode['value']
                    resNode = newNode
                if minVal < alpha:
                    break
                # beta := min(beta, value)
                if minVal < beta:
                    beta = minVal
            return resNode

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        node = {'state': gameState, 'actions': [], 'value': -float('inf'), 'depth': 0}
        node['value'] = self.evaluationFunction(gameState)
        resNode = self.minimax(node, self.depth, 0, gameState.getNumAgents())        
        return resNode['actions'][0]

    def minimax(
        self,
        node: dict,     # initial node.
        depth: int,     # depth of search
        agentIdx: int,  # current agent index.
        numAgents: int  # number of agents.
    ) -> dict:
        """ expectmax implementation"""
        # the ghost agent may not be optimal...
        # this function can partially solve the problem...
        if depth == 0 or node['state'].isWin() or node['state'].isLose():
            node['value'] = self.evaluationFunction(node['state'])
            return node
        if agentIdx == 0:
            # pacman's turn.
            maxVal = -float('inf')
            prevState = node['state']
            resNode = None
            for action in prevState.getLegalActions(0):
                newState = prevState.generateSuccessor(0, action)
                newVal = self.evaluationFunction(newState)
                newNode = {
                    'state': newState,
                    'actions': node['actions'] + [action],
                    'value': newVal,
                    'depth': node['depth'] + 1
                }
                newNode = self.minimax(newNode, depth, 1, numAgents)
                if newNode['value'] > maxVal:
                    maxVal = newNode['value']
                    resNode = newNode 
            return resNode
        else:
            # ghosts' turn.
            # this times ghosts' moves are chosen uniformly at random.
            value = 0.0
            prevState = node['state']
            resNode = None
            legalActions = prevState.getLegalActions(agentIdx)
            for action in legalActions:
                newState = prevState.generateSuccessor(agentIdx, action)
                newVal = self.evaluationFunction(newState)
                newNode = {
                    'state': newState,
                    'actions': node['actions'],
                    'value': newVal,
                    'depth': node['depth']
                }
                if agentIdx == numAgents - 1:
                    # turn to pacman.
                    resNode = self.minimax(newNode, depth - 1, 0, numAgents)
                    value += resNode['value'] / len(legalActions)
                else:
                    # turn to next ghost.
                    resNode = self.minimax(newNode, depth, agentIdx + 1, numAgents)
                    value += resNode['value'] / len(legalActions)

            return {
                'state': prevState,
                'actions': node['actions'],
                'value': value,
                'depth': node['depth']
            }

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodPos = currentGameState.getFood().asList()
    ghostPos = currentGameState.getGhostPositions()
    pos = currentGameState.getPacmanPosition()
    foodDist = []
    for food in foodPos:
        foodDist.append(util.manhattanDistance(pos, food))
    ghostDist = []
    for ghost in ghostPos:
        ghostDist.append(util.manhattanDistance(ghost, pos)) 
    return min(ghostDist) - sum(foodDist) / len(foodDist)

# Abbreviation
better = betterEvaluationFunction
