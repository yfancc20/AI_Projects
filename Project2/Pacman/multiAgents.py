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
import time, math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # current pos
        currentPos = gameState.getPacmanPosition();

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        # debugging output
        # print 'next: %s' % (legalMoves[chosenIndex])
        # raw_input()

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        "[Project 3] YOUR CODE HERE"

        # Map Information: 9*25 here, including walls
        mapWidth = 23
        mapHeight = 7

        # Successor Information
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Position Information 
        newPos = successorGameState.getPacmanPosition()
        newX, newY = [newPos[0], newPos[1]]      

        """ 
        kinds of features 
        """
        def featureGhost(newX, newY, state):
            """ Feature about Ghosts """

            # Ghost Information
            newGhostStates = state.getGhostStates()
            newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
            ghostPos = state.getGhostPosition(1)
            ghostX, ghostY = [ghostPos[0], ghostPos[1]]

            # featureScore
            fScore = 0
            if abs(newX-ghostX) == 0 and abs(newY-ghostY) == 0:
                fScore = -30
            elif abs(newX-ghostX) <= 1 and abs(newY-ghostY) <= 1:
                fScore = -20
            return fScore


        def featureFood(newX, newY, state, action):
            """ Feature about Food """

            fScore = 0

            # have food to eat
            newFoodExist = state.hasFood(newX, newY)
            if newFoodExist == True:
                fScore += 20

            # some lists of each action
            if action == "West":
                dx = [-2, -1, 0, -2, -1, 0, -2, -1, 0]
                dy = [1, 1, 1, 0, 0, 0, -1, -1, -1]
                xStart, xEnd, yStart, yEnd = [1, newX+1, 1, 9]
            elif action == "North":
                dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
                dy = [2, 2, 2, 1, 1, 1, 0, 0, 0]
                xStart, xEnd, yStart, yEnd = [1, 24, newY, 9]
            elif action == "East":
                dx = [0, 1, 2, 0, 1, 2, 0, 1, 2]
                dy = [1, 1, 1, 0, 0, 0, -1, -1, -1]
                xStart, xEnd, yStart, yEnd = [newX, 24, 1, 9]
            elif action == "South":
                dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
                dy = [0, 0, 0, -1, -1, -1, -2, -2, -2]
                xStart, xEnd, yStart, yEnd = [1, 24, 1, newY+1]
            elif action == "Stop":
                return 0

            # count food around the newPos (3x3)
            for i in range(9):
                countX = newX + dx[i]
                countY = newY + dy[i]
                if countX > 0 and countX < 25 and countY > 0 and countY < 9:
                    if state.hasFood(countX, countY):
                        fScore += 1

            # count food from the action's direction
            minValue = 100
            for y in range(yStart, yEnd, 1):
                for x in range(xStart, xEnd, 1):
                    if x == newX and y == newY:
                        continue
                    if state.hasFood(x,y) and abs(newX-x)+abs(newY-y) < minValue:
                        minValue = abs(newX-x) + abs(newY-y)
            if minValue != 100:
                fScore += math.floor((30-minValue)/2)

            return fScore


        def featureCapsule(newX, newY, state):
            """ Feature about Capsule """

            fScore = 0

            # Capsules are eaten or not
            newCapsule = state.getCapsules()
            if len(newCapsule) > 0:
                capsuleX, capsuleY = [newCapsule[0][0], newCapsule[0][1]]
            else:
                fScore = 0
                return fScore

            # Distance to capsule
            if newX == capsuleX and newY == capsuleY:
                fScore += 30
            elif abs(newX-capsuleX) <= 1 and abs(newY-capsuleY) <= 1:
                fScore += 25
            elif abs(newX-capsuleX) <= 2 and abs(newY-capsuleY) <= 2:
                fScore += 20
            elif abs(newX-capsuleX) <= 3 and abs(newY-capsuleY) <= 3:
                fScore += 15

            return fScore


        # Each kind of feature's weight and function
        w1 = 5
        f1 = featureGhost(newX, newY, successorGameState)
        w2 = 1
        f2 = featureFood(newX, newY, currentGameState, action)
        w3 = 4
        f3 = featureCapsule(newX, newY, currentGameState)
        evalScore = w1 * f1 + w2 * f2 + w3 * f3

        # debugging output
        # print '%s, (%d %d %d) => %d' % (action, f1, f2, f3, evalScore)
        # print 'next:(%d,%d); Food:%s; ghost:(%d,%d)' % (newX, newY, newFoodExist, ghostX, ghostY)
  
        # return final score      
        return evalScore


def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        
        "[Project 3] YOUR CODE HERE" 
        """ Functions of Min-Max """
        def minimax(state, depth, agentIndex):
            """ Min-Max head function"""

            agentNum = state.getNumAgents()

            # Terminal-state or maximum depth
            if depth == 0 or state.isLose() or state.isWin():
                return {'value': self.evaluationFunction(state)}

            # Pacman: maxValue; Ghost: minValue
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex)
            else:
                return minValue(state, depth, agentIndex)


        def maxValue(state, depth, agentIndex):
            """ Function of Max-Value"""

            legalMoves = state.getLegalActions(agentIndex)
            agentNum = state.getNumAgents()

            # next agent's index
            if agentIndex == agentNum - 1:
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1

            # Find the best value(action), maximum
            bestValue = -10000
            actionList = []
            for action in legalMoves:
                successorState = state.generateSuccessor(agentIndex, action)
                value = minimax(successorState, depth-1, nextAgentIndex)['value']
                actionList.append({'action': action, 'value': value})
                bestValue = max(bestValue, value)

            # Choose it from the list
            index = 0
            for i in range(len(actionList)):
                if actionList[i]['value'] == bestValue:
                    index = i
                    break

            return actionList[index]


        def minValue(state, depth, agentIndex):
            """ Function of Min-Value """
            # Similar to maxValue, but opposite logic

            legalMoves = state.getLegalActions(agentIndex)
            agentNum = state.getNumAgents()

            if agentIndex == agentNum - 1:
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1

            bestValue = 10000
            actionList = []
            for action in legalMoves:
                successorState = state.generateSuccessor(agentIndex, action)
                value = minimax(successorState, depth-1, nextAgentIndex)['value']
                actionList.append({'action': action, 'value': value})
                bestValue = min(bestValue, value)

            index = 0
            for i in range(len(actionList)):
                if actionList[i]['value'] == bestValue:
                    index = i
                    break

            return actionList[index]


        # start min-max and get the result
        agentNum = gameState.getNumAgents()
        result = minimax(gameState, self.depth*agentNum, 0)
        actionEval = result['value']
        actionMove = result['action']

        # Debugging output
        # print 'Final: %s, eval: %f' % (actionMove, actionEval)

        # Stop and wait for a while
        # raw_input() 
        
        # return action
        return actionMove


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        "[Project 3] YOUR CODE HERE"    
        """ Functions of Alpha-Beta Pruning """
        def alphabeta(state, depth, alpha, beta, agentIndex):
            """ Alpha-Beta head function"""

            agentNum = state.getNumAgents()

            # Terminal-state or maximum depth
            if depth == 0 or state.isLose() or state.isWin():
                return {'value': self.evaluationFunction(state)}

            # Pacman: maxValue; Ghost: minValue
            if agentIndex == 0:
                return maxValue(state, depth, alpha, beta, agentIndex)
            else:
                return minValue(state, depth, alpha, beta, agentIndex)


        def maxValue(state, depth, alpha, beta, agentIndex):
            """ Function of Max-Value"""

            legalMoves = state.getLegalActions(agentIndex)
            agentNum = state.getNumAgents()

            # next agent's index
            if agentIndex == agentNum - 1:
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1

            # Find the best value(action), maximum
            bestValue = -10000
            actionList = []
            for action in legalMoves:
                successorState = state.generateSuccessor(agentIndex, action)
                value = alphabeta(successorState, depth-1, alpha, beta, nextAgentIndex)['value']
                bestValue = max(bestValue, value)
                alpha = max(alpha, bestValue)
                actionList.append({'action': action, 'value': value})

                # pruning 
                # original logic of alogrithm is beta <= alpha
                # but it would fail unless I changed it to <
                if beta < alpha:
                    break

            # Choose it from the list
            index = 0
            for i in range(len(actionList)):
                if actionList[i]['value'] == bestValue:
                    index = i
                    break

            return actionList[index]


        def minValue(state, depth, alpha, beta, agentIndex):
            """ Function of Min-Value """
            # Similar to maxValue, but opposite logic

            legalMoves = state.getLegalActions(agentIndex)
            agentNum = state.getNumAgents()

            if agentIndex == agentNum - 1:
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1

            bestValue = 10000
            actionList = []
            for action in legalMoves:
                successorState = state.generateSuccessor(agentIndex, action)
                value = alphabeta(successorState, depth-1, alpha, beta, nextAgentIndex)['value']
                bestValue = min(bestValue, value)
                beta = min(beta, bestValue)
                actionList.append({'action': action, 'value': value})

                # pruning
                # original logic of alogrithm is beta <= alpha
                # but it would fail unless I changed it to <
                if beta < alpha:
                    break

            index = 0
            for i in range(len(actionList)):
                if actionList[i]['value'] == bestValue:
                    index = i
                    break

            return actionList[index]  


        # start alpha-beta and get the result
        agentNum = gameState.getNumAgents()
        result = alphabeta(gameState, self.depth*agentNum, -10000, 10000, 0)
        actionEval = result['value']
        actionMove = result['action']

        # raw_input()
        return actionMove  
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        """ Functions of Expectimax """
        def expectimax(state, depth, agentIndex):
            """ Expectimax head function"""

            agentNum = state.getNumAgents()

            # Terminal-state or maximum depth
            if depth == 0 or state.isLose() or state.isWin():
                return {'value': self.evaluationFunction(state)}

            # Pacman: maxValue; Ghost: minValue
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex)
            else:
                return expValue(state, depth, agentIndex)


        def maxValue(state, depth, agentIndex):
            """ Function of Max-Value"""

            legalMoves = state.getLegalActions(agentIndex)
            agentNum = state.getNumAgents()

            # next agent's index
            if agentIndex == agentNum - 1:
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1

            # Find the best value(action), maximum
            bestValue = -10000
            actionList = []
            for action in legalMoves:
                successorState = state.generateSuccessor(agentIndex, action)
                value = expectimax(successorState, depth-1, nextAgentIndex)['value']
                actionList.append({'action': action, 'value': value})
                bestValue = max(bestValue, value)

            # Choose it from the list
            index = 0
            for i in range(len(actionList)):
                if actionList[i]['value'] == bestValue:
                    index = i
                    break

            return actionList[index]


        def expValue(state, depth, agentIndex):
            """ Function of Exp-Value """

            legalMoves = state.getLegalActions(agentIndex)
            agentNum = state.getNumAgents()

            # the probability (ghost's action is random)
            legalCount = len(legalMoves)
            probability = 1.0 / legalCount

            if agentIndex == agentNum - 1:
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1

            # bestValue: expectic value of a ghost
            bestValue = 0
            actionList = []
            for action in legalMoves:
                successorState = state.generateSuccessor(agentIndex, action)
                value = expectimax(successorState, depth-1, nextAgentIndex)['value']
                bestValue += probability * value

            return {'value': bestValue}


        # start searching and get the result
        agentNum = gameState.getNumAgents()
        result = expectimax(gameState, self.depth*agentNum, 0)
        actionEval = result['value']
        actionMove = result['action']

        return actionMove

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

    
    There are 8 features we choose to formulate the evaluation function:
    1: current state
        win or lose
    2: current score
        current state's score
    3. number of current food
    4. distance from closet food
    5. number of current capsules
    6. distance from closet capsule
    7. distance from closet active ghost
    8. distance from closet scared ghost

    """
    
    "[Project 3] YOUR CODE HERE"
    # Position Information 
    curPos = currentGameState.getPacmanPosition()
    curX, curY = [curPos[0], curPos[1]]

    # Ghost Information
    ghostState = currentGameState.getGhostStates()

    def featureState(state):
        # Win, return 100
        # Lose, return -100
        # Nothing, return 0
        if state.isWin():
            return 100
        elif state.isLose():
            return -100
        
        return 0


    def featureScore(state):
        # return current score
        return state.getScore()


    def featureScared(state, curX, curY, ghostState):
        # return distance from closest scared ghost
        minDistance = 10000
        for ghost in ghostState:
            if ghost.scaredTimer:
                ghostX = ghost.getPosition()[0]
                ghostY = ghost.getPosition()[1]
                distance = abs(curX - ghostX) + abs(curY - ghostY)
                minDistance = min(minDistance, distance)
        
        # no scared ghost
        if minDistance == 10000:
            minDistance = 0

        return minDistance


    def featureGhost(state, curX, curY, ghostState):
        # return distance from closest active ghost
        minDistance = 10000
        for ghost in ghostState:
            if not ghost.scaredTimer:
                ghostX = ghost.getPosition()[0]
                ghostY = ghost.getPosition()[1]
                distance = abs(curX - ghostX) + abs(curY - ghostY)
                minDistance = min(minDistance, distance)

        if minDistance == 10000 or minDistance == 0:
            minDistance = 0.1

        return minDistance


    def featureNumFood(state):
        # return number of food
        return state.getNumFood()


    def featureFood(state, curX, curY):
        # return distance from closest food
        foodList = currentGameState.getFood().asList()
        minDistance = 10000
        for foodPos in foodList:
            foodX, foodY = [foodPos[0], foodPos[1]]
            distance = abs(curX - foodX) + abs(curY - foodY)
            minDistance = min(minDistance, distance)

        if minDistance == 10000:
            minDistance = 0

        return minDistance

    def featureNumCapsule(state):
        # return number of capsule
        return len(state.getCapsules())


    def featureCapsule(state, curX, curY):
        # return distance from closest capsule
        capsuleList = state.getCapsules()
        minDistance = 10000
        for capsulePos in capsuleList:
            capsuleX, capsuleY = [capsulePos[0], capsulePos[1]]
            distance = abs(curX - capsuleX) + abs(curY - capsuleY)
            minDistance = min(minDistance, distance)

        if minDistance == 10000:
            minDistance = 0

        return minDistance


    w1 = 3
    f1 = featureState(currentGameState)
    w2 = 1
    f2 = featureScore(currentGameState)
    w3 = -5
    f3 = featureNumFood(currentGameState)
    w4 = -35
    f4 = featureNumCapsule(currentGameState)
    w5 = -2
    f5 = featureFood(currentGameState, curX, curY)
    w6 = -20
    f6 = featureCapsule(currentGameState, curX, curY)
    w7 = -30
    f7 = featureScared(currentGameState, curX, curY, ghostState)
    w8 = -1
    f8 = 1.0 / featureGhost(currentGameState, curX, curY, ghostState)


    evalScore = w1 * f1 \
              + w2 * f2 \
              + w3 * f3 \
              + w4 * f4 \
              + w5 * f5 \
              + w6 * f6 \
              + w7 * f7 \
              + w8 * f8 \

    # print '1: %d| 2: %d| 3: %d| 4: %d| 5: %d| 6: %2.f 7: %d| 8: %d' % \
          # (w1*f1, w2*f2, w3*f3, w4*f4, w5*f5, w6*f6, w7*f7, w8*f8)

    return evalScore   
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

