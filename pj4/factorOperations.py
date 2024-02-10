# factorOperations.py
# -------------------
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

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

class joinHelper():
    """ joinFactors helper."""
    def __init__(self, d: dict):
        """ d is a dict whose value is list"""
        self.currentIdx = []
        self.maxIdx = []
        self.dat = d
        for key in d:
            self.currentIdx.append(0)
            self.maxIdx.append(len(d[key]))
    
    def next(self):
        """ 
        return the next processed dict.
        >>> helper = joinHelper({'a': [1, 2], 'b': [3, 4]})
        >>> helper.next()
        {'a': 1, 'b': 3}
        >>> helper.next()
        {'a': 1, 'b': 4}
        >>> helper.next()
        {'a': 2, 'b': 3}
        >>> helper.next()
        {'a': 2, 'b': 4}
        >>> helper.next()
        None
        """
        if self.currentIdx[0] >= self.maxIdx[0]:
            # no more dicts can be evicted!
            return None
        res = {}
        i = 0
        for key in self.dat:
            res[key] = self.dat[key][self.currentIdx[i]]
            i += 1
        # increment current idx.
        i = len(self.currentIdx) - 1
        while i > 0:
            self.currentIdx[i] += 1
            if self.currentIdx[i] == self.maxIdx[i]:
                self.currentIdx[i] = 0
            else:
                return res 
            i -= 1
        # deal with i = 0.
        # get to this position means that we should increment the msb...
        self.currentIdx[0] += 1
        return res

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    unconditionals = set()
    conditionals = set()
    domain = {}
    for factor in factors:
        for u in factor.unconditionedVariables():
            unconditionals.add(u)
        for c in factor.conditionedVariables():
            conditionals.add(c)
        domainDict = factor.variableDomainsDict()
        for key in domainDict:
            domain[key] = domainDict[key]
    for var in unconditionals & conditionals:
        conditionals.remove(var)
    res = Factor(unconditionals, conditionals, domain)
    # raiseNotDefined()
    helper = joinHelper(domain)
    d = helper.next()
    while d is not None:
        prob = 1.0
        for factor in factors:
            prob *= factor.getProbability(d)
        res.setProbability(d, prob)
        d = helper.next()
    "*** END YOUR CODE HERE ***"
    return res

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        unconditionals = set()
        conditionals = set()
        domain = {}
        queryDomain = {}
        for u in factor.unconditionedVariables():
            unconditionals.add(u)
        for c in factor.conditionedVariables():
            conditionals.add(c)
        unconditionals.remove(eliminationVariable)
        domainDict = factor.variableDomainsDict()
        for key in domainDict:
            queryDomain[key] = domainDict[key]
            if key != eliminationVariable:
                domain[key] = domainDict[key]
         
        res = Factor(unconditionals, conditionals, domain)
        helper = joinHelper(queryDomain)
        d = helper.next() 
        while d is not None:
            res.setProbability(d, factor.getProbability(d) + res.getProbability(d))
            d = helper.next()
        # raiseNotDefined()
        "*** END YOUR CODE HERE ***"
        return res

    return eliminate

eliminate = eliminateWithCallTracking()

