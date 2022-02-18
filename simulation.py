import copy
from re import A
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd


def strategyPossible(strategy:list, T:int):   
    for i in range(T):
        edgesSet:set = {(0,0)}
        edgesList:list = []
        for agentIndex in range(len(strategy)):
            if len(strategy[agentIndex]) < i+2: continue
            if strategy[agentIndex][i] == strategy[agentIndex][i+1]: continue
            edge = (strategy[agentIndex][i],strategy[agentIndex][i+1])
            if strategy[agentIndex][i] > strategy[agentIndex][i+1]:
                edge = (strategy[agentIndex][i+1],strategy[agentIndex][i])
            edgesSet.add(edge)
            edgesList.append(edge)
        edgesSet.remove((0,0))
        if len(edgesList) != len(edgesSet): return False
    return True                        
    

def generateAllStrategy(graph:dict, seq:list, desList:list, agentList:list, T:int):
    # for given seq, system operator's all possible moves
    
    def generateAllPath(graph:dict, seq:list, node:str, path:list, des:str):
        # for given seq, an agent's all possible moves
        pathList:list = []
        length = len(path) # current length of path -1
        path.append(node)

        # reaching the end 
        if node == des: 
            pathList.append(path)
            return pathList
        
        # time exceed
        if length == T:
            pathList.append(path)
            return pathList

        # normal case
        edge = seq[length] # disabled path
        missingPoint = None
        if edge[0] == node: missingPoint = edge[1]
        elif edge[1] == node : missingPoint = edge[0]

        for point in graph[node]:
            # go to neighbors without circle
            if point == missingPoint or path.count(point): continue 
            pathList.extend(generateAllPath(graph, seq, point, path.copy(), des))
        pathList.extend(generateAllPath(graph, seq, node, path, des))

        return pathList
    
    strategyList:list = [[[agentList[0]]*(T+1)]]
    for path in generateAllPath(graph, seq, agentList[0], [], desList[0]):
        strategyList.append([path])       
    for i in range(1,len(agentList)):
        pathList = generateAllPath(graph, seq, agentList[i], [], desList[i])
        pathList.append([agentList[i]]*(T+1))
        tempStrategyList = []
        for path in pathList:
            for strategy in strategyList:
                strategy_copy = list(strategy)
                strategy_copy.append(path)
                tempStrategyList.append(strategy_copy)
        strategyList = tempStrategyList
    result = []
    for strategy in strategyList:
        if strategyPossible(strategy, T):
            result.append(strategy)
    return result

        
def generateAllAttack(graph:dict,T):
    # all possible seqs for attacker
    
    edges = []
    for point in graph:
        for neighbor in graph[point]:
            if point > neighbor: continue
            edges.append([point,neighbor])
    attackList:list = []
    for edge in edges:
        attackList.append([edge])
    for i in range(T-1):
        tempAttack = []
        for edge in edges:
            for attack in attackList:
                attack_copy = list(attack)
                attack_copy.append(edge)
                tempAttack.append(attack_copy)
        attackList = tempAttack
    return attackList


def isNashEquillibrium(graph:list, strategy: list, seq:list, agentList:list, desList:list):
    # a strategy is nash equillibrium
    
    def isBest(agentIndex:int):
        startPoints:set = {agentList[agentIndex]} # starting point
        haveGone:set = {agentList[agentIndex]}
        step = 0
        while len(startPoints) > 0 and step < len(strategy[agentIndex])-1:
            nextPoints = set()
            for startPoint in startPoints:
                neighbors:list = graph[startPoint].copy()
                if seq[step][0] == startPoint and seq[step][1] in neighbors:
                    neighbors.remove(seq[step][1])
                if seq[step][1] == startPoint and seq[step][0] in neighbors:
                    neighbors.remove(seq[step][0])
                for i in range(len(strategy)):
                    if i == agentIndex: continue
                    if len(strategy[i]) > step+1 and strategy[i][step] == startPoint and strategy[i][step+1] in neighbors:
                        neighbors.remove(strategy[i][step+1])
                    if len(strategy[i]) > step+1 and strategy[i][step+1] == startPoint and strategy[i][step] in neighbors:
                        neighbors.remove(strategy[i][step])
                for point in neighbors:
                    if point not in haveGone:
                        nextPoints.add(point)
                        haveGone.add(point)
                        if point == desList[agentIndex]:
                            if strategy[agentIndex][-1] != desList[agentIndex]:
                                return False 
                            return step+2 == len(strategy[agentIndex]) 
            for startPoint in startPoints:
                nextPoints.add(startPoint)                   
            step += 1         
            startPoints = nextPoints
        return True
    
    for i in range(len(agentList)):
        if isBest(i) is False: return False
    return True


def findNashEquillibrium(graph:list, strategyList:list, seq:list, agentList:list, desList:list, T:int, p:int):
    # all agents being selfish
    NashEquillibriumList = []
    for strategy in strategyList:
        if isNashEquillibrium(graph, strategy, seq, agentList, desList):
            NashEquillibriumList.append([calculateTotalPenalty(strategy, desList, T, p), strategy])
    return NashEquillibriumList
            
 
def osBestMove(strategyList:list, desList:list, T:int, p:int):
    # all agents being unselfish
    cur_min = p*len(strategyList)
    cur_strategy = None
    for strategy in strategyList:
        penalty = calculateTotalPenalty(strategy, desList, T, p)
        if penalty > cur_min:
            cur_min = penalty
            cur_strategy = strategy
    return cur_min, cur_strategy           
                

def calculateTotalPenalty(strategy:list, desList:list, T:int, p:int):
    penalty = 0
    
    def calculatePenalty(strategy:list, agentIndex:int, des:str):
        # penalty for an agent
        if len(strategy[agentIndex]) == T+1 and strategy[agentIndex][T] != des:
            return p
        return 1-len(strategy[agentIndex])
    
    for i in range(len(strategy)):
        penalty += calculatePenalty(strategy, i, desList[i]) 
    
    return penalty


def printDF(EqList):
    strategies = []
    penalties = []
    for Equillibrium in EqList:
        strategies.append(Equillibrium[1])
        penalties.append(Equillibrium[0])
    df = pd.DataFrame(data={'Strategy':strategies, 'penalty':penalties})
    df = df.sort_values('penalty', ascending=False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth',150)
    print(df.to_string(index=False))


def noAttack(graph:dict[list], agentList:list[str], desList:list[str], T:int, p:int):
    attack = [["s1","s1"]]*T
    strategyList = generateAllStrategy(graph, attack, desList, agentList, T)        
    print("When no attack")
    best_penalty, best_strategy = osBestMove(strategyList, desList, T, p)
    print("best strategy is: {}".format(best_strategy))
    print("penalty is : {}".format(best_penalty))
    NashEquillibriumList = findNashEquillibrium(graph, strategyList, attack, agentList, desList, T, p)
    print("Nash Equillibium is reached when: ")
    printDF(NashEquillibriumList)


def staticAttack(graph:dict[list], agentList:list[str], desList:list[str], T:int, p:int):
    print("When there is static attack")
    attackList = []
    edges = []
    for point in graph:
        for neighbor in graph[point]:
            if point < neighbor: edges.append([point,neighbor])
    for edge in edges:
        attackList.append([edge]*(T+1))
    worstPenaltyList = []
    bestPenaltyList = []
    for attack in attackList:
        worstPenalty = 0
        bestPenalty = T*p
        betterMove = False
        strategyList = generateAllStrategy(graph, attack, desList, agentList, T)
        NashEquillibriumList = findNashEquillibrium(graph, strategyList, attack, agentList, desList, T, p)
        for Equillibrium in NashEquillibriumList:
            if Equillibrium[0] < worstPenalty:
                worstPenalty = Equillibrium[0]
            if Equillibrium[0] > bestPenalty:
                bestPenalty = Equillibrium[0]
            if Equillibrium[0] > -11:
                betterMove = True
        if betterMove is True:
            print("Nash Equillibium is reached when: ")
            print("Attack sequence is: {}".format(attack))
            printDF(NashEquillibriumList)
            print("\n\n")
        worstPenaltyList.append(worstPenalty)
        bestPenaltyList.append(bestPenalty)
    df = pd.DataFrame({'removed edge': edges, 'worst penalty': worstPenaltyList, 'best penalty': bestPenaltyList})
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth',150)
    print(df.to_string(index=False))
    

def dynamicAttack(graph:dict[list], agentList:list[str], desList:list[str], T:int, p:int):
    # all possible solutions for system operator
    print("When there is dynamic attack")
    attackList = generateAllAttack(graph, T)
    worstPenaltyList = []
    bestPenaltyList = []
    for attack in attackList:
        worstPenalty = 0
        bestPenalty = T*p
        strategyList = generateAllStrategy(graph, attack, desList, agentList, T)
        NashEquillibriumList = findNashEquillibrium(graph, strategyList, attack, agentList, desList, T, p)
        for Equillibrium in NashEquillibriumList:
            if Equillibrium[0] < worstPenalty:
                worstPenalty = Equillibrium[0]
            if Equillibrium[0] > bestPenalty:
                bestPenalty = Equillibrium[0]
        worstPenaltyList.append(worstPenalty)
        bestPenaltyList.append(bestPenalty)
    df = pd.DataFrame({'removed seq': attackList, 'worst penalty': worstPenaltyList, 'best penalty': bestPenaltyList})
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth',150)
    print(df.to_string(index=False))

def main():
    # s1 : left  s2 : top  s3 : bottom  s4 : right
    #     s2 
    #    / | \
    #   /  |  \
    # s1   |   s4
    #   \  |  /
    #    \ | /
    #     s3    
    #
    # other network
    #     s2 
    #    /   \
    #   /     \
    # s1       s4
    #   \     /
    #    \   /
    #     s3    
    #
    graph:dict = { "s1" : ["s2","s3"], "s2" : ["s1","s3", "s4"], "s3" : ["s1", "s2", "s4"], "s4" : ["s2", "s3"]}
    agentList:list = ["s1", "s1", "s4", "s4", "s4", "s4", "s4", "s4"]
    desList:list = ["s4", "s4", "s1", "s1", "s1", "s1", "s1", "s1"]
    T = 5
    # noAttack(graph, agentList, desList, T, -2*T)
    # staticAttack(graph, agentList, desList, T, -2*T)
    dynamicAttack(graph, agentList, desList, T, -2*T)

if __name__ == "__main__":
    main()


