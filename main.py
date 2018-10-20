import pandas as pd
import numpy as np
import csv
import math
from collections import defaultdict
from collections import Counter
import operator
import sys
import random
import copy

''' This nodes of the decision tree belong to the class TreeNode '''

class TreeNode(object):
	def __init__(self, attribute, parentValue, nodeNo = 0, depth = 0):
		self.nodeId = nodeNo
		self.nodeName = attribute
		self.parentNode = parentValue
		self.childrenDict = {}
		self.depth = depth
		self.valueDict = {}
		self.entropy = None
		self.infoGain = None	
		self.indices = []
		self.nodeData = {}	
		self.classData = []
		self.childName = {}
		self.classValue = None
		self.parent = None
		self.weightData = []

class DecisionTree(object):


	''' This is the constructor for the decision tree. It initialises the path for the train and test data '''
	def __init__(self, maxDepth, trainPath, testPath= None, className = None):
		self.dictDF = self.readFile(trainPath)
		self.classVar = self.dictDF[className].values()
		self.className = className
		del self.dictDF[className]
		self.maxDepth = maxDepth		
		self.nodeQueue = []
		self.nodeStack = []
		self.trainDF = pd.DataFrame.from_dict(self.dictDF)
		self.testDict = self.readFile(testPath)	
		self.testData = pd.DataFrame.from_dict(self.testDict)
		self.rootNode = None
		self.className = className
		self.trainPath = trainPath
		self.className = className
		self.testPath = testPath
	
	''' get the data with specifoed indices'''
	def getData(self, dataDict, indices):
		newDict = {}
		for attribute in dataDict.keys():
			data = dataDict[attribute].values()
			data = [data[position] for position in indices]
			newDict[attribute] = dict(enumerate(data))
		return (newDict)	

	'''This function predicts using recursion and goes to the depth until it finds the nodes do not have any further children  '''

        def Predict(self,rootNode, row, data):
                if(not rootNode.childrenDict):
                        return rootNode.classValue
                childDict = rootNode.childrenDict
                value = data[rootNode.nodeName][row]
		if value in rootNode.childrenDict.keys():
	                attribute = rootNode.childrenDict[value]
        	        return(self.Predict(attribute, row, data))
		else:
			return rootNode.classValue

	''' This function is used to print the confusion matrix  '''
	def confusionMatrix(self, t, obs):
		true_pos = 0
		true_neg = 0
		false_neg = 0
		false_pos = 0
		for i in xrange(0,len(obs)):
			if (obs[i] ==1):
				if(t[i] ==1):true_pos +=1
				else:false_pos +=1
			else:
				if(t[i] == 0):true_neg +=1
				else:false_neg +=1
		Accuracy = ((true_pos + true_neg)*1.0)/(len(obs))
		error = 1 - Accuracy
		print "Predicted\t|  0\t1"
                print "-----------------------------"
                print "Actual\t\t|"
                print "0\t\t|" , true_neg,"\t", false_pos
                print "1\t\t|" , false_neg,"\t", true_pos
		print "Accuracy: ",Accuracy
		print "Error: ", error
		print "misclassifications:" , false_pos+false_neg


	'''This function is used to print the most common element. It is used to get the class value for each node. It returns the value which has maximum occurances  '''
	def mostCommonItem(self, numList):
		d = defaultdict(int)
		for i in numList:
			d[i] += 1
		result = max(d.iteritems(), key=lambda x: x[1])[0]
		return(result)

	''' This function is used ot read the file and return the dictionary '''
	def readFile(self, pathCSV):
		df = pd.read_csv(pathCSV)
		d = df.to_dict()
		return (d)


	''' This function is used to calculate the entropy '''
	def calcEntropy(self, a):
		Calcsum = float(sum(a))
		entropy = 0
		for x in range(len(a)):
			prob = (a[x]*1.0)/Calcsum
			logCalc = math.log(prob,2)
			entropy+= -(prob * logCalc)
		return(entropy)

	''' This function is used to calculate the two attribute entropy  '''
	def twoAttrEntropy(self, att, classV):
		classEntropy = 0
		mergeList = zip(att, classV)
		dictDef = defaultdict(list)
		for attribute, classvar in mergeList:
			dictDef[attribute].append(classvar)
		count = Counter(dictDef)
		attrKeys = dictDef.keys()
		for x in attrKeys:
			dictX= Counter(dictDef[x])
			classValues = dictX.keys()
			classCount = dictX.values()
			totalX = sum(classCount)
			probX = totalX*1.0/len(att)
			classEntropy += probX * self.calcEntropy(dictX.values())
		return (classEntropy)

	''' This function is used to calculate the information gain at each node '''
	def infoGain(self, attribute, classVar, parent):
		gain = 0
		ent1 = self.calcEntropy(Counter(parent.classData).values())
		ent2 = self.twoAttrEntropy(attribute, classVar)
		gain = self.calcEntropy(Counter(parent.classData).values()) - self.twoAttrEntropy(attribute, classVar)
		return(gain)

	''' This function is used for attribute selection at each point using the entropy '''
	def attributeSelection(self, dataDict, weights, classVar, parentNode = None):
		attributes = dataDict.keys()
		entropyDict = {}
		if (parentNode != None) : attributes.remove(parentNode)
		for x in attributes:
			attData = dataDict[x].values()
			entropyValue = self.twoAttrEntropy_Weighted(attData, classVar, weights)
			entropyDict[x] = entropyValue
		minAttr = min(entropyDict.iteritems(), key=operator.itemgetter(1))[0]
		entropy = min(entropyDict.iteritems(), key=operator.itemgetter(1))[1]
		return (minAttr, entropy)		

	''' This function is used to create the rest of the nodes of the tree until the depth specified by the user '''

	def createNodesBagging(self, rootNode, dictDF, nodeStack):
                if (not nodeStack):
                        return rootNode
                classes = set(dictDF[rootNode.nodeName].values())
                for value in classes:
                                indices = [i for i, x in enumerate(rootNode.nodeData[rootNode.nodeName].values()) if x == value]
                                data = self.getData(rootNode.nodeData, indices)
                                classData = [rootNode.classData[position] for position in indices]
				weightData =[rootNode.weightData[position] for position in indices]
                                rootNode.valueDict[value] = indices
                                if (rootNode.depth < self.maxDepth and classData):
                                        childAttribute, ent = self.attributeSelection(data, weightData, classData, rootNode.nodeName)
                                        gain = rootNode.entropy - ent
                                        node = TreeNode(childAttribute, rootNode.nodeName, 0, 1)
                                        node.indices = indices
                                        node.nodeData = data
                                        rootNode.childrenDict[value] = node
                                        node.depth = rootNode.depth + 1
                                        node.classData = classData
                                        node.parentNode = rootNode
                                        node.childName[value] = childAttribute
					node.weightData =weightData
                                        node.infoGain = gain
                                        node.entropy = ent
                                        if not classData:
                                                node.classValue = rootNode.classValue
                                        else:
                                                node.classValue = self.mostCommonItem(node.classData)
                                                nodeStack.append(node)
                return(self.createNodesBagging(nodeStack.pop(), dictDF, nodeStack))	

	''' This function is used to create root nodes.'''
	
        def createRootBagging(self, dictDF, classVar, weights,depth = 0): 
		nodeQueue = []
		nodeStack = []
		
                if (depth == 0): 
                        (root, ent) = self.attributeSelection(dictDF , weights,classVar)
                        rootNode = TreeNode(root , None, 0, 1)
                        rootNode.entropy = ent 
                        classes = set(dictDF[root].values())
                        rootNode.nodeData = dictDF
                        rootNode.classData = classVar
                        rootNode.classValue = 0 
			rootNode.weightData = weights
                        nodeQueue.append(rootNode)
                        nodeStack.append(rootNode)
                        rootNode.depth = 0 
                self.createNodesBagging(rootNode,dictDF, nodeStack )
		return rootNode
  

        '''This function is used for predition and calls another function to print the confusion matrix for the prediction  '''
 
	def PredictOnTestBagging(self,data, rootNode, classList):
                data['pred'] = -1
                for i in xrange(0,data.shape[0]):
                        pv = self.Predict(rootNode, i, data)
                        data['pred'][i] = pv
		return data['pred']
	
	''' This function takes in the list of all the root nodes and gives out the prediction for the class which has maximum votes'''

	def predictBagging(self, modelLis, testDict, classVar):
		predictedDict = []
		predResult = []
		for node in modelLis:
			predictedDict.append(self.PredictOnTestBagging(pd.DataFrame.from_dict(testDict),node, classVar))
		for l in xrange(0,len(predictedDict[0])):
			lis = []
			for k in xrange(0, len(predictedDict)):
				lis.append(predictedDict[k][l])
			p = self.mostCommonItem(lis)
			predResult.append(p)
		self.confusionMatrix(classVar, predResult)
		return predResult

	''' This is the main Function which is used start the Bagging operation. It first deleted the unwanted columns. It created the number of models as specified and '''
	def learn_bagged(self, nummodels = 5, depth = 3, dataPath = None):
		print "************* BAGGING ***************"
		self.maxDept = depth
		print "DEPTH: ", self.maxDept
		print "Number of Models: ", nummodels
		trainDF = self.readFile(self.trainPath)
		del trainDF['bruises?-no']
		testDict = self.readFile(self.testPath)
		trainPandaDF = pd.DataFrame.from_dict(trainDF)
		l = len(trainPandaDF)
		lis = []
		modelLis = []
		weights = []
		for i in xrange(0, len(trainPandaDF)):
			weights.append(1)
		while(nummodels > 0):
			for i in xrange(0,l):
				lis.append(i)
			randomId = [random.choice(lis) for _ in range(int(0.6*l))]
			selectedData = self.getData(trainDF,randomId)
			classVar = selectedData[self.className].values()
			del selectedData[self.className]
			node = self.createRootBagging(selectedData, classVar, weights)
			modelLis.append(node)
			nummodels -=1
		self.predictBagging(modelLis, testDict, testDict[self.className])
	

	''' This function is used to update the weights in Boosting after every Iteration.'''
	def update_weights(self, weights, actClass, predClass):
		a = copy.copy(actClass)
		b= copy.copy(predClass)
		for i in xrange(0, len(actClass)):
			if a[i] == 0:
				a[i] = -1	
			if b[i] == 0:
				b[i] = -1
		pError = 0
		# Beelow is one of the method to calculate the Error
		'''for i in xrange(0, len(actClass)):
			pError = pError + 1.0* weights[i]* a[i] * b[i]
		pError = pError/2
		Error = 0.5 - pError'''
		sumNeg = 0
		for i in xrange(0, len(actClass)):
			if actClass[i] != predClass[i]:
				sumNeg = sumNeg + weights[i]
		Error = (sumNeg * 1.0)/sum(weights)
		alpha = ((1-Error)/Error)
		alphaLog = 0.5 * math.log(alpha,2)
		sumWeights = sum(weights)
		for i in xrange(0, len(weights)):
			weights[i] = (1.0/sumWeights)*weights[i]* math.exp((-1)* a[i] * alphaLog * b[i])
		return weights, alphaLog

	'''This function is used to calculate the prior weighted Entropy '''
	def calc_WeightedEntropy(self, a, weights):
                Calcsum = float(sum(a))
		TotalWeights = sum(weights)
                entropy = 0
                for x in range(len(a)):
                        prob = (a[x]*1.0)/Calcsum
                        logCalc = math.log(prob,2)
                        entropy+= -(prob * logCalc)
                return(entropy)

	'''This function is used to calculate the two attribute weighted Entropy. '''
	def twoAttrEntropy_Weighted(self, att, classV, weights):
                classEntropy = 0
                mergeList = zip(att, classV, weights)
		totWt = sum(weights)
                dictDef = {}
		for attribute, classvar, weights in mergeList:
                        dictDef[attribute] = {}
		for attribute, classvar, weights in mergeList:
                        dictDef[attribute][classvar] = 0
                for attribute, classvar, weights in mergeList:
                        dictDef[attribute][classvar] +=weights
                attrKeys = dictDef.keys()
                for x in attrKeys:
			dictX = dictDef[x]
                        classValues = dictX.keys()
                        classCount = dictX.values()
                        totalX = sum(classCount)
                        probX = totalX*1.0/len(att)
                        classEntropy += probX * self.calcEntropy(dictX.values())
                return (classEntropy)

	'''This is the main function which is called to predict the value of Boosting '''
	def predict_boost(self, modelLis, alphaLis, testDict, classVar):
		predictedLis = []
                predResult = []
                for node in modelLis:
                        predictedLis.append(self.PredictOnTestBagging(pd.DataFrame.from_dict(testDict),node, classVar))
		for l in xrange(0,len(predictedLis[0])):
                        lis = []
                        for k in xrange(0, len(predictedLis)):
                                lis.append(predictedLis[k][l])
                        p = self.predictValueBoost(lis,alphaLis)
                        predResult.append(p)
                self.confusionMatrix(classVar, predResult)
                return predResult

	''' calculating the value for each row '''
	def predictValueBoost(self, allTree, alpha):
		total = 0
		a = copy.copy(allTree)
		for i in xrange(0, len(a)):
			if a[i] == 0:
				a[i] = -1
			total = total + alpha[i] * a[i]
		return self.sign(total)

	''' returns the value 1 if its greater than 0 else returns 0'''
	def sign(self, num):
		if num >=0:
			return 1
		if num < 0:
			return 0

	''' This function is first called for boosting. It deletes the unwanted Column from Mushroom Dataset, It creates the number of models as specified. All the trees and their corresponding alpha values are then stored in a list and passed to the function predict_boost() '''

	def learn_boosted(self, nummodels = 10, depth = 2, dataPath = None):
		print "************* BOOSTING ***************"
		print "Depth : ", depth
		print "Bags: ", nummodels
		self.maxDept = depth
		bags = 2
		trainDF = self.readFile(self.trainPath)
		testDict = self.readFile(self.testPath)		
		del trainDF['bruises?-no']
		trainPandaDF = pd.DataFrame.from_dict(trainDF)
		classVar = trainDF[self.className].values()
		del trainDF[self.className]
		lenTrain = len(trainPandaDF)
		bagsLis = []
		weights = []
		for i in xrange(0,lenTrain):    
                        w = (1*1.0/lenTrain)
       	                weights.append(w)
		node = self.createRootBagging(trainDF, classVar,weights)
		modelCount = nummodels
		bags -=1
		modelLis = []
		alphaLis = []
		while(modelCount > 0):
			modelCount -= 1
			node = self.createRootBagging(trainDF, classVar,weights)
			prediction = self.PredictOnTestBagging(pd.DataFrame.from_dict(trainDF),node, classVar)
			weights, alpha = self.update_weights(weights, classVar, prediction)
			modelLis.append(node)
			alphaLis.append(alpha)
		self.predict_boost(modelLis, alphaLis, testDict, testDict[self.className])

def main():
	sys.setrecursionlimit(50000)
	entype = sys.argv[1]
	tdepth = int(sys.argv[2])
	nummodels = int(sys.argv[3])
	datapath = sys.argv[4]
	tree = DecisionTree(maxDepth = tdepth, trainPath = datapath+"/agaricuslepiotatrain1.csv",testPath= datapath+"/agaricuslepiotatest1.csv" ,className = "bruises?-bruises")
	if entype == "bag":	
		tree.learn_bagged(nummodels,tdepth, dataPath= datapath)
	else:
		tree.learn_boosted(nummodels = nummodels, depth = tdepth, dataPath= datapath)

if __name__ == '__main__':
main()
