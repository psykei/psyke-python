from scipy.optimize import minimize
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pylab as pl
import networkx as nx
import warnings



class Node:
    
    def __init__(self):
        self.reference = None
        self.left = None
        self.right = None
        self.feature = None
        self.value = None
        self.cluster = None

    def isLeaf(self):
        return self.cluster != None

class IMM(object):
    
    def __init__(self,k):
        self.k = k        #number of clusters
        self.tree = None    #will be the created tree
        self.nodeCount = 0
    
    def fit(self,x):
        
        #perform kmeans clustering on dataset        
        clt = KMeans(n_clusters = self.k, random_state = 0)
        clusterK = clt.fit(x)
        
        u = np.array(clusterK.cluster_centers_)   #clusters' centers
        y = np.array(clusterK.labels_)    #clusters' labels
            
        self.tree = self.build_tree(x, y, u)
            
    def build_tree(self,x,y,u):
        
        #check if array is homogenous (i.e. all the elements are equal)
        if np.all(y == y[0]):
            leaf = Node()
            leaf.reference = self.nodeCount
            self.nodeCount += 1
            leaf.cluster = y[0]
            return leaf
        
        else:
            dataDim = len(x)
            n_features = len(x[0])

            #populate arrays of l and r
            l = np.zeros(n_features)
            r = np.zeros(n_features)
            
            for i in range(n_features):
                arr = np.array([u[y[j]][i] for j in range(dataDim)]) #getting the i-th feature from the y[j]-th center
                l[i] = np.amin(arr)
                r[i] = np.amax(arr)
            
            

            optimizedMistakes = []  #at index i will it contain minimalOfMistakes on feature i
            optimalTheta = []   #at index i will contain the value of theta that ensures minimalOfMistakes[i] mistakes for feature i
            initialGuesses = np.vstack([l[:], r[:]]).mean(axis = 0) #initial guesses

            for i in range(n_features):
                optimizedResult = minimize(fun=self.sum_mistakes, 
                                          x0=(initialGuesses[i]), 
                                          args=(x, u, y, i),
                                          bounds=[(l[i], r[i])]
                                          )
                
                optimizedMistakes.append(optimizedResult.fun) #value of the objective function
                optimalTheta.append(optimizedResult.x[0])  #best theta                      
                                        
            
            i = np.argmin(optimizedMistakes)
            theta = optimalTheta[i]

            M = []
            L = []
            R = []
            
            for j in range(dataDim):
                if self.mistake(x[j], u[y[j]], i, theta) == 1:
                    M.append(j)
                elif x[j][i] <= theta:
                    L.append(j)
                elif x[j][i] > theta:
                    R.append(j)
            
            leftx = []
            lefty = []
            
            for e in range(dataDim):
                if e in L:
                    leftx.append(x[e])
                    lefty.append(y[e])
            
            rightx = []
            righty = []
            
            for e in range(dataDim):
                if e in R:
                    rightx.append(x[e])
                    righty.append(y[e])
            
            node = Node()
            node.reference = self.nodeCount
            self.nodeCount += 1
            node.feature = i
            node.value = theta
            node.left = self.build_tree(leftx, lefty, u)
            node.right = self.build_tree(rightx, righty, u)
            
            return node
             
    def mistake(self, x , u, i, theta):
        if (x[i] <= theta) != (u[i] <= theta):
            return 1
        else:
            return 0
    
    def sum_mistakes(self, theta, x, u, y, i):
        return sum([self.mistake(x[j], u[y[j]], i, theta) for j in range(len(x))])
    
    def predict(self, x):
        
        if self.tree is None:
            raise TypeError('Model is untrained.')
        else:
            return self.find_cluster(self.tree, x) 
        
    def find_cluster(self, tree, x):
        
        feature = tree.feature
        value = tree.value
        
        if tree.cluster is not None:
            return tree.cluster
        
        if x[feature] <= value:
            return self.find_cluster(tree.left, x)
        else:
            return self.find_cluster(tree.right, x)
    
    def explore(self, node):
        nodeList = [node]
        edgeList = []

        if node.cluster != None:
            return nodeList, edgeList
        else:
            nodeBelowListLeft, edgeBelowListLeft = self.explore(node.left)
            nodeBelowListRight, edgeBelowListRight = self.explore(node.right)

            nodeList = nodeList + nodeBelowListLeft + nodeBelowListRight
            edgeList = [(node, node.left), (node, node.right)] + edgeBelowListLeft + edgeBelowListRight

            return nodeList, edgeList
            
    def drawTree(self):
        if self.tree is None:
            raise TypeError('Model is untrained.')
        else:
            warnings.filterwarnings("ignore", category=DeprecationWarning) #to draw the tree with no deprecation warning

            allNodes, allEdges = self.explore(self.tree)
            nodeDict = {i.reference: i for i in allNodes}

            #edges' labels are "less than or equal" (<=) and "greater than" (>)
            edgeLabelDict = {(e[0].reference, e[1].reference): "<=" if e[1]==e[0].left else ">" for e in allEdges}

            #Create the tree as a networkx Graph
            T = nx.Graph([(i[0].reference,i[1].reference) for i in allEdges])

            pos = nx.drawing.nx_pydot.graphviz_layout(T, prog="dot")  #extract positions where nodes will be displayed
            pl.figure(figsize=(2*self.k,1.5*self.k)) 


            labels = {}
            for i, node in nodeDict.items():
                if node.isLeaf():
                    labels[i] = f"Cluster {node.cluster}" #leaves' lables contain clusters 
                else:
                    labels[i] = f"Feature {node.feature}\n\u03B8 = {node.value:.2f}"  #inner nodes' lables contain the feature and the threshold of the split

            #draw nodes
            nx.draw(T, 
                    pos=pos, 
                    with_labels=True,  
                    labels=labels, 
                    node_shape="s",  
                    node_color="none", 
                    bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))

            #draw edge labels
            nx.draw_networkx_edge_labels(T, 
                                        pos,
                                        edge_labels=edgeLabelDict,
                                        font_color='blue',
                                        rotate=False)

            pl.axis("off")  # turn off frame
            pl.show()

            warnings.resetwarnings()    #reset warning to default 