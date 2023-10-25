import numpy as np
import math
import sys
from queue import PriorityQueue



sys. setrecursionlimit(50000)


#cost_matrix and 
class Dijkstra_algorithm:
    def __init__(self,height, width, target_locations, check_for_obstacle):
        self.cost_matrix = np.full((height, width), np.inf) # to make it modular,grid is considered to be size of the current scenartio.
        self.width=width
        self.height=height
        self.target_locations=target_locations
        self.check_for_obstacle=check_for_obstacle




    def execute(self):

        queue=PriorityQueue()
        sqrt2=round(math.sqrt(2),2)

        for target in self.target_locations:
            x,y = target[0], target[1]
            self.cost_matrix[x][y]=0
            queue.put((0,(x,y)))


        while not queue.empty():
            cost,(x,y) = queue.get()
            for i in [-1,0,1]: #will be used for x
                for j in [-1,0,1]: # will be used for y
                    if i==0 and j==0: continue #skip the current point
                    if x+i>=0 and x+i<self.height and y+j>=0 and y+j<self.width and not self.check_for_obstacle[x+i][y+j]:
                        if (i*j!=0): #if true, then diagonal
                            if self.cost_matrix[x+i][y+j] >sqrt2+cost : 
                                self.cost_matrix[x+i][y+j]=sqrt2+cost #diagonal moves are sqrt(2) long.
                                queue.put((sqrt2+cost, (x+i,y+j))) 
                        else:  
                            if self.cost_matrix[x+i][y+j] > 1+cost :
                                self.cost_matrix[x+i][y+j]=1+cost
                                queue.put((1+cost,(x+i,y+j)))


        self.cost_matrix = self.cost_matrix.T



    def estimate_cost(self,x,y): #Warning: This does not consider interaction between people, only cost for dijkstra's algorithm
        
        if x%1==0 and y%1==0:
            print(self.cost_matrix[x,y],x,y)
            return self.cost_matrix[x,y]
        elif x%1==0:
            distance1=abs(math.ceil(y)-y)
            distance2=abs(math.floor(y)-y)

            effect1= self.cost_matrix[x]  [math.ceil(y)]   /  distance1
            effect2=  self.cost_matrix[x] [math.floor(y)] /  distance2
            
            return (effect1+effect2) / (1/distance1+1/distance2)


        elif y%1==0:
            distance1=abs(math.ceil(x)-x)
            distance2=abs(math.floor(x)-x)

            effect1= self.cost_matrix[math.ceil(x)]  [y]   /  distance1
            effect2=  self.cost_matrix[math.floor(x)] [y] /  distance2
            
            return (effect1+effect2) / (1/distance1+1/distance2)

        
        else:
            distance1= math.sqrt((math.ceil(x)-x )**2 +   (math.ceil(y)-y)**2)
            distance2= math.sqrt((math.floor(x)-x)**2 +   (math.ceil(y)-y)**2)
            distance3= math.sqrt((math.ceil(x)-x )**2 +   (math.floor(y)-y)**2)
            distance4= math.sqrt((math.floor(x)-x)**2 +   (math.floor(y) -y)**2)

            effect1= self.cost_matrix[math.ceil(x)]  [math.ceil(y)]   /  distance1
            effect2=  self.cost_matrix[math.floor(x)] [math.floor(y)] /  distance2
            effect3= self.cost_matrix[math.ceil(x)]  [math.ceil(y)]   /  distance3
            effect4= self.cost_matrix[math.floor(x)] [math.floor(y)]  /  distance4

            return (effect1+effect2+effect3+effect4)/(1/distance1+1/distance2+1/distance3+1/distance4)
