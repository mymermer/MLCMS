import numpy as np
import math
import sys


sys. setrecursionlimit(50000)


#cost_matrix and 
class Dijkstra_algorithm:
    def __init__(self,height, width, target_locations, check_for_obstacle):
        self.cost_matrix = np.full((height, width), np.inf) # to make it modular,grid is considered to be size of the current scenartio.
        self.width=width
        self.height=height
        self.target_locations=target_locations
        self.check_for_obstacle=check_for_obstacle




    def expanding_point_initlizer(self):
        for target in self.target_locations:
            print("next target")
            x,y = target[0], target[1]
            self.cost_matrix[x][y]=0
            for i in [-1,0,1]: #will be used for x
                for j in [-1,0,1]: # will be used for y
                    if i==0 and j==0: continue #skip the target point
                    if x+i>=0 and x+i<self.height and y+j>=0 and y+j<self.width and not self.check_for_obstacle[x+i][y+j]:
                        if (i*j!=0): #if true, then diagonal
                            if self.cost_matrix[x+i][y+j] > round(math.sqrt(2),2): #we found a shorter path!!
                                self.cost_matrix[x+i][y+j]=round(math.sqrt(2),2) #diagonal moves are sqrt(2) long.
                                self.expanding_point(x+i,y+j,round(math.sqrt(2),2))  #we will start recursive for this point
                            else: pass #this point is already visited, for smaller number so we don't need to look at it again and again.
                        else:  
                            if self.cost_matrix[x+i][y+j] > 1 : #we found a shorter path!!
                                self.cost_matrix[x+i][y+j]=1
                                self.expanding_point(x+i,y+j,1)
                            else: pass #this point is already visited, for smaller number so we don't need to look at it again and again.

        
            
            

    def expanding_point(self,x,y, current_cost): # need to be recursive, because number of next point will increase
        # if(current_cost>self.width*math.sqrt(2)): 
        #     print(self.cost_matrix)
        for i in [-1,0,1]: #will be used for x
                for j in [-1,0,1]: # will be used for y
                    if i==0 and j==0: continue #skip the current point, it already calculated

                    if x+i>=0 and x+i<self.height and y+j>=0 and y+j<self.width and not self.check_for_obstacle[x+i][y+j]:
                    
                        if (i*j!=0): #if true, then diagonal
                            if self.cost_matrix[x+i][y+j] > round(math.sqrt(2),2)+ current_cost: #we found a shorter path!!
                                self.cost_matrix[x+i][y+j]= round(math.sqrt(2),2) + current_cost #diagonal moves are sqrt(2) long.
                                self.expanding_point(x+i,y+j,round(math.sqrt(2),2) + current_cost)  #we will start recursive for this point
                            else: pass #this point is already visited, for smaller number so we don't need to look at it again and again.
                        else:  
                            if self.cost_matrix[x+i][y+j] > 1 + current_cost: #we found a shorter path!!
                                self.cost_matrix[x+i][y+j]=1+current_cost
                                self.expanding_point(x+i,y+j,1+current_cost)
                            else: pass #this point is already visited, for smaller number so we don't need to look at it again and again.



    def estimate_cost(self,x,y): #Warning: This does not consider interaction between people, only cost for dijkstra's algorithm
        
        distance1= math.sqrt((math.ceil(x)-x )**2 +   (math.ceil(y)-y)**2)
        distance2= math.sqrt((math.floor(x)-x)**2 +   (math.ceil(y)-y)**2)
        distance3= math.sqrt((math.ceil(x)-x )**2 +   (math.floor(y)-y)**2)
        distance4= math.sqrt((math.floor(x)-x)**2 +   (math.floor(y) -y)**2)

        if distance1==0: return self.cost_matrix[math.ceil(x)]  [math.ceil(y)]  #avoid division by zero errors
        if distance2==0:return self.cost_matrix[math.floor(x)] [math.floor(y)]
        if distance3==0: return self.cost_matrix[math.ceil(x)]  [math.ceil(y)]  
        if distance4==0: return self.cost_matrix[math.floor(x)] [math.floor(y)] 

        effect1= self.cost_matrix[math.ceil(x)]  [math.ceil(y)]   /  distance1
        effect2=  self.cost_matrix[math.floor(x)] [math.floor(y)] /  distance2
        effect3= self.cost_matrix[math.ceil(x)]  [math.ceil(y)]   /  distance3
        effect4= self.cost_matrix[math.floor(x)] [math.floor(y)]  /  distance4

        return (effect1+effect2+effect3+effect4)/(1/distance1+1/distance2+1/distance3+1/distance4)
