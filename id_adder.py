'''acceptinfg the boundery box cordinate and claculatinfg the centroid
    calculating the ecludian distance between and new and existing objects
    regester the new object
    deregister old object (time function to be implemented)
'''

from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict

class CentroidTracer():
    def __init__(self,maxDisappeared = 50):
        self.nextObjectId = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self,centroid):
        # adding a new object and setting initial disappeared to 0
        self.objects[self.nextObjectId] = centroid
        self.disappeared[self.nextObjectId] = 0
        self.nextObjectId += 1

    def deregister(self,objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self,rect):
        if len(rect) == 0:
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        #calculation of centroid of object detedted
        inputCentroids = np.zeros((len(rect),2),dtype=int)
        for (i, (startX,startY,endX,endY)) in enumerate(rect):
            cX = int((startX+endX)/2.0)
            cY = int((startY+endY)/2.0)
            inputCentroids[i] = (cX,cY)
        #if no object is being tracked to register new object
        if len(self.objects) == 0:
            for i in range(0,len(inputCentroids)):
                self.register(inputCentroids[i])
        #matching input centroid with existing centroids for
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows,usedCols = set(),set()
            for (row,col) in zip(rows,cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[cols]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

                unusedRows = set(range(0,D.shape[0])).difference(usedRows)
                unusedCols = set(range(D.shape[1])).difference(usedCols)

                if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        objectID = objectIDs[row]
                        self.disappeared[objectID] += 1

                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)
                else:
                    for col in unusedCols:
                        self.register(inputCentroids[col])
        return self.objects