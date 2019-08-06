'''
 dictionary structure 'id of the object': count down timer of the object as per frames
'''

from id_adder import CentroidTracer
import numpy as np
from collections import Counter
from sort import *
import tkinter
from tkinter import messagebox
# object creation for CentroidTracer
ct = CentroidTracer()
mot_tracker = Sort()
count = 0
id_record = dict()
count_time = 12

root = tkinter.Tk()
root.withdraw()

def tracePlace(image_np,detection_rect,detection_num):
    global count,count_time
    global id_record
    count = count+1
    #print(count,":difference between them:")
    print(mot_tracker.update(detection_rect))
    id_detection_value = mot_tracker.update(detection_rect)
    row_count = np.size(id_detection_value,0)
    print(row_count)

    if count > 3:
        for itr in range(0,row_count):
            if id_detection_value[itr,4] in id_record:
                id_record[id_detection_value[itr,4]] = id_record.pop(id_detection_value[itr,4])-1
                if id_record.get(id_detection_value[itr,4]) < 0:
                    id_record[id_detection_value[itr,4]] = 'alert'
                    messagebox.showwarning('Warning','suspicious object in sight')
                print(id_record)
            else:
                id_record[id_detection_value[itr,4]] = count_time
                print('object added')
                print(id_record)
    return image_np
