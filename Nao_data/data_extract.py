# -*- encoding: UTF-8 -*- 
#!/usr/bin/env python
from naoqi import ALProxy
import vision_definitions
import Image
import operator
import numpy as np
import cv2
import csv
import time

NAO_IP = "192.168.11.68"
PORT = 9559

alconnman = ALProxy("ALConnectionManager", NAO_IP, PORT)

# Proxy for Speech
tts = ALProxy("ALTextToSpeech", "nao.local", PORT)

# Proxy for VideoDevice
camProxy = ALProxy("ALVideoDevice", NAO_IP, PORT)
# Register a Generic Video Module
resolution = vision_definitions.kQVGA
colorSpace = vision_definitions.kBGRColorSpace
fps = 30
# Bottom camera
camera_code = 1
camProxy.setActiveCamera("ALVideoDevice", camera_code)
nameId = camProxy.subscribeCamera("python_GVM", camera_code, resolution, colorSpace, fps)
print nameId
resolution = vision_definitions.kQQVGA
camProxy.setResolution(nameId, resolution)

# Proxy for Motion
motionProxy = ALProxy("ALMotion", NAO_IP, PORT)

def get_contourcenter(cnts_list):
    c=0
    if len(cnts_list)>1:
        for cnts in cnts_list:
            #cnts=cnts_list
            if c<cv2.contourArea(cnts):#max(cnts,key=cv2.contourArea)
                ((x,y),radius)=cv2.minEnclosingCircle(cnts)
                c= cv2.contourArea(cnts)
        return (x,y)
    else:
        return (0,0)

time0 = time.time()
csv_2darray=[]
for k in range(100):
    time1 = time.time()
    print k
    #print 'getting images in remote'
    naoImage=camProxy.getImageRemote(nameId)
    
    #print 'end of gvm_getImageLocal python script'
    
    # Get the image size and pixel array.
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    
    # Create a PIL Image from our pixel array.
    im = Image.frombytes("RGB", (imageWidth, imageHeight), array)
    #im_edge = cv2.Canny(im)
    # Save the image.
    img_path = "./img_data/camImage" + str(k) + ".png"
    #img_path_edge = "./img_data_edge/camEdge" + str(k) + ".png"
    im.save(img_path, "PNG")
    #im_edge.save(img_path_edge, "PNG")    
    #im.show()
    
    image= np.asarray(im)
    
    #################################################################### Get joint Angles
    
    # Example that finds the difference between the command and sensed angles.
    #names        = "Body"
    names         = ['RShoulderRoll', 'RShoulderPitch', 'RElbowYaw', 'RElbowRoll']
    useSensors    = False
    commandAngles = motionProxy.getAngles(names, useSensors)
    
    useSensors  = True
    sensorAngles = motionProxy.getAngles(names, useSensors)
    
    errors = []
    for i in range(0, len(commandAngles)):
        errors.append(commandAngles[i]-sensorAngles[i])
    #print "Errors"
    #print errors
    
    ###################################### normalize joints
    min_vals = [-1.3265,-2.0857,-2.0857,0.0349]
    size_vals = [0.3142+1.3265,2.0857+2.0857,2.0857+2.0857,1.5446-0.0349]
    sub_vals = map(operator.sub, sensorAngles, min_vals)
    normal_joints = map(operator.truediv, sub_vals, size_vals)
    
    csv_list = normal_joints
    csv_2darray.append(csv_list)
    
    time2 = time.time()
    print "time diff = " + str(time2-time1)

################################################################### Save CSV
time3 = time.time()
print "total time diff = " + str(time3-time0)
camProxy.unsubscribe(nameId)

f = open('nao_data_joint.csv', 'w')

writer = csv.writer(f, lineterminator='\n')
writer.writerows(csv_2darray)

f.close()
