# -*- encoding: UTF-8 -*- 
#!/usr/bin/env python
from naoqi       import ALProxy
import vision_definitions
import Image
import operator
import numpy as np
import cv2
import csv
import time

NAO_IP = "192.168.11.47"
PORT = 9559

alconnman = ALProxy("ALConnectionManager", NAO_IP, PORT)

# Proxy for Speech
tts = ALProxy("ALTextToSpeech", "nao.local", PORT)

# Proxy for VideoDevice
camProxy = ALProxy("ALVideoDevice", NAO_IP, PORT)
# Register a Generic Video Module
resolution = vision_definitions.kQVGA
colorSpace = vision_definitions.kYUVColorSpace
fps = 30
nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
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


#################################################################### Scanning is required to update the services list

#alconnman.scan()
#services = alconnman.services()

#for service in services:
#    network = dict(service)
#    if network["Name"] == "":
#        print "{hidden} " + network["ServiceId"]
#    else:
#        print network["Name"] + " " + network["ServiceId"]

# Speaking
#tts = ALProxy("ALTextToSpeech", "nao.local", PORT)
#tts.say("hello")

time0 = time.time()
csv_2darray=[]
for k in range(1200):
    time1 = time.time()
    print k
    
    # Example that finds the difference between the command and sensed angles.
    #names         = "Body"
    names         = ['LShoulderRoll', 'LShoulderPitch', 'LElbowYaw', 'LElbowRoll','RShoulderRoll', 'RShoulderPitch', 'RElbowYaw', 'RElbowRoll']
    useSensors    = False
    commandAngles = motionProxy.getAngles(names, useSensors)

    useSensors  = True
    sensorAngles = motionProxy.getAngles(names, useSensors)

    errors = []
    for i in range(0, len(commandAngles)):
      errors.append(commandAngles[i]-sensorAngles[i])

    #################################################################### Preprocessing Joint data

    min_values=[-0.3142,-2.0857,-2.0857,-1.5446,-1.3265,-2.0857,-2.0857,0.0349]
    size_values=[0.3142+1.3265,2.0857+2.0857,2.0857+2.0857,1.5446-0.0349,0.3142+1.3265,2.0857+2.0857,2.0857+2.0857,1.5446-0.0349]
    substracted_values=map(operator.sub, sensorAngles, min_values)
    normal_joints=map(operator.truediv, substracted_values, size_values)
#    print names
    print normal_joints

    ################################################################### CSV Formatted data
    
    #csv_list=normal_joints+normal_vision
    csv_list=normal_joints
    csv_2darray.append(csv_list)
    
    time2 = time.time()
    while time2 - time1 < 0.00999:
        time2 = time.time()
    
    time2 = time.time()
    print "time diff = " + str(time2-time1)

################################################################### Save CSV
time3 = time.time()
print "total time diff = " + str(time3-time0)

camProxy.unsubscribe(nameId)

f = open('nao_data_left.csv', 'w')

writer = csv.writer(f, lineterminator='\n')
writer.writerows(csv_2darray)

f.close()
