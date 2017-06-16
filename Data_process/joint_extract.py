# -*- encoding: UTF-8 -*- 
#!/usr/bin/env python
from naoqi       import ALProxy
import vision_definitions
import Image
import operator
import numpy as np
import cv2
import csv

NAO_IP = "192.168.11.81"
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


csv_2darray=[]
for k in range(40):
	print k
#################################################################### Vision
#	print 'getting images in remote'
#	for i in range(0, 1):
#	  naoImage=camProxy.getImageRemote(nameId)
#
#	print 'end of gvm_getImageLocal python script'
#
#	# Get the image size and pixel array.
#	imageWidth = naoImage[0]
#	imageHeight = naoImage[1]
#	array = naoImage[6]
#
#	# Create a PIL Image from our pixel array.
#	im = Image.frombytes("RGB", (imageWidth, imageHeight), array)
#
#	# Save the image.
#	im.save("camImage.png", "PNG")
#
#	im.show()


	#################################################################### Get joint Angles

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
	print "Errors"
	print errors

	#################################################################### Preprocessing Joint data

	min_values=[-0.3142,-2.0857,-2.0857,-1.5446,-1.3265,-2.0857,-2.0857,0.0349]
	size_values=[0.3142+1.3265,2.0857+2.0857,2.0857+2.0857,1.5446-0.0349,0.3142+1.3265,2.0857+2.0857,2.0857+2.0857,1.5446-0.0349]
	substracted_values=map(operator.sub, sensorAngles, min_values)
	normal_joints=map(operator.truediv, substracted_values, size_values)
	print names
	print normal_joints

	#################################################################### Preprocessing Vision data
#
#	image= np.asarray(im)
#	#Tracking Point
#	greenLower=(35,86,6)
#	greenUpper=(64,255,255)
#	redLower=(1,86,6)
#	redUpper=(18,255,255)
#	blueLower=(110,86,6)
#	blueUpper=(150,255,255)
#	yellowLower=(24,86,6)
#	yellowUpper=(28,255,255)
#
#	#frame= imutils.resize(image,width=600)
#	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#	#Green Mask
#	mask1 = cv2.inRange(hsv, greenLower, greenUpper)
#	mask1 = cv2.erode(mask1, None, iterations=2)
#	mask1 = cv2.dilate(mask1, None, iterations=2)
#	#Blue Mask
#	mask2 = cv2.inRange(hsv, redLower, redUpper)
#	mask2 = cv2.erode(mask2, None, iterations=2)
#	mask2 = cv2.dilate(mask2, None, iterations=2)
#	#Red Mask
#	mask3 = cv2.inRange(hsv, blueLower, blueUpper)
#	mask3 = cv2.erode(mask3, None, iterations=2)
#	mask3 = cv2.dilate(mask3, None, iterations=2)
#	#Yellow Mask
#	mask4 = cv2.inRange(hsv, yellowLower, yellowUpper)
#	mask4 = cv2.erode(mask4, None, iterations=2)
#	mask4 = cv2.dilate(mask4, None, iterations=2)
		
#	#Calculating Location
#	l1=(150,171)
#	l2=(5,215)
#	l3=(553,297)
#	l4=(549,366)
		
#	cnts_list1=cv2.findContours(mask1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
#	cnts_list2=cv2.findContours(mask2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
#	cnts_list3=cv2.findContours(mask3.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
#	cnts_list4=cv2.findContours(mask4.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

#	def get_contourcenter(cnts_list):
#		c=0
#		if len(cnts_list)>1:
#			for cnts in cnts_list:
#				#cnts=cnts_list
#				if c<cv2.contourArea(cnts):#max(cnts,key=cv2.contourArea)
#					((x,y),radius)=cv2.minEnclosingCircle(cnts)
#					c= cv2.contourArea(cnts)
#			return (x,y)
#		else:
#			return (0,0)

#	(x,y)=get_contourcenter(cnts_list1) #,"green")
#	(x,y)=get_contourcenter(cnts_list2) #,"red")
#	(x,y)=get_contourcenter(cnts_list3) #,"blue")
#	(x,y)=get_contourcenter(cnts_list4) #,"yellow")
#
#	normal_vision=[x/160,y/120]

	################################################################### CSV Formatted data

	#csv_list=normal_joints+normal_vision
	csv_list=normal_joints
	csv_2darray.append(csv_list)

################################################################### Save CSV

camProxy.unsubscribe(nameId)

f = open('nao_data.csv', 'w')

writer = csv.writer(f, lineterminator='\n')
writer.writerows(csv_2darray)

f.close()
