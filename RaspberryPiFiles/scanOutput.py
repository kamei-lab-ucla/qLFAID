from time import sleep 
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib as Motor
from picamera import PiCamera, PiResolution
from datetime import datetime
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

#Initialize TTS functionality
pygame.mixer.init()
baseDir = '/home/kameiraspi/Desktop/my_py_files/sound_files/'
ext = '.mp3'

def Text_to_speech(baseDir, audioFile, ext, audioPin):
    GPIO.output(audioPin, True)
    pygame.mixer.music.load(baseDir+audioFile+ext)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    GPIO.output(audioPin, False)

def User_output(result, concentration, Text_to_speech, startBut, repBut, audioPin, vibPin, baseDir, ext):
	def concentrationTTS(concentration):
		concentrationText = str(concentration)
		for i in range(len(concentrationText)):
			number = concentrationText[i]
			if number == '.':
				number = 'point'
			Text_to_speech(baseDir, number, ext, audioPin)
		Text_to_speech(baseDir, 'units', ext, audioPin)

	if result == True:
		resultText = 'positive'
		buzz = 3
	elif result == False:
		resultText = 'negative'
		buzz = 1
	else:
		resultText = 'invalid'
		buzz = 5
	
	if result == True:
		scriptConcentration = 'theBiomarkerConcentration'
	elif result == False:
		scriptConcentration = 'thereWasNoBiomarker'
	else:
		scriptConcentration = 'yourLFAMayBeInvalid'

	Text_to_speech(baseDir, 'ready', ext, audioPin)	
	GPIO.output(vibPin, True)
	sleep(3)
	GPIO.output(vibPin, False)

	start = 0
	while start == 0:
		start = GPIO.input(startBut)

	Text_to_speech(baseDir, 'yourTestIs', ext, audioPin)
	Text_to_speech(baseDir, resultText, ext, audioPin)

	for x in range(buzz):
		GPIO.output(vibPin, True)
		sleep(0.5)
		GPIO.output(vibPin, False)
		sleep(0.5)	
	Text_to_speech(baseDir, scriptConcentration, ext, audioPin)
	if result == True:
		concentrationTTS(concentration)
	Text_to_speech(baseDir, 'repeat', ext, audioPin)

	start = 0
	while start == 0:
		rep = GPIO.input(repBut)
		start = GPIO.input(startBut)
		if rep == 1:
			Text_to_speech(baseDir, 'yourTestIs', ext, audioPin)
			Text_to_speech(baseDir, resultText, ext, audioPin)
			for x in range(buzz):
				GPIO.output(vibPin, True)
				sleep(0.5)
				GPIO.output(vibPin, False)
				sleep(0.5)      
			Text_to_speech(baseDir, scriptConcentration, ext, audioPin)
			if result == True:
				concentrationTTS(concentration)

#Setting up folder name for scan images
now = datetime.now()
dateStr = now.strftime("%d-%m-%y-%H.%M.%S")
dirName = '/home/kameiraspi/Desktop/my_media/' + dateStr + '/'

if not (os.path.isdir(dirName)):
	os.mkdir(dirName)

#Camera setup (video recorded for prototyping purposes)
camera = PiCamera()
camera.resolution = (2048, 1536)
camera.rotation = 180
camera.start_preview(fullscreen = False, window = (100, 20, 640, 480))

"""
Picamera2 Code
camera = Picamera2()
camera_config = camera.create_preview_configuration(main={"size": (2048, 1536)}, lores={"size": (320, 240)}, encode="lores", transform=libcamera.Transform(hflip=1, vflip=1))
camera.configure(camera_config)
camera.start_preview(Preview.QTGL)

encoder = H264Encoder(bitrate=10000000)
output = "/home/kameiraspi/Desktop/my_media/scanTest1/video.h264"
"""

#GPIO Pins
GpioPins = [6, 13, 19, 26] #For the stepper motor driver
lever = [5, 12]
vib = 1
startBut = 16
repBut = 20
devMode = 21
led = 0
audio = 7
motor = 11

#Will automatically set pinMode to BCM
mymotor = Motor.BYJMotor("MyMotorOne", "28BYJ")

#GPIO Setup
for i in range(0, len(lever)):
	GPIO.setup(lever[i], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(vib, GPIO.OUT)
GPIO.setup(startBut, GPIO.IN)
GPIO.setup(repBut, GPIO.IN)
GPIO.setup(devMode, GPIO.IN)
GPIO.setup(led, GPIO.OUT)
GPIO.setup(audio, GPIO.OUT)
GPIO.setup(motor, GPIO.OUT)

#Stepper motor parameters
stepdelay = 0.003
steps = 200
small_steps = 50
steptype = "wave"
verbose = False
initdelay = 0
counterclockwise = True
clockwise = False

#Define variables to store rolling levers' states
backLever = 1
frontLever = 1

#Move camera to the back of the device
#camera.start_recording(encoder, output)
#camera.start()
GPIO.output(motor, True)
while backLever == 1: 
	backLever = GPIO.input(lever[1])
	if backLever == 1:
		mymotor.motor_run(GpioPins, stepdelay, small_steps, counterclockwise, verbose, steptype, initdelay)
GPIO.output(motor, False)
step = 1

#Ask the user to put LFA in and to press start when ready
start = 0
Text_to_speech(baseDir, 'insert',  ext, audio)

while start == 0:
	start = GPIO.input(startBut) 

#Turn on LED and adjust camera settings to turn off auto-adjust
GPIO.output(led, True)
Text_to_speech(baseDir, 'analysis',  ext, audio)
sleep(1)
camera.exposure_mode = 'off'
camera.awb_mode = 'incandescent'

GPIO.output(motor, True)
#Scan from back to front of device
while frontLever == 1:
	#print('Step ' + str(step) + ', Run detection algorithm')
	frontLever = GPIO.input(lever[0])
	camera.capture(dirName + 'step' + str(step)  + '.jpg')
	if frontLever == 1:
		mymotor.motor_run(GpioPins, stepdelay, steps, clockwise, verbose, steptype, initdelay)
		frontLever = GPIO.input(lever[0])
		if frontLever == 1:
			mymotor.motor_run(GpioPins, stepdelay, steps, clockwise, verbose, steptype, initdelay)		
	step = step + 1
GPIO.output(motor, False)
GPIO.output(led, False)

print(dirName)
'''
#Dummy output
result = GPIO.input(devMode)
concentration = 2.982

User_output(result, concentration, Text_to_speech, startBut, repBut, audio, vib, baseDir, ext)
'''

#camera.stop_recording()
camera.stop_preview()
GPIO.cleanup()
