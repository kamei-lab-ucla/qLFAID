from time import sleep 
import RPi.GPIO as GPIO
from datetime import datetime
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import argparse
import sys

#GPIO Pins
vib = 1
startBut = 16
repBut = 20
audio = 7

#Sound Files
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

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--positive",
        dest='pos',
        help="Whether test result is positive or negative.",
        default=False,
        type=eval,
        choices=[True, False],
        required=True)
    parser.add_argument(
        '--conc',
        dest='conc',
        help='Quantitative Result',
        type=float,
        default=0)
    parser.add_argument(
        '--units',
        dest='units',
        help='Concentration Units',
        type=str,
        default=None)

    return parser.parse_args()


def main(args):
	#Initialize TTS functionality
	pygame.mixer.init()
	
	#GPIO Setup
	GPIO.setmode(GPIO.BCM)
	GPIO.setup(vib, GPIO.OUT)
	GPIO.setup(startBut, GPIO.IN)
	GPIO.setup(repBut, GPIO.IN)
	GPIO.setup(audio, GPIO.OUT)
	
	User_output(args.pos, args.conc, Text_to_speech, startBut, repBut, audio, vib, baseDir, ext)
	GPIO.cleanup()
	return 'Done'


if __name__ == '__main__':
    args = parse_args()
    print(main(args))
