import paramiko
import os
import predict
import numpy as np
import time
import deploy.python.infer as infer

output_file = 'paramiko.org' #text file to store output from commands

local_dir = r"pi_files"

ip_add = 'kameipi.local' #adjust based on your Raspberry Pi's name

x = [5.20623612, -0.71608794, -0.90561461] # parameters from calibration curve

#additional parameters from calibration curve
logCoef = -1.97824053

logInt = 4.1242898

def exponen(theta, x): #exponential
    return theta[0]*np.exp(theta[1]*x)+theta[2]

def log(theta, x): #logarithmic
    return theta[0]*np.log(x + theta[1]) + theta[2]

def inv(theta, x): #inverse
    return (theta[0]/(x+theta[1])) + theta[2]

def poly(theta, x, deg): #polynomial
    out = 0
    for i in range(deg):
        out += theta[i]*(x**(i))
    return out

def lin(theta,x): #linear
    return theta[0]*x + theta[1]

def logistic(x, coef, intercept):
    inp = x * coef + intercept
    return 1.0 / (1 + np.exp(-inp))
 
 
def getImages(hostname, command):
    print('running')
    try:
        local_dir = r"pi_files" #local directory on computer to contain images
        port = '22'
         
        # created client using paramiko
        client = paramiko.SSHClient()
         
        # here we are loading the system
        # host keys
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
         
        # connecting paramiko using host
        # name and password
        client.connect(hostname, port=22, username='',
                       password='')
         
        # below line command will actually
        # execute in your remote machine
        sleeptime = 0.001
        outdata, errdata = '', ''
        ssh_transp = client.get_transport()
        chan = ssh_transp.open_session()
        chan.setblocking(0)
        chan.exec_command(command)
        while True:  # monitoring process
            # Reading from output streams
            while chan.recv_ready():
                outdata += str(chan.recv(1000).decode('utf-8').strip())
            while chan.recv_stderr_ready():
                errdata += str(chan.recv_stderr(1000))
            if chan.exit_status_ready():  # If completed
                break
            time.sleep(sleeptime)
        retcode = chan.recv_exit_status()
         
        cmd_output = outdata
        print(cmd_output)

        ftp_client=client.open_sftp()
        remote_dir = cmd_output #get directory 

        dir = os.path.basename(os.path.split(remote_dir)[0])
        local_dir = os.path.join(local_dir, dir)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        
        # save files to local directory
        for filename in ftp_client.listdir(remote_dir):
            ftp_client.get(remote_dir+filename, os.path.join(local_dir, filename))
        ftp_client.close()
         
        # we are creating file which will read our
        # cmd_output and write it in output_file
        with open(output_file, "w+") as file:
            file.write(str(cmd_output))
             
        # we are returning the output
        return local_dir
    finally:
        client.close()
 

def outputAudio(hostname, command):
    print('running')
    try:
        local_dir = r"pi_files"
        port = '22'
         
        # created client using paramiko
        client = paramiko.SSHClient()
         
        # here we are loading the system
        # host keys
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
         
        # connecting paramiko using host
        # name and password
        client.connect(hostname, port=22, username='',
                       password='')
         
        # below line command will actually
        # execute in your remote machine
        sleeptime = 0.001
        outdata, errdata = '', ''
        ssh_transp = client.get_transport()
        chan = ssh_transp.open_session()
        chan.setblocking(0)
        chan.exec_command(command)
        while True:  # monitoring process
            # Reading from output streams
            while chan.recv_ready():
                outdata += str(chan.recv(1000).decode('utf-8').strip())
            while chan.recv_stderr_ready():
                errdata += str(chan.recv_stderr(1000))
            if chan.exit_status_ready():  # If completed
                break
            time.sleep(sleeptime)
        retcode = chan.recv_exit_status()

        cmd_output = outdata

        with open(output_file, "w+") as file:
            file.write(str(cmd_output))

        return
    finally:
        client.close()


out = getImages(ip_add, 'python /home/kameiraspi/Desktop/my_py_files/scanOutput.py')

argString = "--config output/inference_model/deploy.yaml --image_path "+out+" --save_dir output/result/infer"

# run neural network + calculated test line intensity
args = infer.parse_args(argString.split())
inten, conc = infer.main(args) #inten contains two columns: first contains test line intensity, and the second contains the area of the cropped detection zone

# sort intensities based on area of cropped detection zone
inten[inten[:,1].argsort()]

# intensity of second largest detection zone
intensity = inten[len(inten)-2,0]

logResult = logistic(intensity, logCoef, logInt)

biomarker = exponen(x,intensity) # change function based on calibration curve

if logResult < 0.5:
    result = 'False'
    biomarker = 0
else:
    result = 'True'

argString = "--positive " + result + " --conc " + str(np.round(biomarker,3)) + " --units nanograms/milliliter"

argString = 'python /home/kameiraspi/Desktop/my_py_files/output.py ' + argString

outputAudio(ip_add, argString)
