import paramiko
import os
import predict
import numpy as np
import time
import deploy.python.infer as infer

output_file = 'paramiko.org'

local_dir = r"pi_files"

# x = [3.02857431,  0.23344494, -0.69082306]

x = [5.20623612, -0.71608794, -0.90561461]

logCoef = -1.97824053

logInt = 4.1242898

def inv(theta, x):
    return (theta[0]/(x+theta[1])) + theta[2]

def exponen(theta, x):
    return theta[0]*np.exp(theta[1]*x)+theta[2]

def logistic(x, coef, intercept):
    inp = x * coef + intercept
    return 1.0 / (1 + np.exp(-inp))
 
 
def paramiko_GKG(hostname, command):
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
        client.connect(hostname, port=22, username='kameiraspi',
                       password='kameiBestCapstone')
         
        # below line command will actually
        # execute in your remote machine
        # (stdin, stdout, stderr) = client.exec_command(command)
        sleeptime = 0.001
        outdata, errdata = '', ''
        ssh_transp = client.get_transport()
        chan = ssh_transp.open_session()
        # chan.settimeout(3 * 60 * 60)
        chan.setblocking(0)
        chan.exec_command(command)
        while True:  # monitoring process
            # Reading from output streams
            # time.sleep(120)
            while chan.recv_ready():
                outdata += str(chan.recv(1000).decode('utf-8').strip())
            while chan.recv_stderr_ready():
                errdata += str(chan.recv_stderr(1000))
            if chan.exit_status_ready():  # If completed
                break
            time.sleep(sleeptime)
        retcode = chan.recv_exit_status()
         
        # redirecting all the output in cmd_output
        # variable
        # cmd_output = stdout.read().decode('utf-8').strip()
        # print('log printing: ', cmd_output)
        cmd_output = outdata
        print(cmd_output)



        ftp_client=client.open_sftp()
        remote_dir = cmd_output

        dir = os.path.basename(os.path.split(remote_dir)[0])
        local_dir = os.path.join(local_dir, dir)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        

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
 

def paramiko_Gen(hostname, command):
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
        client.connect(hostname, port=22, username='kameiraspi',
                       password='kameiBestCapstone')
         
        # below line command will actually
        # execute in your remote machine
        # (stdin, stdout, stderr) = client.exec_command(command)
        sleeptime = 0.001
        outdata, errdata = '', ''
        ssh_transp = client.get_transport()
        chan = ssh_transp.open_session()
        # chan.settimeout(3 * 60 * 60)
        chan.setblocking(0)
        chan.exec_command(command)
        while True:  # monitoring process
            # Reading from output streams
            # time.sleep(120)
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


# out = paramiko_GKG('164.67.191.144', 'python /home/kameiraspi/Desktop/my_py_files/scanOutput.py')
out = paramiko_GKG('kameipi.local', 'python /home/kameiraspi/Desktop/my_py_files/scanOutput.py')

# out = r"pi_files\Positions"

argString = "--config output/inference_model/deploy.yaml --image_path "+out+" --save_dir output/result/infer"

args = infer.parse_args(argString.split())
inten, names = infer.main(args)

# biomarker = np.zeros(len(inten))

# biomarker = exponen(x, inten)

# for i in range(len(inten)):
#     biomarker[i] = exponen(x, inten[i])

# print(names)
# print(inten)
# print(biomarker)

inten[inten[:,1].argsort()]

intensity = inten[len(inten)-2,0]

logResult = logistic(intensity, logCoef, logInt)

biomarker = exponen(x,intensity)

if logResult < 0.5:
    result = 'False'
    biomarker = 0
else:
    result = 'True'

argString = "--positive " + result + " --conc " + str(np.round(biomarker,3)) + " --units nanograms/milliliter"

argString = 'python /home/kameiraspi/Desktop/my_py_files/output.py ' + argString

# argString = 'python /home/kameiraspi/Desktop/my_py_files/output.py --positive True --conc 2.541 --units nanograms/milliliter'

# paramiko_Gen('164.67.191.144', argString)
paramiko_Gen('kameipi.local', argString)
