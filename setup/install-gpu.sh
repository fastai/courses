# This script is designed to work with ubuntu 16.04 LTS

fCudaPackage="cuda-repo-ubuntu1604_8.0.44-1_amd64.deb"
fAnaconda2Sh="Anaconda2-4.2.0-Linux-x86_64.sh"
dAnaconda2Install="$HOME/anaconda2/"
fcuDNN="cudnn.tgz"
dcuDNN="cuda"
hKeras="$HOME/.keras/"

# Ensure system is updated and has basic build tools
sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common

# Download and install GPU drivers
read -r -p " Do you want to install CUDA?[Y/n]" cuResponse
if [[ "$cuResponse" =~ ^([yY][eE][sS]|[yY])+$ ]];
then
    if [ ! -f ${fCudaPackage} ];
    then
        wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb" -O ${fCudaPackage}
        sudo dpkg -i ${fCudaPackage}
        sudo apt-get update
        sudo apt-get -y install ${dcuDNN}
        sudo modprobe nvidia
    fi
fi

#Removing the below line, After installation of cuda without reboot nvidia-smi throw an error
#nvidia-smi

# Install Anaconda for current user
mkdir downloads
cd downloads
if [ ! -f ${fAnaconda2Sh} ];
then
    wget "https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh" -O ${fAnaconda2Sh}
fi

if [ -d ${dAnaconda2Install} ];
then
    echo "Anaconda already installed"
    read -r -p " Do you wanna delete and install it again[Y/n]" condaResponse
    if [[ "$condaResponse" =~ ^([yY][eE][sS]|[yY])+$ ]]
    then
        rm -rf ${dAnaconda2Install}
        bash ${fAnaconda2Sh} -b
        conda install -y bcolz
        conda upgrade -y --all
    fi
else
    bash ${fAnaconda2Sh} -b
    echo "export PATH=\"$HOME/anaconda2/bin:\$PATH\"" >> ~/.bashrc
    #conda2 Env update
    export PATH="$HOME/anaconda2/bin:$PATH"

    #Proxy config if your behind the Corporate Proxy
    read -r -p " Are you behind the Corporate Proxy[Y/n]" proxyYes
    if [[ "$proxyYes" =~ ^([yY][eE][sS]|[yY])+$ ]];
    then
        read -r -p " Type Proxy IP address[host:port]" proxyIP
        touch ~/.condarc
        echo "http: http://${proxyIP}
https: https://${proxyIP}
ssl_verify: False" >> ~/.condarc

        #Configure proxy for pip
        echo "export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt python" >> ~/.bashrc
    fi
    conda install -y bcolz
    conda upgrade -y --all
fi

# install and configure theano
pip install theano --user
echo "[global]
device = gpu
floatX = float32

[cuda]
root = /usr/local/cuda" > ~/.theanorc

# install and configure keras
pip install keras==1.2.2 --user

if [ ! -d ${hKeras} ];
then
    mkdir ~/.keras
    echo '{
        "image_dim_ordering": "th",
        "epsilon": 1e-07,
        "floatx": "float32",
        "backend": "theano"
    }' > ~/.keras/keras.json
fi

# install cudnn libraries
if [ ! -f ${fcuDNN} ];
then
    if [ ! -d ${dcuDNN} ];
    then
        wget "http://files.fast.ai/files/cudnn.tgz" -O ${fcuDNN}
        tar -zxf ${fcuDNN}
        cd ${dcuDNN}
        sudo cp lib64/* /usr/local/cuda/lib64/
        sudo cp include/* /usr/local/cuda/include/
    fi
else
    echo "Skipping cuDNN Installation"
fi

# configure jupyter and prompt for password
jupyter notebook --generate-config
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
echo "c.NotebookApp.password = u'"$jupass"'" >> $HOME/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py

# clone the fast.ai course repo and prompt to start notebook
cd ~
git clone https://github.com/fastai/courses.git
echo "\"jupyter notebook\" will start Jupyter on port 8888"
echo "If you get an error instead, try restarting your session so your $PATH is updated"
