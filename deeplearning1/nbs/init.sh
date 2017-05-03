#!/usr/bin/env bash

#
# README:
#
# This script creates openssl certificates which are necessary for
# publishing Jupyter notebooks through openssl when using EC2 instances.
#
# And it also configures Jupyter to use the created cerficates.
#
# NOTE: Your Jupyter notebook will be listening port number 8888.
#       This will also be configured by this script.
#

# Code adapted from https://gist.githubusercontent.com/rashmibanthia/5a1e4d7e313d6832f2ff/raw/1f32274758851a32444491500fef4852496596ce/jupyter_notebook_ec2.sh

cd ~

jupyter notebook --generate-config

key=$(python -c "from notebook.auth import passwd; print(passwd())")

cd ~
mkdir certs
cd certs
certdir=$(pwd)
openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.key -out mycert.pem

cd ~
sed -i "1 a\
c = get_config()\\
c.NotebookApp.certfile = u'$certdir/mycert.pem'\\
c.NotebookApp.ip = '*'\\
c.NotebookApp.open_browser = False\\
c.NotebookApp.password = u'$key'\\
c.NotebookApp.port = 8888" .jupyter/jupyter_notebook_config.py
