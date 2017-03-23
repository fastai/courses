#!/bin/bash

#
# README:
#
# This file will launch Jupyter notebook server with openssl certificates
# which might have been created through helper script init.sh.
#
# Your Jupyter notebook server will be accessible at:
#  https://YourEC2IP:8888
#

jupyter notebook --certfile=~/certs/mycert.pem --keyfile=~/certs/mycert.key
