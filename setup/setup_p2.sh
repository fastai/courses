#!/bin/bash
#
# Configure a p2.xlarge instance

# get the correct ami
export region=$(aws configure get region)
if [ $region = "us-west-2" ]; then
   export ami="ami-8c4288f4" # Oregon
elif [ $region = "eu-west-1" ]; then
   export ami="ami-b93c9ec0" # Ireland
elif [ $region = "us-east-1" ]; then
  export ami="ami-c6ac1cbc" # Virginia
elif [ $region = "ap-southeast-2" ]; then
  export ami="ami-b93c9ec0" # Sydney
elif [ $region = "ap-south-1" ]; then
  export ami="ami-c53975aa" # Mumbai
else
  echo "Only us-west-2 (Oregon), eu-west-1 (Ireland), us-east-1 (Virginia), ap-southeast-2 (Sydney), and ap-south-1 (Mumbai) are currently supported"
  exit 1
fi

export instanceType="p2.xlarge"

. $(dirname "$0")/setup_instance.sh
