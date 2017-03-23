#!/bin/bash
#
# This script should be invoked via setup_t2.sh or setup_p2.sh; those scripts
# will export the right environment variables for this to succeed.

# uncomment for debugging
# set -x

if [ -z "$ami" ] || [ -z "$instanceType" ]; then
    echo "Missing \$ami or \$instanceType; this script should be called from"
    echo "setup_t2.sh or setup_p2.sh!"
    exit 1
fi

# settings
export name="fast-ai"
export cidr="0.0.0.0/0"

hash aws 2>/dev/null
if [ $? -ne 0 ]; then
    echo >&2 "'aws' command line tool required, but not installed.  Aborting."
    exit 1
fi

if [ -z "$(aws configure get aws_access_key_id)" ]; then
    echo "AWS credentials not configured.  Aborting"
    exit 1
fi

export vpcId=$(aws ec2 create-vpc --cidr-block 10.0.0.0/28 --query 'Vpc.VpcId' --output text)
aws ec2 create-tags --resources $vpcId --tags --tags Key=Name,Value=$name
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-support "{\"Value\":true}"
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-hostnames "{\"Value\":true}"

export internetGatewayId=$(aws ec2 create-internet-gateway --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 create-tags --resources $internetGatewayId --tags --tags Key=Name,Value=$name-gateway
aws ec2 attach-internet-gateway --internet-gateway-id $internetGatewayId --vpc-id $vpcId

export subnetId=$(aws ec2 create-subnet --vpc-id $vpcId --cidr-block 10.0.0.0/28 --query 'Subnet.SubnetId' --output text)
aws ec2 create-tags --resources $subnetId --tags --tags Key=Name,Value=$name-subnet

export routeTableId=$(aws ec2 create-route-table --vpc-id $vpcId --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-tags --resources $routeTableId --tags --tags Key=Name,Value=$name-route-table
export routeTableAssoc=$(aws ec2 associate-route-table --route-table-id $routeTableId --subnet-id $subnetId --output text)
aws ec2 create-route --route-table-id $routeTableId --destination-cidr-block 0.0.0.0/0 --gateway-id $internetGatewayId

export securityGroupId=$(aws ec2 create-security-group --group-name $name-security-group --description "SG for fast.ai machine" --vpc-id $vpcId --query 'GroupId' --output text)
# ssh
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 22 --cidr $cidr
# jupyter notebook
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 8888-8898 --cidr $cidr

if [ ! -d ~/.ssh ]
then
	mkdir ~/.ssh
fi

if [ ! -f ~/.ssh/aws-key-$name.pem ]
then
	aws ec2 create-key-pair --key-name aws-key-$name --query 'KeyMaterial' --output text > ~/.ssh/aws-key-$name.pem
	chmod 400 ~/.ssh/aws-key-$name.pem
fi

export instanceId=$(aws ec2 run-instances --image-id $ami --count 1 --instance-type $instanceType --key-name aws-key-$name --security-group-ids $securityGroupId --subnet-id $subnetId --associate-public-ip-address --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 128, \"VolumeType\": \"gp2\" } } ]" --query 'Instances[0].InstanceId' --output text)
aws ec2 create-tags --resources $instanceId --tags --tags Key=Name,Value=$name-gpu-machine
export allocAddr=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)

echo Waiting for instance start...
aws ec2 wait instance-running --instance-ids $instanceId
sleep 10 # wait for ssh service to start running too
export assocId=$(aws ec2 associate-address --instance-id $instanceId --allocation-id $allocAddr --query 'AssociationId' --output text)
export instanceUrl=$(aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
#export ebsVolume=$(aws ec2 describe-instance-attribute --instance-id $instanceId --attribute  blockDeviceMapping  --query BlockDeviceMappings[0].Ebs.VolumeId --output text)

# reboot instance, because I was getting "Failed to initialize NVML: Driver/library version mismatch"
# error when running the nvidia-smi command
# see also http://forums.fast.ai/t/no-cuda-capable-device-is-detected/168/13
aws ec2 reboot-instances --instance-ids $instanceId

# save commands to file
echo \# Connect to your instance: > $name-commands.txt # overwrite existing file
echo ssh -i ~/.ssh/aws-key-$name.pem ubuntu@$instanceUrl >> $name-commands.txt
echo \# Stop your instance: : >> $name-commands.txt
echo aws ec2 stop-instances --instance-ids $instanceId  >> $name-commands.txt
echo \# Start your instance: >> $name-commands.txt
echo aws ec2 start-instances --instance-ids $instanceId  >> $name-commands.txt
echo \# Reboot your instance: >> $name-commands.txt
echo aws ec2 reboot-instances --instance-ids $instanceId  >> $name-commands.txt
echo ""
# export vars to be sure
echo export instanceId=$instanceId >> $name-commands.txt
echo export subnetId=$subnetId >> $name-commands.txt
echo export securityGroupId=$securityGroupId >> $name-commands.txt
echo export instanceUrl=$instanceUrl >> $name-commands.txt
echo export routeTableId=$routeTableId >> $name-commands.txt
echo export name=$name >> $name-commands.txt
echo export vpcId=$vpcId >> $name-commands.txt
echo export internetGatewayId=$internetGatewayId >> $name-commands.txt
echo export subnetId=$subnetId >> $name-commands.txt
echo export allocAddr=$allocAddr >> $name-commands.txt
echo export assocId=$assocId >> $name-commands.txt
echo export routeTableAssoc=$routeTableAssoc >> $name-commands.txt

# save delete commands for cleanup
echo "#!/bin/bash" > $name-remove.sh # overwrite existing file
echo aws ec2 disassociate-address --association-id $assocId >> $name-remove.sh
echo aws ec2 release-address --allocation-id $allocAddr >> $name-remove.sh

# volume gets deleted with the instance automatically
echo aws ec2 terminate-instances --instance-ids $instanceId >> $name-remove.sh
echo aws ec2 wait instance-terminated --instance-ids $instanceId >> $name-remove.sh
echo aws ec2 delete-security-group --group-id $securityGroupId >> $name-remove.sh

echo aws ec2 disassociate-route-table --association-id $routeTableAssoc >> $name-remove.sh
echo aws ec2 delete-route-table --route-table-id $routeTableId >> $name-remove.sh

echo aws ec2 detach-internet-gateway --internet-gateway-id $internetGatewayId --vpc-id $vpcId >> $name-remove.sh
echo aws ec2 delete-internet-gateway --internet-gateway-id $internetGatewayId >> $name-remove.sh
echo aws ec2 delete-subnet --subnet-id $subnetId >> $name-remove.sh

echo aws ec2 delete-vpc --vpc-id $vpcId >> $name-remove.sh
echo echo If you want to delete the key-pair, please do it manually. >> $name-remove.sh

chmod +x $name-remove.sh

echo All done. Find all you need to connect in the $name-commands.txt file and to remove the stack call $name-remove.sh
echo Connect to your instance: ssh -i ~/.ssh/aws-key-$name.pem ubuntu@$instanceUrl
