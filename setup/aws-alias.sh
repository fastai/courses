alias aws-get-p2='export instanceId=`aws ec2 describe-instances --filters "Name=instance-state-name,Values=stopped,Name=instance-type,Values=p2.xlarge" --query "Reservations[0].Instances[0].InstanceId"` && echo $instanceId'
alias aws-get-t2='export instanceId=`aws ec2 describe-instances --filters "Name=instance-state-name,Values=stopped,Name=instance-type,Values=t2.xlarge" --query "Reservations[0].Instances[0].InstanceId"` && echo $instanceId'
alias aws-start='aws ec2 start-instances --instance-ids $instanceId && aws ec2 wait instance-running --instance-ids $instanceId && export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress"` && echo $instanceIp'
alias aws-ip='export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress"` && echo $instanceIp'
alias aws-ssh='ssh -i ~/.ssh/aws-key-fast-ai.pem ubuntu@$instanceIp'
alias aws-stop='aws ec2 stop-instances --instance-ids $instanceId'
alias aws-state='aws ec2 describe-instances --instance-ids $instanceId --query "Reservations[0].Instances[0].State.Name"'


if [[ `uname` == *"CYGWIN"* ]]
then
    # This is cygwin.  Use cygstart to open the notebook
    alias aws-nb='cygstart http://$instanceIp:8888'
fi

if [[ `uname` == *"Linux"* ]]
then
    # This is linux.  Use xdg-open to open the notebook
    alias aws-nb='xdg-open http://$instanceIp:8888'
fi

if [[ `uname` == *"Darwin"* ]]
then
    # This is Mac.  Use open to open the notebook
    alias aws-nb='open http://$instanceIp:8888'
fi
