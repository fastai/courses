# ------------------------------------------------------------------------- 
# 	AWS Shortcuts for the 'Practical Deep Learning' Server" 
#
#   remember to laod it by doing this in your console
#	$ source aws-alias.sh 
#
#   it will output all available commands:
#
#	awsai-ip       > returns the ip of the server
#	awsai-start    > start the server
#	awsai-stop     > stop the server so you don't get billed
#	awsai-ssh      > ssh into the server  
#	awsai-nb       > open the server's jupyter notebook (nb) in the browser
#	awsai-status   > shows if the server is running or stopped
#	awsai-help     > display this list of available commands
#
# -------------------------------------------------------------------------
                   

# retrieve the instance of your machines by searching for the name "fast-ai-gpu-machine" that is assigned by the setup script       
export instanceId="$(aws ec2 describe-instances --filters='Name=tag:Name,Values=fast-ai-gpu-machine' --output text --query 'Reservations[*].Instances[*].InstanceId')"
  
# IP of the instance
alias awsai-ip='export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query Reservations[0].Instances[0].PublicIpAddress` && echo $instanceIp' 

# start the server - it will ask again for a new ip but it's not needed, it will be the same as the ip doesn't change when the instance is stopped
# TODO see if you really need to reassign the instanceIp variable
alias awsai-start='aws ec2 start-instances --instance-ids $instanceId && aws ec2 wait instance-running --instance-ids $instanceId && export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress"` && echo $instanceIp'
         
# stop the server
alias awsai-stop='aws ec2 stop-instances --instance-ids $instanceId' 

# ssh into the server
alias awsai-ssh='ssh -i ~/.ssh/aws-key-fast-ai.pem ubuntu@$instanceIp'  
                                                                     
# check the status (running, stopping, stopped) of the server
# WARNING: describe-instance-status doesn't work (it returns empty if it's not running) use describe-instances with the query
#alias awsai-status='export instanceStatus=`aws ec2 describe-instance-status --instance-ids $instanceId --filters "instance-state-name"` && echo $instanceStatus'
# this command works in the console but has some problems in the 
# aws ec2 describe-instances --instance-ids i-083038ae21434741d --query Reservations[].Instances[].State.Name --output text
#alias awsai-status='export instanceStatus=`aws ec2 describe-instances --instance-ids $instanceId --query Reservations[0].Instances[0].State.Name && echo $instanceStatus`'
#make it into a function because the variable keeps adding a string every time you invoke the function
function awsai-status() {
	echo "`aws ec2 describe-instances --instance-ids $instanceId --query Reservations[0].Instances[0].State.Name`"  
} 
           
                   
      
# Open The jupyter Notebook 
function awsai-nb (){
	# TODO improve it by checking if the url actually exists  follow this http://stackoverflow.com/questions/2924422/how-do-i-determine-if-a-web-page-exists-with-shell-scripting
	echo " "
	echo "Opening jupyter notebook"
	echo "Remember: if the page doesn't load and returns a 404 it can mean either one of 2 things:"
	echo "1- you haven't started the server - check it's status with with awsai-status > current status: $(awsai-status)"
	echo "2- the server is started by you haven't launched jupyter on it"
	echo "to do that first ssh into the server"
	echo "   $ awsai-ssh"
	echo "then start the jupyter server"
	echo "   $ jupyter notebook"
	echo " "
	
  	if [[ `uname` == *"CYGWIN"* ]]
	then
	    # This is cygwin.  Use cygstart to open the notebook
	    cygstart http://$instanceIp:8888
	fi

	if [[ `uname` == *"Linux"* ]]
	then
	    # This is linux.  Use xdg-open to open the notebook
	    xdg-open http://$instanceIp:8888
	fi

	if [[ `uname` == *"Darwin"* ]]
	then
	    # This is Mac.  Use open to open the notebook
	    open http://$instanceIp:8888
	fi  
}



# echo the available commands and aliases when you load it 
function awsai-help() {
	echo " "
	echo "-------------------------------------------------------------------------" 
	echo "AWS Shortcuts for the 'Practical Deep Learning' Server"
	echo "-------------------------------------------------------------------------" 
	echo "instanceId:    $instanceId"
#	echo "instanceIp:    $instanceIp"
	echo "instanceIp:    $(awsai-ip)"		 #call the alias because nothing is generating the variable untill you launch a command
    echo "status:        $(awsai_getStatus)" #call the function because it can be different every time
	echo "-------------------------------------------------------------------------" 
	echo "awsai-ip       > returns the ip of the server: $instanceIp"
	echo "awsai-start    > start the server"
	echo "awsai-stop     > stop the server so you don't get billed"
	echo "awsai-ssh      > ssh into the server"  
	echo "awsai-nb       > open the server's jupyter notebook (nb) in the browser"
	echo "awsai-status   > shows if the server is running or stopped"
	echo "awsai-help     > display this list of available commands"
	echo "-------------------------------------------------------------------------" 
} 


# done loading display the commands
awsai-help    
