## List of files
| File                  | Purpose       |
| --------------------- | ------------- |
| `aws-alias.sh`        | Command aliases that make AWS server management easier. |
| `instal-gpu-azure.sh` | Installs required software on an Azure Ubuntu server. Instructions available on the [wiki](http://wiki.fast.ai/index.php/Azure_install). |
| `install-gpu.sh`      | Installs required software on an Ubuntu machine. Instructions available on the [wiki](http://wiki.fast.ai/index.php/Ubuntu_installation). |
| `setup_instance.sh`   | Sets up an AWS environment for use in the course including a server that has the required software installed. This script is used by `setup_p2.sh` or `setup_t2.sh`. You probably don't need to call it by itself. |
| `setup_p2.sh` and `setup_t2.sh` | Configure environment variables for use in `setup_instance.sh`. These files call `setup_instance.sh`, which does the actual work of setting up the AWS instance. |

## Setup Instructions

### AWS
If you haven't already, view the video at http://course.fast.ai/lessons/aws.html for the steps you need to complete before running these scripts. More information is available on the [wiki](http://wiki.fast.ai/index.php/AWS_install).
1. Decide if you will use a GPU server or a standard server. GPU servers process deep learning jobs faster than general purpose servers, but they cost more per hour. Compare server pricing at https://aws.amazon.com/ec2/pricing/on-demand/.
2. Download `setup_p2.sh` if you decided on the GPU server or `setup_t2.sh` for the general purpose server. Also download `setup_instance.sh`.
3. Run the command `bash setup_p2.sh` or `bash setup_t2.sh`, depending on the file you downloaded. Run the command locally from the folder where the files were downloaded. Running `bash setup_p2.sh` sets up a p2.xlarge GPU server. `bash setup_p2.sh` sets up a t2.xlarge general purpose server.
4. The script will set up the server you selected along with other pieces of AWS infrastructure. When it finishes, it will print out the command for connecting to the new server. The server is preloaded with the software required for the course.
5. Learn how to use the provided AWS aliases on the [wiki](http://wiki.fast.ai/index.php/AWS_install#Once_you_create_an_instance).

### Azure
Once you have an Azure GPU server set up, download and run the `install-gpu-azure.sh` script on that server. More instructions available on the [wiki](http://wiki.fast.ai/index.php/Azure_install).

### Ubuntu
Download and run the `install-gpu.sh` script to install required software on an Ubuntu machine. More instructions available on the [wiki](http://wiki.fast.ai/index.php/Ubuntu_installation).