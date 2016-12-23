# Docker Quickstart

1. [Install Docker](https://docs.docker.com/engine/installation/)

2. Clone this repo
    ```bash
    git clone https://github.com/fastai/courses.git
    ```
3. Add this in your .bash_profile or equivalent

    ```
      jn() {
        docker run --rm -it -v $PWD:/src -v $PWD/data:/data -p 8888:8888 -e "KERAS_BACKEND=tensorflow" waieez/keras-notebook:latest
      }
    ```
4. cd into this directory and type jn or use the entire docker run command above to launch juptyer notebook in a container. The first time you run this it will pull the image from [Dockerhub](https://hub.docker.com/r/jupyter/)
    ```bash
      jn
      
    ```
