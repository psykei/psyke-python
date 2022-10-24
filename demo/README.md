## About Notebooks files in this directory

Notebook files (`.ipynb`) in this directory contain demos of PSyKE.

These are NOT meant to be executed directly, but rather via some container started from the [PSyKE Image](https://hub.docker.com/r/pikalab/psyke) on DockerHub.

To do so, please ensure 
- [Docker](https://docs.docker.com/engine/install/) is installed on your system
- the Docker service is up and running

and then follow these steps:

1. Choose a target version of PSyKE, say `X.Y.Z`
    - cf. [DockerHub image tags](https://hub.docker.com/r/pikalab/psyke/tags)
    - cf. [PyPi releases history](https://pypi.org/project/psyke/#history)
    - cf. [GitHub releases](https://github.com/psykei/psyke-python/releases)

2. Pull the corresponding image:
    ```bash
    docker pull pikalab/psyke:X.Y.Z
    ```

2. Run the image into a container:
    ```bash
    docker run --rm -it -p 8888:8888 --name psyke pikalab/psyke:X.Y.Z
    ```

3. Some logs such as the following ones should eventually be printed
    ```
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-7-open.html
    Or copy and paste one of these URLs:
        http://66fa9b93bbe7:8888/?token=<LONG TOKEN HERE>
     or http://127.0.0.1:8888/?token=<LONG TOKEN HERE>
    ```

4. Open your browser and browse to `http://localhost:8888/?token=<LONG TOKEN HERE>`
    - use the token from the logs above, if requested by the Web page

5. Enjoy PSyKE-powered notebook!
