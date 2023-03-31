# Notes

## Dockerfile
* Primary method for building and maintaining images.
* Encapsulate the OS, environmental variables, and configurations needed to run your programs.
* Transferrable into new settings.
## Context Directory
* By default Dockerfiles are only aware of what is in their **context directory**.  A context directory is the directory where a Dockerfile is located in.  It cannot access any files outside of the context directory.
* The context directory can be set from the command line or from a `docker_compose.yml` file.
* It is recommended that all folders/files be copied be placed in an **artifacts** directory that is under the context directory.
## Docker Compose
* Allows you to create multiple different builds using different Docker files.
* The context directory can be set from the `docker_compose.yml` file.
* More options than a standard Docker file.

## Running Commands
* In Docker each `RUN` command is treated as a new shell instance.  This means that commands cannot persist across different RUN commands.

```Docker
RUN command1    # this command is run in its own shell!
RUN command2    # this command is run in its own shell!
```

## Copying Folders/Files
* If you need to copy files or directories into the Docker container, first put them in an **artifacts** directory that can be seen from the context directory.


```Docker
COPY <artifact_directory> <container_path>
```
## Out of Memory
* Occasionally you will need to clean your docker images and docker volumes in order to recover memory.

> docker volume prune
>
> docker image prune
>
> docker container prune
