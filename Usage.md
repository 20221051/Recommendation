# Usage

Please follow the below steps for installation and execution.

## Prerequisite 

* **Cam** connected as `device:video0`
* xhost commoand-line tool installed
    * `sudo apt-get install x11-xserver-utils` on **Ubuntu** and **Debian** based distributions
    * `sudo dnf install xorg-x11-xauth` on **Fedora** and **CentOS** based distributions

## Build

Build Docker image.

Note that `cs-project` can be replaced by preferred image name.

```bash
sudo docker build -t cs-project .
```

## Run

Run built Docker image.

Note that if you have built Docker image with preferred image name instead of `cs-project`, it should be applied here too.

```bash
sudo xhost +local:docker
sudo docker run -it --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix cs-project
```

* `--device=/dev/video0` flag allows the Docker container to access the video device (webcam) located at `/dev/video0`
* `-e DISPLAY=$DISPLAY` and `-v /tmp/.X11-unix:/tmp/.X11-unix` flags allow the Docker container to access **display**.
* Please note that sudo might not be required.

## More detail in docs directory
