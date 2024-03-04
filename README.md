## Getting Started

Install `docker` and `docker-compose` and have the **docker application running** before attempting to run commands with `docker`, `docker-compose`, or spinning up a development environment

Make sure to clone this repo an `cd` into the **root directory** of this project before running any commands

### Visual Studio code (vscode) & Docker (Recommended)

This method requires vscode to be installed as this method will attach to running local development containers using vscode. Futhermore, in vscode, two extension are **needed** `Dev Containers` and the `Docker` extension. These can be found and installed in vscode under the *Extensions* tab. Specificity, Dev Containers will allow us to to attach a vscode instance to a running container. Dev Containers can also spin up the containers too if they are not running already when attaching. 

First, open vscode in the root directory of this project. This is the directory where `docker-compose.dev.yml` is in.

Then open the Command Palette under View

the shortcut is as follows

Windows: `f1`

Mac: `Shift` + `Command` + `P`


1. Run Dev Containers: Open Folder in Container... from the Command Palette and select .devcontainer/cpu or .devcontainer/gpu (selects the type of hardware you are looking to use for ML)

2. VS Code will then start up the container and connect this window to it