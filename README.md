#  simulated gym environments for the baesian balancing monopod platform.

## Installation for use without git control
` pip3 install git+https://github.com/Baesian-Balancer/gym-bb.git `

## Install as a developer with git control

`cd location/you/want/repo`

`git clone git@github.com:Baesian-Balancer/gym-bb.git`

`cd gym-bb`

`pip install -e .`

Now you can import gym_bb in python globally while developing and making changes to the environments in the repo

Many examples of how to use this package can be found in the examples folder.


# Installing Ignition Fortress simulator...

* ```
sudo apt-get update
sudo apt-get install lsb-release wget gnupg
```
* ```
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install ignition-fortress
```
* ```
# Remove the config folder so that Ignition creates a new one the next time it starts
mv $HOME/.ignition $HOME/.ignition_bak

# If there are still folder errors, try to create the folder yourself
mkdir -p $HOME/.ignition/gazebo/6
 ```
 * add this to your bashrc
 ```
 export IGN_GAZEBO_PHYSICS_ENGINE_PATH=${IGN_GAZEBO_PHYSICS_ENGINE_PATH}:/usr/lib/x86_64-linux-gnu/ign-physics-5/engine-plugins/
 ```
