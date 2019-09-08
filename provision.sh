#!/bin/bash
#######################
#
# This is a provision script
# it will be called once when the vagrant vm is first provisioned
# If you have commands that you want to run always please have a
# look at the bootstrap.sh script
#
# Contributor: Bernhard Blieninger, Robert Hamsch
######################

sudo apt update -qq

sudo apt install python3.5 python3-pip tmux -qq

sudo apt install python3-venv
#pip3 install --user virtualenv
