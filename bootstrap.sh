
##############################
#
# This is a bootstrap script which is
# run at every startup of the vagrant machine
# If you want to run something just once at provisioning
# and first bootup of the vagrant machine please see
# provision.sh
#
# Contributor: Bernhard Blieninger
##############################

python3 -m venv lstm-virtenv
source lstm-virtenv/bin/activate
pip3 install  -r python3-lstm/requirements.txt
