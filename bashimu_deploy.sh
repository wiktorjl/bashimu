#!/bin/bash

# Generate a random filename in /tmp
sh_file="/tmp/bashimu_setup_$(date +%s).sh"

# Download the bashimu.sh script to /tmp
curl -s https://raw.githubusercontent.com/wiktorjl/bashimu/main/bashimu.sh > $sh_file 2>/dev/null 

# Same for bashimu_setup.sh
setup_file="/tmp/bashimu_setup_$(date +%s).sh"
curl -s https://raw.githubusercontent.com/wiktorjl/bashimu/main/bashimu_setup.sh > $setup_file 2>/dev/null

# Run the setup script
bash $setup_file $sh_file

