#!/bin/bash


# This script configures user's environment for bashimu


# Get OPENAI_API_KEY key
read -p "Please enter your OPENAI_API_KEY: " openai_api_key
if [ -z "$openai_api_key" ]; then
  echo "OPENAI_API_KEY is required."
  exit 1
fi

# Get OPENAI_MODEL_NAME value or default to 'gpt4o-mini'
read -p "Please enter the OPENAI_MODEL_NAME (default: gpt-4o-mini): " OPENAI_MODEL_NAME
OPENAI_MODEL_NAME=${openai_model_name:-gpt-4o-mini}

# Ask where to drop the script, default to ~/bin
read -p "Where would you like to drop the script? (default: ~/bin): " script_dir
script_dir=${script_dir:-~/bin}
mkdir -p "$script_dir"
echo "Script will be dropped in: $script_dir"

# Ask where to keep the log file, default to ~/.bashimu.log
read -p "Where would you like to keep the log file? (default: ~/.bashimu.log): " log_file
log_file=${log_file:-~/.bashimu.log}
echo "Log file will be kept in: $log_file"


# Print config values
echo
echo "Configuration values:"
echo "OPENAI KEY: " $OPENAI_API_KEY
echo "OPENAI MODEL: " $OPENAI_MODEL_NAME
echo "Script location: " $script_dir
echo "Log file location: " $log_file

# Continue?
read -p "I will add these to your shell's config file. Do you want to continue? (y/n): " continue
if [ "$continue" != "y" ]; then
  echo "Exiting..."
  exit 0
fi

# Find the name of this shell's config file
if [ -f ~/.bashrc ]; then
  config_file=~/.bashrc
elif [ -f ~/.bash_profile ]; then
  config_file=~/.bash_profile
else
  echo "Could not find a shell config file. Please add the following lines to your shell config file:"
  echo "export OPENAI_API_KEY=$OPENAI_API_KEY"
  echo "export OPENAI_MODEL_NAME=$OPENAI_MODEL_NAME"
  echo "export PATH=$PATH:$script_dir"
  echo "Exiting..."
  exit 1
fi

# Add the config values to the shell config file
echo "export OPENAI_API_KEY=$OPENAI_API_KEY" >> $config_file
echo "export OPENAI_MODEL_NAME=$OPENAI_MODEL_NAME" >> $config_file
echo "Added configuration values to $config_file"

# Check if $script_dir is in $PATH and if not, ask to add it
if [[ ! "$PATH" == *"$script_dir"* ]]; then
  echo "WARNING: $script_dir is not in your PATH."
  read -p "Do you want to add it to your PATH? (y/n): " add_to_path
  if [ "$add_to_path" == "y" ]; then
    echo "export PATH=$PATH:$script_dir" >> $config_file
    echo "Added $script_dir to PATH in $config_file"
  fi
fi

# Ask user if they want the convience alias
read -p "Do you want to add a convience alias for bashimu? (y/n): " add_alias
if [ "$add_alias" == "y" ]; then
  echo "alias ?=$script_dir/bashimu.sh" >> $config_file
  echo "Added alias for bashimu in $config_file"
fi

# Copy bashimy.sh to $script_dir
cp bashimu.sh $script_dir
echo "Copied bashimu.sh to $script_dir"

# Source the config file
source $config_file
