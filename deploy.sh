#!/bin/bash

echo
echo
echo "██████       ████       ███████    ███   ███    ████     ████████    ███    ██"
echo "██    █     █    █     ██          ██    ██      ██     ██  ██  ██    ██    ██"
echo "██    █    ██    ██    ██          ██    ██      ██     ██  ██  ██    ██    ██"
echo "██████     ████████     ████       ████████      ██     ██  ██  ██    ██    ██"
echo "██    ██   █      █         ██     ██    ██      ██     ██  ██  ██    █     ██"
echo "██    ██   █      █         ██     ██    ██      ██     ██  ██  ██    █     ██"
echo "██████     ██     ██  ███████     ███  ████    █████   ██  ██   ██    ████████"
echo
echo  "Welcome to Bashimu! A simple script to interact with OpenAI's API for bash and Linux related questions."
echo  "This script will send your question to the OpenAI API and return a concise command line answer."
echo  "Please follow the prompts to configure your environment." 
echo 

# Download the bashimu.sh script to /tmp
sh_file="/tmp/bashimu_$(date +%s).sh"
echo "Downloading bashimu.sh script to $sh_file"
curl -s https://raw.githubusercontent.com/wiktorjl/bashimu/main/bashimu.sh > $sh_file 2>/dev/null 

echo "Moving the bashimu.sh script to /usr/local/bin"
sudo mv $sh_file /usr/local/bin/bashimu.sh
chmod +x /usr/local/bin/bashimu.sh

echo "Creating ~/.bashimurc file"
echo "#!/bin/sh" > ~/.bashimurc
echo "export OPENAI_API_KEY=\"\"" >> ~/.bashimurc
echo "export OPENAI_MODEL_NAME=\"gpt-4o-mini\"" >> ~/.bashimurc
echo "alias ?=\"/usr/local/bin/bashimu.sh\"" >> ~/.bashimurc

echo "Adding source ~/.bashimurc to ~/.bashrc"
echo "source ~/.bashimurc" >> ~/.bashrc

echo "Done! Please open a new terminal to start using Bashimu."
echo "Or you can run 'source ~/.bashrc' to start using it in the current terminal."