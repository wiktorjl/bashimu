#!/bin/bash

# bashimu.sh - A simple script to interact with OpenAI's API for bash and Linux related questions
#
# Usage:
#   bashimu.sh <bash or linux related question>
#
# Requirements:
#   - Set the OPENAI_API_KEY environment variable with your OpenAI API key.
#   - Optionally, set the OPENAI_MODEL_NAME environment variable to specify the model (default: gpt-4o-mini).
#   - Ensure 'curl' and 'jq' are installed on your system.
#
# Example:
#   bashimu.sh "How do I list all files in a directory?"
#
# This script will send your question to the OpenAI API and return a concise command line answer.
#
# Author: Wiktor Lukasik
# Date: 2022-10-10
# Version: 1.0


# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Please set the OPENAI_API_KEY environment variable in ~/.bashimurc."
    exit 1
fi

# Check if OPENAI_MODEL_NAME is set and if not, set it to the default value (o1-mini)
# Models as of 2022-10-10: 
#   gpt-4o-mini, gpt-4o
#   gpt-4-turbo, gpt-4 
#   gpt-3.5-turbo, gpt-3.5
if [ -z "$OPENAI_MODEL_NAME" ]; then
    OPENAI_MODEL_NAME="gpt-4o-mini"
fi


# Check if the user has provided a prompt and if not, print helpful tips.
if [ -z "$1" ]; then
    # Display usage
    echo "Usage: bashimu.sh <bash or linux related question>"
    exit 1
fi

# If argument is ?, then display the last answer from the log file
# If log file empty or does not exist, then display a message
if [ "$1" == "?" ]; then
    if [ ! -f ~/.bashimu.log ]; then
        echo "No questions have been asked yet."
        exit 0
    fi

    tail -n 1 ~/.bashimu.log | sed 's/A: //'
    exit 0
fi

# If argument is !, then execute the last command from the log file
if [ "$1" == "!" ]; then
    if [ ! -f ~/.bashimu.log ]; then
        echo "No questions have been asked yet."
        exit 0
    fi

    last_command=$(tail -n 1 ~/.bashimu.log | sed 's/A: //')
    echo "Executing: $last_command"
    eval $last_command
    exit 0
fi

# Check if curl is installed
if ! command -v curl &> /dev/null
then
    echo "curl could not be found"
    exit 1
fi

query_str="$@"

# Call the OpenAI API
json=$(curl "https://api.openai.com/v1/chat/completions" \
    -s \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "'"$OPENAI_MODEL_NAME"'",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant and a bash/linux expert. Answer in one line with just the command line, no formatting or non-ASCII characters."
            },
            {
                "role": "user",
                "content": "'"$query_str"'"
            }
        ]
    }')


# Extract the response from the JSON
response=$(echo "$json" | jq -r '.choices[0].message.content')

# Log question and response to a log file
echo -E "Q: $query_str" >> ~/.bashimu.log
echo -E "A: $response" >> ~/.bashimu.log

# Print the response
echo -E "$response"

