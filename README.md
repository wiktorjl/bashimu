<img src="img/bashimu.jpeg" alt="drawing" width="200"/>

### About
BASHIMU is a simple bash script to interact with OpenAI's ChatGPT models.
It allows you to ask a question in the command line and get a one-liner response as a result.


### Setup
The simplest way is to use the deployment script:
```curl -s https://raw.githubusercontent.com/wiktorjl/bashimu/refs/heads/main/deploy.sh  | sh```

Otherwise, clone this repository and run ```bashimu_setup.sh``` script. It will set up the following variables:

```
$OPENAI_API_KEY - your private key
$OPENAI_MODEL_NAME - the default OpenAI model to be used
```

It will also ask you where to deploy the script, adjust the PATH accordingly, and set up ```?``` as an alias for ```bashimu.sh```.
Please view the script's source code before you run it so you know what it is doing.

### Usage:
Executing a query: ```? get current time```

Executing the last suggested command: ```? !```


### Demo:
<img src="img/bashimu_demo_2x.gif" width="800"/>
