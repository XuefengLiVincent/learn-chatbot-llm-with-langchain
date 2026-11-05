## Project :  FAISS & OpenAI
....
> This is a Python application that uses the OpenAI API and LangChain to generate text based on a user's input and context retrieved from a vectorstore (FAISS)

## **Install Python** 

A [Quick Guide for Installing](https://github.com/PackeTsar/Install-Python/blob/master/README.md#install-python-) Python on Common Operating Systems


## packages

- **LangChain** :[LangChain](https://www.langchain.com/) is a Python library that translates text to and from any language. It uses the Google Translate API to translate text. It also uses the Google Cloud Text-to-Speech API to convert text to speech.
- **Faiss** : [FAISS](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/) : Facebook AI Similarity Search (Faiss) is a library for efficient similarity search and clustering of dense vectors. 


- **OpenAI** : [OpenAI](https://python.langchain.com/docs/integrations/platforms/openai) is a Python library that provides a simple interface to the OpenAI API. It also provides a command-line interface (CLI) for interacting with the API.
- **python-dotenv** : [python-dotenv](https://pypi.org/project/python-dotenv/) is a Python library that loads environment variables from a .env file. It is used to load the OpenAI API key from the .env file.
- **Streamlit** : [Streamlit](https://streamlit.io/) is a Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. It is used to create the web app.


....

## Create a virtual environment :

**MacOS/Linux**:
```
python3 -m venv env
```
**Windows**:
```
python -m venv env
```

## Activate the virtual environment :
**MacOS/Linux**:
```
source env/bin/activate
```
**Windows**:
```
.\env\Scripts\activate
```

## Installation: Install the necessary dependencies by running the following command:
**MacOS/Linux**:
```
pip3 install -r requirements.txt
pip3 install faiss-cpu
pip install -qU langchain-community
```
**Windows**:
```
pip install -r requirements.txt
pip install faiss-cpu
pip install -qU langchain-community
```

## [Get an API key](https://platform.openai.com/account/api-keys)

### Set the key as an environment variable:
Additionally, you need to obtain an OpenAI API key and add it to the `.env` file.

`export OPENAI_API_KEY='sk-brHeh...A39v5iXsM2'`

```
OPENAI_API_KEY=sk-brHeh...A39v5iXsM2
```

## Run the script:

**MacOS/Linux**:
```
python3 main.py
```
**Windows**:
```
python main.py


