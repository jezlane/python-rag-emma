### Pre Requesits

1. Python

2. Poetry
``` bash
pip install poetry
```

### Installation
``` bash
poetry init
poetry install
poetry config virtualenvs.in-project true
```

### Add Libraries
``` bash
poetry add python-dotenv
poetry add langchain
poetry add openai
poetry add chromadb
poetry add tiktoken
```


### Run the program
``` bash
poetry shell
- python main.py
```