# Medical-Chatbot-Project



# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
uv init(if no project.toml)
uv venv(create venv)
```

```bash
.venv\Scripts\activate
```


### STEP 02- install the requirements
```bash
uv add <packages>
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
GROQ_API_KEY1=
GROQ_API_KEY2=
GROQ_BASE_URL=https://api.groq.com/openai/v1
HF_TOKEN=
```


```bash
# Finally run the following command
chainlit run app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- BMIndex

