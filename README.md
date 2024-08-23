<p align="center">
    <img src="./public/logo_light.png" width="100px" alt="WikiChat Logo" style="display: block; margin: 0 auto;" />
    <h1 align="center">
        <b>WikiChat</b>
        <br>
        <a href="https://arxiv.org/abs/2305.14292">
            <img src="https://img.shields.io/badge/cs.CL-2305.14292-b31b1b" alt="arXiv">
        </a>
        <a href="https://github.com/stanford-oval/WikiChat/stargazers">
            <img src="https://img.shields.io/github/stars/stanford-oval/WikiChat?style=social" alt="Github Stars">
        </a>
    </h1>
</p>
<p align="center">
    Stopping the Hallucination of Large Language Models
</p>
<p align="center">
    <!-- <a href="https://stanford.edu" target="_blank">
        <img src="./public/stanford.png" width="140px" alt="Stanford University" />
    </a> -->
</p>
<p align="center">
    Online demo:
    <a href="https://wikichat.genie.stanford.edu" target="_blank">
        https://wikichat.genie.stanford.edu
    </a>
    <br>
</p>

<!-- <hr /> -->


# Introduction

Large language model (LLM) chatbots like ChatGPT and GPT-4 get things wrong a lot, especially if the information you are looking for is recent ("Tell me about the 2024 Super Bowl.") or about less popular topics ("What are some good movies to watch from [insert your favorite foreign director]?").
WikiChat uses Wikipedia and the following 7-stage pipeline to makes sure its responses are factual.


<p align="center">
    <img src="./public/pipeline.svg" width="700px" alt="WikiChat Pipeline" />
</p>

Check out our paper for more details:
Sina J. Semnani, Violet Z. Yao*, Heidi C. Zhang*, and Monica S. Lam. 2023. [WikiChat: Stopping the Hallucination of Large Language Model Chatbots by Few-Shot Grounding on Wikipedia](https://arxiv.org/abs/2305.14292). In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore. Association for Computational Linguistics.

## 🚨 **Announcements** 
- (August 22, 2024) WikiChat 2.0 is now available! Highlights:
    - WikiChat now supports retrieval from structured data like tables, infoboxes and lists, in addition to text.
    - WikiChat is now multilingual. By default, it retrieves information from 10 Wikipedias ( :us: English, 🇨🇳 Chinese, 🇪🇸 Spanish, 🇵🇹 Portuguese, 🇷🇺 Russian, 🇩🇪 German, 🇮🇷 Farsi, 🇯🇵 Japanese, 🇫🇷 French, 🇮🇹 Italian)
    - Supports 100+ LLMs through a unified interface, thanks to [LiteLLM](https://github.com/BerriAI/litellm).
    - Uses the state-of-the-art multilingual retrieval model [BGE-M3](https://huggingface.co/BAAI/bge-m3).
    - Uses [Qdrant](https://github.com/qdrant/qdrant) for scalable vector search. We also provide a high-quality free (but rate-limited) search API for access to 10 Wikipedias, over 250M vector embeddings.
    - Option for faster and cheaper pipeline by merging the "generate" and "extract claim" stages.
    - Has the highest quality public Wikipedia preprocessing scripts (event better than what is used to pre-train LLMs, see below).
    - Uses and is compatible with LangChain 🦜️🔗.
    - Uses [RankGPT](https://github.com/sunnweiwei/RankGPT) for more relevant results.
    - Lots more!
- (June 20, 2024) WikiChat won the 2024 Wikimedia Research Award!
  <blockquote class="twitter-tweet"><p lang="en" dir="ltr">The <a href="https://twitter.com/Wikimedia?ref_src=twsrc%5Etfw">@Wikimedia</a> Research Award of the Year 2024 goes to &quot;WikiChat: Stopping the hallucination of large language model chatbots by few-shot grounding on Wikipedia&quot; ⚡<br><br>📜 <a href="https://t.co/d2M8Qrarkw">https://t.co/d2M8Qrarkw</a> <a href="https://t.co/P2Sh47vkyi">pic.twitter.com/P2Sh47vkyi</a></p>&mdash; Wiki Workshop 2024 (@wikiworkshop) <a href="https://twitter.com/wikiworkshop/status/1803793163665977481?ref_src=twsrc%5Etfw">June 20, 2024</a></blockquote>
  
- (May 16, 2024) Our follow-up paper _"🍝 SPAGHETTI: Open-Domain Question Answering from Heterogeneous Data Sources with Retrieval and Semantic Parsing"_ is accepted to the Findings of ACL 2024. This paper adds support for structured data like tables, infoboxes and lists.
- (January 8, 2024) Distilled LLaMA-2 models are released. You can run these models locally for a cheaper and faster alternative to paid APIs.
- (December 8, 2023) We present our work at EMNLP 2023.
- (October 27, 2023) The camera-ready version of our paper is now available on arXiv.
- (October 06, 2023) Our paper is accepted to the Findings of EMNLP 2023.


# Installation

Installing WikiChat involves the following steps:

1. Install dependencies
2. Configure the LLM of your choice. WikiChat supports over 100 LLMs, including models from OpenAI, Azure, Anthropic, Mistral, HuggingFace, Together.ai, and Groq.
3. Select an information retrieval source. This can be any HTTP endpoint that conforms to the interface defined in `retrieval/retriever_server.py`. We provide instructions and scripts for the following options:
    1. Use our free, rate-limited API for Wikipedia in 10 languages.
    1. Download and host our provided Wikipedia index yourself.
    1. Create and run a new custom index from your own documents.
4. Run WikiChat with your desired configuration.
5. [Optional] Deploy WikiChat for multi-user access. We provide code to deploy a simple front-end and backend, as well as instructions to connect to an Azure Cosmos DB database for storing conversations.

This project has been tested using Python 3.10 on Ubuntu Focal 20.04 (LTS), but should run on most recent Linux distributions.
If you want to use this on Windows WSL or Mac, or with a different Python version, expect to do some troubleshooting in some of the installation steps.

## Install Dependencies

Clone the repo:
```
git clone https://github.com/stanford-oval/WikiChat.git
cd WikiChat
```


We recommend using the conda environment in `conda_env.yaml`. This will create a conda environment with [Python 3.10](https://docs.python.org/3.10/), and install [pip](https://pip.pypa.io/en/stable/), [gcc](https://gcc.gnu.org/onlinedocs/), [g++](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/libstdc++/manual/), [make](https://www.gnu.org/software/make/manual/make.html), [Redis](https://redis.io/documentation), and all required Python packages.

Make sure you have one of [Conda](https://docs.conda.io/en/latest/), [Anaconda](https://docs.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

Create and activate the conda environment by running:

```bash
conda env create --file conda_env.yml
conda activate wikichat
python -m spacy download en_core_web_sm # Spacy is only needed for certain configurations of WikiChat
```
Make sure this environment is activated whenever you run any of the following commands.

Install Docker for your operating system by following the instructions at https://docs.docker.com/engine/install/. This project uses Docker primarily for creating and serving vector databases for retrieval, specifically [🤗 Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference) and [Qdrant](https://github.com/qdrant/qdrant). If you are using a recent Ubuntu, you can try running `inv install-docker`

WikiChat uses `invoke` (https://www.pyinvoke.org/) to add custom commands to WikiChat for various purposes.
You can run `invoke --list`, or the shorthand `inv -l` to list all available commands and a short description of what they do. You can also run `inv [command name] --help` to see more details about each of the available commands.
These commands are implemented in the `tasks/` folder.


## Configure the LLM of Your Choice

WikiChat is compatible with various LLMs including models from OpenAI, Azure, Anthropic, Mistral, Together.ai, and Groq.
You can also use WikiChat with many locally hosted models via HuggingFace.
To configure the LLM you want to use, fill out the appropriate fields in `llm_config.yaml`.

Then create a file named `API_KEYS` (which is included in `.gitignore`), and set the API key for the LLM endpoint you want to use.
The name of the API key in this file should match the name you provide in `llm_config.yaml` under `api_key`.
For example, if you are using OpenAI models via openai.com and Mistral endpoints in your code, your `API_KEYS` file might look like this:

```bash
# Fill in the following values with your own keys for the API you are using. Make sure there is not extra space after the key.
# Changes to this file are ignored by git, so that you can safely store your keys here during development.
OPENAI_API_KEY=[Your OpenAI API key from https://platform.openai.com/api-keys]
MISTRAL_API_KEY=[Your Mistral API key from https://console.mistral.ai/api-keys/]
```

Note that locally hosted models do NOT need an api key, but you need to provide an OpenAI-compatible endpoint in `api_base`. The code has been tested with [🤗 Text Generation Inference](https://github.com/huggingface/text-generation-inference/) endpoints, but you can try other similar endpoints like [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), etc.


## Configure an Information Retrieval Source

### Option 1 (default): Use our free rate-limited Wikipedia search API
By default, WikiChat retrieves information from 10 Wikipedias via the endpoint at https://wikichat.genie.stanford.edu/search/. If you want to just try WikiChat, you do not need to modify anything.

### Option 2: Download and host our Wikipedia index
Run `inv download-wikipedia-index --workdir ./workdir` to download the index from [stanford-oval/wikipedia_10-languages_bge-m3_qdrant_index](🤗 Hub) and extract it.

Note that this index contains ~180M vector embeddings and therefore requires a at least 800 GB of empty disk space. It uses [Qdrant's binary quantization](https://qdrant.tech/articles/binary-quantization/) to reduce RAM requirements to 55 GB without sacrificing accuracy or latency.

This command will start a FastAPI server similar to option 1 that responds to HTTP POST requests. Note that this server and even its embedding model runs on CPU, and does not require GPU. For better performance, on compatible systems you can add `--user-onnx` to use the ONNX version of the embedding model, for lower latency.
`inv start-retriever --embedding-model BAAI/bge-m3 --use-onnx --retriever-port <port number>`

### Option 3: Build your own index using your own Documents
The following command will download, preprocess, and index the latest HTML dump of the [Kurdish Wikipedia](ku.wikipedia.org), which we use in this example for its relatively small size.

`inv index-wikipedia-dump  --embedding-model BAAI/bge-m3 --workdir ./workdir --language ku`

For indexing a different set of documents, you need to preprocess your data into a [JSON Lines](https://jsonlines.org/) file (with .jsonl or .jsonl.gz file extension) where each line  has the following fields:
```json
{"content_string": "string", "article_title": "string", "full_section_title": "string", "block_type": "string", "language": "string", "last_edit_date": "string (optional)", "num_tokens": "integer (optional)"}
```
`content_string` should be the chunked text of your documents. We recommend chunking to less than 500 tokens of the embedding model's tokenizer. See [this](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/) for an overview on chunking methods.
`block_type` and `language` are only used to provide filtering on search results. If you do not need them, you can simply set them to `block_type=text` and `language=en`.
The script will feed `full_section_title` and `content_string` to the embedding model to create embedding vectors.

See `wikipedia_preprocessing/preprocess_html_dump.py` for details on how this is implemented for Wikipedia HTML dumps.

Then run `inv index-collection --collection-path <path to your preprocessed JSONL collection file> --collection-name `



## Run WikiChat in Terminal

You can run different configurations of WikiChat using commands like these:

```
inv demo --engine gpt-4o # engine can be any value configured in llm_config, for example, mistral-large, claude-sonnet-35, local
inv demo --pipeline generate_and_correct # available pipelines are early_combine, generate_and_correct and retrieve_and_generate
inv demo --temperature 0.9 # changes the temperature of the user-facing stages like refinement
```

For a full list of all available options, you can run `inv demo --help`

## [Optional] Deploy WikiChat for Multi-user Access
This repository provides code to deploy a web-based chat interface via [Chainlit](https://github.com/Chainlit/chainlit), and store user conversations to a [Cosmos DB](https://azure.microsoft.com/en-us/products/cosmos-db) database.
These are implemented in `backend_server.py` and `database.py` respectively. If you want to use other databases or front-ends, you need to modify these files. For development, it should be straightforward to remove the dependency on Cosmos DB and simply store conversations in memory.
You can also configure chatbot parameters defined in `backend_server.py`, for example to use a different LLM or add/remove stages of WikiChat.

### Set up Cosmos DB
After creating an instance via Azure, obtain the connection string and add this value in `API_KEYS`.
```bash
COSMOS_CONNECTION_STRING=[Your Cosmos DB connection string]
```

Running this will start the backend and front-end servers. You can then access the front-end at the specified port (5001 by default).
`inv chainlit --backend-port 5001`



# Other Commands

## Using the free Rate-limited Wikipedia search API
See https://wikichat.genie.stanford.edu/search/redoc

## Simulate User



## Upload an Index
Split the index into smaller files:
`tar -cvf - ./workdir/qdrant_index/ | pigz -p 14 | split --bytes=10GB --numeric-suffixes=0 --suffix-length=4 - /mnt/ephemeral_nvme/qdrant_index.tar.gz.part-`

Then update the arguments in `retrieval/upload_to_hf_hub.py` and run it.


## Run a distilled model for lower latency and cost
WikiChat 2.0 is not compatible with (fine-tuned LLaMA-2 checkpoints released)[https://huggingface.co/collections/stanford-oval/wikichat-v10-66c580bf15e26b87d622498c]. Please refer to v1.0 for now.

## Simulate Conversations
In order to evaluate a chatbot, you can simulate conversations with a user simulator. `subset` can be one of `head`, `tail`, or `recent`, corresponding to the three subsets introduced in the WikiChat paper. We have also added the option to specify the language of the user (WikiChat always replies in the language of the user).
This script will read the topic (i.e. a Wikipedia title and article) from the corresponding `benchmark/topics/(subset)_articles_(language).json.` file. `--num-dialogs` is the number of simulated dialogs to generate, and `--num-turns` is the number of turns in each dialog.

```bash
inv simulate-users --num-dialogs 1 --num-turns 2 --simulation-mode passage --language en --subset head
```
Depending on the engine you are using, this might take some time. The simulated dialogs and the log file will be saved in `benchmark/simulated_dialogs/`.
You can also provide any of the pipeline parameters from above.
You can experiment with different user characteristics by modifying `user_characteristics` in `benchmark/user_simulator.py`.


# Wikipedia Preprocessing: Why is it difficult?


# License
WikiChat code and models are released under Apache-2.0 license.



# Citation

If you have used code or data from this repository, please cite the following papers:

```bibtex
@inproceedings{semnani-etal-2023-wikichat,
    title = "{W}iki{C}hat: Stopping the Hallucination of Large Language Model Chatbots by Few-Shot Grounding on {W}ikipedia",
    author = "Semnani, Sina  and
      Yao, Violet  and
      Zhang, Heidi  and
      Lam, Monica",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.157",
    pages = "2387--2413",
}

@inproceedings{zhang-etal-2024-spaghetti,
    title = "{SPAGHETTI}: Open-Domain Question Answering from Heterogeneous Data Sources with Retrieval and Semantic Parsing",
    author = "Zhang, Heidi  and
      Semnani, Sina  and
      Ghassemi, Farhad  and
      Xu, Jialiang  and
      Liu, Shicheng  and
      Lam, Monica",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.96",
    pages = "1663--1678",
}
```