<p align="center">
    <img src="./public/logo_light.png" width="120px" alt="WikiChat Logo" style="display: block; margin: 0 auto;" />
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




https://github.com/user-attachments/assets/3ac856ba-682c-4aed-9271-ce2f6a27cd5e



# Table of Contents
- [Introduction](#introduction)
  - [🚨 Announcements](#-announcements)
- [Installation](#installation)
  - [System Requirements](#system-requirements)
  - [Install Dependencies](#install-dependencies)
  - [Configure the LLM of Your Choice](#configure-the-llm-of-your-choice)
  - [Configure Information Retrieval](#configure-information-retrieval)
    - [Option 1 (Default): Use our free rate-limited Wikipedia search API](#option-1-default-use-our-free-rate-limited-wikipedia-search-api)
    - [Option 2: Download and host our Wikipedia index](#option-2-download-and-host-our-wikipedia-index)
    - [Option 3: Build your own index](#option-3-build-your-own-index)
      - [To build a Wikipedia index](#to-build-a-wikipedia-index)
      - [To index custom documents](#to-index-custom-documents)
      - [To upload a Qdrant index to 🤗 Hub:](#to-upload-a-qdrant-index-to--hub)
  - [Run WikiChat in Terminal](#run-wikichat-in-terminal)
  - [\[Optional\] Deploy WikiChat for Multi-user Access](#optional-deploy-wikichat-for-multi-user-access)
    - [Set up Cosmos DB](#set-up-cosmos-db)
    - [Run Chainlit](#run-chainlit)
- [The Free Rate-limited Wikipedia Search API](#the-free-rate-limited-wikipedia-search-api)
- [Wikipedia Preprocessing](#wikipedia-preprocessing)
- [Other Commands](#other-commands)
  - [Run a Distilled Model for Lower Latency and Cost](#run-a-distilled-model-for-lower-latency-and-cost)
  - [Simulate Conversations](#simulate-conversations)
- [License](#license)
- [Citation](#citation)



<!-- <hr /> -->


# Introduction

Large language model (LLM) chatbots like ChatGPT and GPT-4 get things wrong a lot, especially if the information you are looking for is recent ("Tell me about the 2024 Super Bowl.") or about less popular topics ("What are some good movies to watch from [insert your favorite foreign director]?").
WikiChat uses Wikipedia and the following 7-stage pipeline to makes sure its responses are factual. Each numbered stage involves one or more LLM calls.


<p align="center">
    <img src="./public/pipeline.svg" width="700px" alt="WikiChat Pipeline" />
</p>

Check out our paper for more details:
Sina J. Semnani, Violet Z. Yao*, Heidi C. Zhang*, and Monica S. Lam. 2023. [WikiChat: Stopping the Hallucination of Large Language Model Chatbots by Few-Shot Grounding on Wikipedia](https://arxiv.org/abs/2305.14292). In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore. Association for Computational Linguistics.

## 🚨 Announcements
- (August 22, 2024) WikiChat 2.0 is now available! Key updates include:
    - **Multilingual Support**: By default, retrieves information from 10 different Wikipedias: 🇺🇸 English, 🇨🇳 Chinese, 🇪🇸 Spanish, 🇵🇹 Portuguese, 🇷🇺 Russian, 🇩🇪 German, 🇮🇷 Farsi, 🇯🇵 Japanese, 🇫🇷 French, and 🇮🇹 Italian.
    - **Improved Information Retrieval**
      - Now supports retrieval from structured data such as tables, infoboxes, and lists, in addition to text.
      - Has the highest quality public Wikipedia preprocessing scripts
      - Uses the state-of-the-art multilingual retrieval model [BGE-M3](https://huggingface.co/BAAI/bge-m3).
      - Uses [Qdrant](https://github.com/qdrant/qdrant) for scalable vector search.
      - Uses [RankGPT](https://github.com/sunnweiwei/RankGPT) to rerank search results.
    - **Free Multilingual Wikipedia Search API**: We offer a high-quality, free (but rate-limited) search API for access to 10 Wikipedias, encompassing over 180M vector embeddings.

    - **Expanded LLM Compatibility**: Supports 100+ LLMs through a unified interface, thanks to [LiteLLM](https://github.com/BerriAI/litellm).
    - **Optimized Pipeline**: Option for a faster and more cost-effective pipeline by merging the "generate" and "extract claim" stages of WikiChat.
    - **LangChain Compatibility**: Fully compatible with LangChain 🦜️🔗.
    - **And Much More!**
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
1. Configure the LLM of your choice. WikiChat supports over 100 LLMs, including models from OpenAI, Azure, Anthropic, Mistral, HuggingFace, Together.ai, and Groq.
1. Select an information retrieval source. This can be any HTTP endpoint that conforms to the interface defined in `retrieval/retriever_server.py`. We provide instructions and scripts for the following options:
    1. Use our free, rate-limited API for Wikipedia in 10 languages.
    1. Download and host our provided Wikipedia index yourself.
    1. Create and run a new custom index from your own documents.
1. Run WikiChat with your desired configuration.
1. [Optional] Deploy WikiChat for multi-user access. We provide code to deploy a simple front-end and backend, as well as instructions to connect to an Azure Cosmos DB database for storing conversations.


## System Requirements
This project has been tested with Python 3.11 on Ubuntu 20.04 LTS (Focal Fossa), but it should be compatible with many other Linux distributions. If you plan to use this on Windows WSL or macOS, or with a different Python version, be prepared for potential troubleshooting during installation.

Hardware requirements vary based on your intended use:

1. **Basic Usage**: Running WikiChat with LLM APIs and our Wikipedia search API has minimal hardware requirements and should work on most systems.

1. **Local Search Index**: If you intend to host a search index locally, ensure you have sufficient disk space for the index. For large indices, retrieval latency is heavily dependant on disk speed, so we recommend using SSDs and preferably NVMe drives. For example, storage-optimized VMs like [`Standard_L8s_v3`](https://learn.microsoft.com/en-us/azure/virtual-machines/lsv3-series) on Azure are suitable for this.

1. **Local LLM**: If you plan to use WikiChat with a local LLM, a GPU is necessary to host the model.

1. **Creating a New Retrieval Index**: If you want to index a collection, you need a GPU to embed documents to vectors. The default embedding model (`BAAI/BGE-M3`) requires at least 13GB of GPU memory to run.


## Install Dependencies

First, clone the repository:
```
git clone https://github.com/stanford-oval/WikiChat.git
cd WikiChat
```

We recommend using the pixi environment specified in `pixi.toml`. This environment includes [Python 3.11](https://docs.python.org/3.11/), [pip](https://pip.pypa.io/en/stable/), [gcc](https://gcc.gnu.org/onlinedocs/), [g++](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/libstdc++/manual/), [make](https://www.gnu.org/software/make/manual/make.html), and all required Python packages.

[Pixi](https://pixi.sh/) is a cross-platform package management tool. It is a much faster alternative to conda. To install it, follow the instructions at https://pixi.sh/latest/#installation. Then create and activate the pixi environment:

```bash
pixi shell
python -m spacy download en_core_web_sm  # Spacy is only needed for user simulation
```

By default, this repository uses [Redis Stack](https://redis.io/about/about-stack/) via Docker.
If you see `Error: Redis lookup failed` after running the chatbot, it probably means Redis is not properly set up. Check the logs of the Redis docker container. Alternatively, you can try installing Redis by following its [official documentation](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/).

Keep this environment activated for all subsequent commands.

Install Docker for your operating system by following the instructions at https://docs.docker.com/engine/install/. WikiChat uses Docker primarily for creating and serving vector databases for retrieval, specifically [🤗 Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference) and [Qdrant](https://github.com/qdrant/qdrant). On recent Ubuntu versions, you can try running `inv install-docker`. For other operating systems, follow the instructions on the docker website.

WikiChat uses [`invoke`](https://www.pyinvoke.org/) to add custom commands for various purposes. To see all available commands and their descriptions, run:
```
invoke --list
```
or the shorthand:
```
inv -l
```

For more details about a specific command, use:
```
inv [command name] --help
```

These commands are implemented in the `tasks/` folder.


## Configure the LLM of Your Choice

WikiChat is compatible with various LLMs, including models from OpenAI, Azure, Anthropic, Mistral, Together.ai, and Groq.
You can also use WikiChat with many locally hosted models via HuggingFace.

To configure your LLM:
1. Fill out the appropriate fields in `llm_config.yaml`.

2. Create a file named `API_KEYS` (which is included in `.gitignore`).
3. In the `API_KEYS` file, set the API key for the LLM endpoint you want to use. The name of the API key should match the name you provided in `llm_config.yaml` under `api_key`.
For example, if you're using OpenAI models via openai.com and Mistral endpoints, your `API_KEYS` file might look like this:

```bash
# Fill in the following values with your API keys. Make sure there is not extra space after the key.
# Changes to this file are ignored by git, so you can safely store your keys here during development.
OPENAI_API_KEY=[Your OpenAI API key from https://platform.openai.com/api-keys]
MISTRAL_API_KEY=[Your Mistral API key from https://console.mistral.ai/api-keys/]
```

Note that locally hosted models do NOT need an API key, but you need to provide an OpenAI-compatible endpoint in `api_base`. The code has been tested with [🤗 Text Generation Inference](https://github.com/huggingface/text-generation-inference/) endpoints, but you can try other similar endpoints like [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), etc.


## Configure Information Retrieval

### Option 1 (Default): Use our free rate-limited Wikipedia search API
By default, WikiChat retrieves information from 25 Wikipedias via the endpoint at https://search.genie.stanford.edu/wikipedia_20250320/. If you want to just try WikiChat, you do not need to modify anything.

### Option 2: Build your own index
#### To build a Wikipedia index
The following command will download, preprocess, and index the latest HTML dump of the [Kurdish Wikipedia](ku.wikipedia.org), which we use in this example for its relatively small size.

```bash
inv index-wikipedia-dump  --embedding-model BAAI/bge-m3 --workdir ./workdir --language ku
```

#### To index custom documents

1. Preprocess your data into a [JSON Lines](https://jsonlines.org/) file (with .jsonl or .jsonl.gz file extension) where each line has the following fields:
```json
{"id": "integer",  "document_title": "string", "section_title": "string", "content": "string", "block_type": "string", "language": "string", "last_edit_date": "string (optional)", "url": "string (optional)", "num_tokens": "integer (optional)", "block_metadata": "dict (optional)"}
```
`content` should be the chunked text of your documents. We recommend chunking to less than 500 tokens of the embedding model's tokenizer. See [this](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/) for an overview on chunking methods.
`block_type` and `language` are only used to provide filtering on search results. If you do not need them, you can simply set them to `block_type=text` and `language=en`.
The script will feed `document_title` > `section_title` and `content` to the embedding model to create embedding vectors.

See `preprocessing/preprocess_wikipedia_html_dump.py` for details on how this is implemented for Wikipedia HTML dumps.

1. Run the indexing command:

```bash
inv index-collection --collection-path <path to preprocessed JSONL> --collection-name <collection name>
```

This command starts docker containers for [🤗 Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference) (one per available GPU). By default, it uses the docker image compatible with NVIDIA GPUs with Ampere 80 architecture, e.g. A100. Support for some other GPUs is also available, but you would need to choose the right docker image from [available docker images](https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker-images).

3. (Optional) Add a [payload index](https://qdrant.tech/documentation/concepts/payload/#payload-indexing)
```bash
python retrieval/add_payload_index.py
```
This will enable queries that filter on `language` or `block_type`. Note that for large indices, it might take several minutes for the index to become available again.

4. After indexing, load and use the index as in option 2. For example:
```bash
inv start-retriever --embedding-model-name BAAI/bge-m3 --retriever-port <port number>
curl -X POST 0.0.0.0:5100/<collection name> -H "Content-Type: application/json" -d '{"query": ["What is GPT-4?", "What is LLaMA-3?"], "num_blocks": 3}'
```

5. Start WikiChat by passing in the URL of this retriever. For example:
```bash
inv demo --retriever-endpoint "http://0.0.0.0:<port number>/<collection name>" --corpus_id "the id of the corpus you are using"
```

The `corpus_id` parameter is used to match a short description of the corpus to give to the LLM to help determine if searching the index is beneficial. It should refer to a `Corpus` object containing a short description of the corpus, e.g. "Kurdish Wikipedia" or "Business documents for company X".

#### To use an Azure AI deployment of the embedding model instead of a local one:
After deploying one of the available embedding models via Azure AI, add your endpoint's key as `EMBEDDING_API_KEY` to `API_KEYS`.
You can then use the following command to index your collection:

```bash
inv index-collection --collection-path <path to preprocessed JSONL> --collection-name <collection name> --embedding-model-name <model name> --embedding-model-url https://<deployment name>.<deployment region>.inference.ml.azure.com --embedding-model-port 443
```

`embedding-model-name` should be the name of the model you deployed on Azure, e.g. `BAAI/bge-m3`.
Note that Azure imposes batch_size limit depending on the model and deploymnet hardware. You are responsible for setting up and tearing down the Azure deployment. The URL and key for your endpoint can be found in the Azure portal under the deployment's details.


#### To upload a Qdrant index to 🤗 Hub:
1. Split the index into smaller parts:
```bash
tar -cvf - <path to the Qdrant index folder> | pigz -p 14 | split --bytes=10GB --numeric-suffixes=0 --suffix-length=4 - <path to the output folder>/qdrant_index.tar.gz.part-
```

2. Upload the resulting parts:
```bash
python retrieval/upload_folder_to_hf_hub.py --folder_path <path to the output folder> --repo_id <Repo ID on 🤗 Hub>
```



## Run WikiChat in Terminal

You can run different configurations of WikiChat using commands like these:

```
inv demo --engine gpt-4o # engine can be any value configured in llm_config, for example, mistral-large, claude-sonnet-35, local
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

### Run Chainlit
Running this will start the backend and front-end servers. You can then access the front-end at the specified port (5001 by default).
`inv chainlit --backend-port 5001`



# The Free Rate-limited Wikipedia Search API
You can use this API endpoint for prototyping high-quality RAG systems.
See https://search.genie.stanford.edu/redoc for the full specification.

Note that we do not provide any guarantees about this endpoint, and it is not suitable for production.


# Wikipedia Preprocessing
We publicly release [preprocessed Wikipedia in 10 languages](https://huggingface.co/datasets/stanford-oval/wikipedia).

# Other Commands

## Run a Distilled Model for Lower Latency and Cost
WikiChat 2.0 is not compatible with [fine-tuned LLaMA-2 checkpoints released](https://huggingface.co/collections/stanford-oval/wikichat-v10-66c580bf15e26b87d622498c). Please refer to v1.0 for now.

## Simulate Conversations
To evaluate a chatbot, you can simulate conversations using a user simulator. The `subset` parameter can be one of `head`, `tail`, or `recent`, corresponding to the three subsets introduced in the WikiChat paper. You can also specify the language of the user (WikiChat always replies in the user's language).
This script reads the topic (i.e., a Wikipedia title and article) from the corresponding `benchmark/topics/{subset}_articles_{language}.json` file. Use `--num-dialogues` to set the number of simulated dialogues to generate, and `--num-turns` to specify the number of turns in each dialogue.

```bash
inv simulate-users --num-dialogues 1 --num-turns 2 --simulation-mode passage --language en --subset head
```
Depending on the engine you are using, this might take some time. The simulated dialogues and log files will be saved in `benchmark/simulated_dialogues/`.
You can also provide any of the pipeline parameters from above.
You can experiment with different user characteristics by modifying `user_characteristics` in `benchmark/user_simulator.py`.

# License
WikiChat code, and models and data are released under Apache-2.0 license.

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
