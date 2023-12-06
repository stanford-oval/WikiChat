include ./API_KEYS
include benchmark/benchmark.mk

### Pipeline parameters and their default values
engine ?= gpt-4
temperature ?= 0.0# only used for user-facing prompts. All other prompts use temperature 0.0 regardless of this value
pipeline ?= early_combine# one of: generate_and_correct, retrieve_and_generate, generate, retrieve_only, early_combine, atlas
evi_num ?= 2# num evi for each subclaim verification
retrieval_num ?= 3# for initial retrieval
reranking_method ?= date# none or date
skip_verification ?= false
do_refine ?= true
refinement_prompt ?= refine_w_feedback.prompt

colbert_endpoint ?= http://127.0.0.1:5000/search
redis_port ?= 5003
debug_mode ?= false

PIPELINE_FLAGS = --pipeline $(pipeline) \
		--engine $(engine) \
		--claim_prompt_template_file split_claims.prompt \
		--refinement_prompt $(refinement_prompt) \
		--colbert_endpoint $(colbert_endpoint) \
		--reranking_method $(reranking_method) \
		--evi_num $(evi_num) \
		--retrieval_num $(retrieval_num) \
		--temperature $(temperature)

ifdef generate_engine
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --generate_engine $(generate_engine)
endif

ifdef draft_engine
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --draft_engine $(draft_engine)
endif

ifeq ($(do_refine), true)
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --do_refine
else
	PIPELINE_FLAGS := $(PIPELINE_FLAGS)
endif

ifeq ($(skip_verification), true)
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --skip_verification
else
	PIPELINE_FLAGS := $(PIPELINE_FLAGS)
endif

ifeq ($(debug_mode), true)
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --debug_mode
else
	PIPELINE_FLAGS := $(PIPELINE_FLAGS)
endif

.PHONY: demo simulate-users start-colbert start-backend start-colbert-gunicorn start-backend-gunicorn download-and-index-wiki download-latest-wiki extract-wiki split-wiki index-wiki db-to-file run-tests

demo:
	python chat_interactive.py \
		$(PIPELINE_FLAGS) \
		--output_file data/demo.txt \

start-colbert:
	python colbert_app.py \
		--colbert_index_path workdir/$(language)/wikipedia_$(wiki_date)/$(index_name) \
		--colbert_checkpoint $(colbert_checkpoint) \
		--colbert_collection_path $(collection) \
		--memory_map "$(colbert_memory_map)"

# using more than 1 worker leads to loading the ColBERT index multiple times, which leads to running out of RAM
start-colbert-gunicorn:
	gunicorn \
		'colbert_app:gunicorn_app( \
		colbert_index_path="workdir/$(language)/wikipedia_$(wiki_date)/$(index_name)", \
		colbert_checkpoint="$(colbert_checkpoint)", \
		colbert_collection_path="$(collection)", \
		memory_map="$(colbert_memory_map)")' \
		--access-logfile=- \
		--bind 0.0.0.0:5000 \
		--workers 1 \
		--timeout 0

start-backend:
	redis-server --port $(redis_port) --daemonize yes
	python server_api.py \
		$(PIPELINE_FLAGS) \
		# --test
		# --no_logging


# can only pass gunicorn built-in arguments; pipeline args uses the defaults
ssl_key_file ?= /path/to/privkey.pem
ssl_certificate_file ?= /path/to/fullchain.pem
start-backend-gunicorn:
	redis-server --port $(redis_port) --daemonize yes
	gunicorn \
		'server_api:gunicorn_app( \
		input_flags="$(PIPELINE_FLAGS)" \
		)' \
		--bind 0.0.0.0:5001 \
		--workers 4 \
		--keyfile $(ssl_key_file) \
		--certfile $(ssl_certificate_file) \
		--access-logfile=- \
		--timeout 0


##### ColBERT indexing #####
nbits ?= 1# number of bits to encode each dimension with
max_block_words ?= 100# maximum number of words in each paragraph
doc_maxlen ?= 140# number of "tokens", to account for (100 "words" + title) that we include in each wikipedia paragraph
colbert_checkpoint ?= colbert-ir/colbertv2.0
split ?= all# '1m' passages or 'all'
experiment_name ?= wikipedia_$(split)
index_name ?= wikipedia.$(split).$(nbits)bits
wiki_date ?= 11_06_2023
language ?= en
collection ?= ./workdir/$(language)/wikipedia_$(wiki_date)/collection_$(split).tsv
nranks ?= 8# number of GPUs to use
colbert_memory_map ?= false


download-colbert-index:
	python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="stanford-oval/wikipedia_colbert_index", repo_type="dataset", local_dir="workdir/", local_dir_use_symlinks=False, allow_patterns="*.tar.*")' ; \
	cat workdir/$(language)/wikipedia_$(wiki_date).tar.* | tar xvf - -C ./workdir/$(language)/ ; \
	rm workdir/$(language)/wikipedia_$(wiki_date).tar.*

# Takes ~70 minutes. using `axel -o` instead of `wget -O` can speed up the download
download-latest-wiki: workdir/
	wget \
		-O workdir/$(language)/wikipedia_$(wiki_date)/$(language)wiki-latest-pages-articles.xml.bz2 \
		https://dumps.wikimedia.org/$(language)wiki/latest/$(language)wiki-latest-pages-articles.xml.bz2

# Takes ~5 hours. Will output a lot of template warnings, which is fine.
extract-wiki: workdir/$(language)/wikipedia_$(wiki_date)/$(language)wiki-latest-pages-articles.xml.bz2
	python -m wikiextractor.wikiextractor.WikiExtractor workdir/$(language)/$(language)wiki-latest-pages-articles.xml.bz2 \
			--templates workdir/$(language)/wiki_templates.txt \
			--output workdir/$(language)/wikipedia_$(wiki_date)/text/ \
			--links

# Takes ~1 minute for English, but more for other languages because we need to also find entity translations
split-wiki: workdir/$(language)/wikipedia_$(wiki_date)/text/
	python wikiextractor/split_passages.py \
		--input_path workdir/$(language)/text/ \
		--output_path $(collection) \
		--max_block_words $(max_block_words) \
		--language $(language) \
		--translation_cache workdir/translation_cache.json

# Takes 24 hours on a 40GB A100 GPU
index-wiki: $(collection)
	python index_wiki.py \
		--nbits $(nbits) \
		--doc_maxlen $(doc_maxlen) \
		--checkpoint $(colbert_checkpoint) \
		--split $(split) \
		--experiment_name $(experiment_name) \
		--index_name $(index_name) \
		--collection $(collection) \
		--nranks $(nranks)

# Takes ~10 minutes
coalesce-index: experiments/$(experiment_name)/indexes/$(index_name)
	python ColBERT/colbert/utils/coalesce.py \
		--input experiments/$(experiment_name)/indexes/$(index_name) \
		--output workdir/$(language)/wikipedia_$(wiki_date)/$(index_name)

tar-wiki-index:
	tar -cvf colbert_wikipedia_index_$(language)_$(wiki_date).tar workdir/$(language)/wikipedia_$(wiki_date)/