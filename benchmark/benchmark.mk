# all targets in this file are related to simulating users

num_output_dialogs ?= -1# used for simulate-users only. -1 means as many as there are topics in the input file
num_output_turns ?= 5
simulation_mode = passage# topic or passage
subset ?= head# recent, head, tail, test_topics, multihop, distillation
input_file ?= $(subset)_articles.json# recent_articles.json, head_articles.json, tail_articles.json
num_workers ?= 3
user_engine ?= gpt-4
user_temperature ?= 1.0

simulate-users:
	python benchmark/scripts/user_simulator.py \
		$(PIPELINE_FLAGS) \
		--num_dialogs $(num_output_dialogs) \
		--user_engine $(user_engine) \
		--user_temperature $(user_temperature) \
		--mode $(simulation_mode) \
		--input_file benchmark/topics/$(input_file) \
		--num_turns $(num_output_turns) \
		--output_file benchmark/simulated_dialogs/$(pipeline)_$(subset)_$(engine).txt \
		--no_logging \
		--num_workers $(num_workers) \

benchmark-articles:
	python benchmark/scripts/get_wikipedia_articles_for_benchmark.py \
		--collection_path ./workdir/wikipedia_04_28_2023/collection_all.tsv \
		--recent_output_file benchmark/topics/recent_articles.json \
		--head_output_file benchmark/topics/head_articles.json \
		--tail_output_file benchmark/topics/tail_articles.json \
		--multihop_output_file benchmark/topics/multihop_articles.json

format-simulated-data:
	python benchmark/scripts/prepare_for_scale.py \
		--subset $(subset) \
		--pipeline $(pipeline) \
		--claim_prompt_template_file benchmark/prompts/split_claim_for_eval.prompt \
		--colbert_endpoint $(colbert_endpoint) \
		--engine $(engine) \
		--reranking_method none \
		--evi_num 5 \
		--scale_template_file benchmark/prompts/scale_factuality.prompt \
		--output_file benchmark/crowdsource_data/$(pipeline)_$(subset)_$(engine)_scale_input.csv \
		--input_file benchmark/simulated_dialogs/$(pipeline)_$(subset)_$(engine).txt \
		--temperature $(temperature) \

automatic-eval:
	python benchmark/scripts/automatic_eval.py \
		--file_to_eval benchmark/simulated_dialogs/$(pipeline)_$(subset)_$(engine).txt \
		--feedback_output_file benchmark/evaluation_results/$(pipeline)_$(subset)_$(engine)-feedback.json \
		--pipeline $(pipeline) \
		--engine $(engine) \
		--mode evaluate_file


aggregate-automatic-eval:
	python benchmark/scripts/automatic_eval.py \
		--file_to_eval benchmark/simulated_dialogs/$(pipeline)_$(subset)_$(engine).txt \
		--feedback_output_file benchmark/evaluation_results/$(pipeline)_$(subset)_$(engine)-feedback.json \
		--pipeline $(pipeline) \
		--engine $(engine) \
		--score_output_file benchmark/evaluation_results/automatic_eval_scores.csv \
		--mode get_stats

process-scale-outputs:
	python benchmark/scripts/scale_factuality_analysis.py \
		--input_file benchmark/crowdsource_data/emnlp_submission_full.json \
		--output_file benchmark/evaluation_results/emnlp_submission_full.txt