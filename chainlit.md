
<p align="center">
    WikiChat is part of a research project at Stanford University's <a href="https://oval.cs.stanford.edu/"><b>Open Virtual Assistant Lab</b></a>.
</p>
<br>



# What is WikiChat?

Large language model (LLM) chatbots like ChatGPT and GPT-4 are great tools for quick access to knowledge.
But they get things wrong a lot, especially if the information you are looking for is recent ("Tell me about the 2024 Super Bowl.") or about less popular topics ("What are some good movies to watch from [insert your favorite foreign director]?").
  
WikiChat uses an LLM as its backbone, but it makes sure the information it provides comes from a reliable source like Wikipedia, so that its responses are more factual.


# What is this website?

We are hosting WikiChat to better understand the system in the wild. Thank you for giving it a try!
For further research on factual chatbots, we store conversations conducted on this website in a secure database. Only the text that you submit is stored. We do NOT collect or store any other information.

# I found a factual mistake in WikiChat's responses.
In our benchmarks, the version of WikiChat that uses GPT-4 as its backbone achieves a factual accuracy of 97.9%, much better than GPT-4 on its own. However, the default version on this website uses OpenAI's gpt-35-turbo-instruct because of its lower cost and latency, which means there will be more inaccuracies.
For the highest factual accuracy, we recommend using WikiChat with GPT-4. You can try it by selecting the "Most Factual" system from the sidebar.
You can try changing even more settings (and prompts) by following the step-by-step guide at the [WikiChat GitHub Repository](https://github.com/stanford-oval/WikiChat).

# How does WikiChat work?
Given the user input and the history of the conversation, WikiChat performs the following actions:

1. Searches Wikipedia to retrieve relevant information.
1. Summarizes and filters the retrieved passages.
1. Generates a response using a Language Learning Model (LLM).
1. Extracts claims from the LLM response.
1. Fact-checks the claims in the LLM response using additional retrieved evidence it retrieves from Wikipedia.
1. Drafts a response.
1. Refines the drafted response.

The following figure shows how these steps are applied during a sample conversation about an upcoming movie at the time, edited for brevity.
<p align="center">
    <img src="public/pipeline.svg" width="800px" alt="WikiChat Pipeline" />
</p>


# How can I learn more?

Check out our paper!

Sina J. Semnani, Violet Z. Yao*, Heidi C. Zhang*, and Monica S. Lam. 2023. **WikiChat: Stopping the Hallucination of Large Language Model Chatbots by Few-Shot Grounding on Wikipedia**. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore. Association for Computational Linguistics. [[arXiv]](https://arxiv.org/abs/2305.14292) [[ACL Anthology]](https://aclanthology.org/2023.findings-emnlp.157/)


# Contact Us

Email: [genie@cs.stanford.edu](mailto:genie@cs.stanford.edu)