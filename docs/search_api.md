# Wikipedia Search API

## Endpoint
`POST /search`

## Description
This endpoint allows you to search in text, table and infoboxes of 10 Wikipedias (🇺🇸 English, 🇨🇳 Chinese, 🇪🇸 Spanish, 🇵🇹 Portuguese, 🇷🇺 Russian, 🇩🇪 German, 🇮🇷 Farsi, 🇯🇵 Japanese, 🇫🇷 French, 🇮🇹 Italian) with various query parameters.

It is currently retrieving from the Wikipedia dump of August 1, 2024.

The search endpoint is a hosted version of `retrieval/retriever_server.py`.
Specifically, it uses the state-of-the-art multilingual vector embedding models for high quality search results.
It also supports batch queries.

## Request Body
The request body should be a JSON object with the following fields:

- `query`: A string or a list of strings representing the search queries.
- `num_blocks`: An integer representing the number of items to retrieve.
- `languages`: (Optional) A string or a list of strings representing the language codes to filter the search results.

### Example
Search for the 3 most relevant text, table or infobox in the any of the 10 Wikipedia languages.
```http
POST https://search.genie.stanford.edu/wikipedia
Content-Type: application/json

{
  "query": ["What is GPT-4?", "What is LLaMA-3?"],
  "num_blocks": 3,
}
```

or equivalently, run

```
curl -X POST https://search.genie.stanford.edu/wikipedia -H "Content-Type: application/json" -d '{"query": ["What is GPT-4?", "What is LLaMA-3?"], "num_blocks": 3}'
```

Response:
```json
[
    {
        "score": [
            0.6902604699134827,
            0.6895850896835327,
            0.6812092661857605
        ],
        "text": [
            "GPT-4（ジーピーティーフォー、Generative Pre-trained Transformer 4）とは、OpenAI (in English: OpenAI)によって開発されたマルチモーダル (in English: Multimodal learning)（英語版）大規模言語モデル (in English: Large language model)である。2023年3月14日に公開された。自然言語処理 (in English: Natural language processing)にTransformer (機械学習モデル) (in English: Transformer)を採用しており、教師なし学習 (in English: Unsupervised learning)によって大規模なニューラルネットワーク (in English: Neural network)を学習させ、その後、人間のフィードバックからの強化学習 (in English: Reinforcement learning from human feedback)（RLHF）を行っている。",
            "Generative Pre-trained Transformer 4 (GPT-4) é um Modelo de linguagem grande (in English: Large language model) multimodal criado pela OpenAI (in English: OpenAI) e o quarto modelo da série GPT.  Foi lançado em 14 de março de 2023, e se tornou publicamente aberto de forma limitada por meio do ChatGPT (in English: ChatGPT) Plus, com o seu acesso à API comercial sendo provido por uma lista de espera. Sendo um Transformador (in English: Transformer), foi pré-treinado para prever o próximo Token (informática) (usando dados públicos e \"licenciados de provedores terceirizados\"), e então foi aperfeiçoado através de uma técnica de aprendizagem por reforço com humanos.",
            "GPT-4 es un modelo de Inteligencia artificial multimodal que puede generar texto a partir de diferentes tipos de entradas, como texto o imágenes. GPT-4 es un modelo multimodal porque puede Procesador de texto (in English: Word processor) y combinar diferentes modalidades de información, como el Lengua natural (in English: Natural language) y la Visión artificial (in English: Computer vision) Esto le da una ventaja sobre los modelos que solo pueden manejar una modalidad, ya que puede aprovechar el contexto y el conocimiento de múltiples fuentes.GPT-4 utiliza una técnica llamada fusión cruzada, que le permite integrar Información (in English: Information) de diferentes modalidades en una sola representación, lo que mejora su capacidad de Entendimiento (in English: Understanding) y generación"
        ],
        "title": [
            "GPT-4",
            "GPT-4",
            "Inteligencia artificial multimodal (in English: Multimodal artificial intelligence)"
        ],
        "full_section_title": [
            "GPT-4",
            "GPT-4",
            "Inteligencia artificial multimodal (in English: Multimodal artificial intelligence) > GPT-4"
        ],
        "block_type": [
            "text",
            "text",
            "text"
        ],
        "language": [
            "ja",
            "pt",
            "es"
        ],
        "last_edit_date": [
            "2024-03-01T14:03:10Z",
            "2024-01-10T18:38:38Z",
            "2024-03-28T22:44:02Z"
        ],
        "prob": [
            0.33441298757005206,
            0.3341872079017531,
            0.3313998045281948
        ]
    },
    {
        "score": [
            0.5762701034545898,
            0.5515922904014587,
            0.5366302728652954
        ],
        "text": [
            "LLaMA (Large Language Model Meta AI) é um grande modelo de linguagem (LLM) lançado pela Meta AI em fevereiro de 2023. Uma variedade de modelo foi treinada, variando de 7 bilhões a 65 bilhões. Os desenvolvedores do LLaMA relataram que o desempenho do modelo de 13 bilhões de parâmetros na maioria dos benchmarks NLP excedeu o do muito maior GPT-3 (in English: GPT-3) (com 175 bilhões de parâmetros) e que o maior modelo era competitivo com modelos de última geração, como PaLM e Chinchilla. Considerando que os LLMs mais poderosos geralmente são acessíveis apenas por meio de APIs limitadas (se é que existem), a Meta lançou os modelo do LLaMA para a comunidade de pesquisa sob uma licença não comercial. Uma semana após o lançamento do LLaMA, seus pesos vazaram para o público no 4chan via BitTorrent (in English: BitTorrent).",
            "LLaMA（Large Language Model Meta AI）は、Meta (企業) (in English: Meta Platforms) が2023年2月に発表した大規模言語モデル (in English: Large language model)。70億パラメータから650億パラメータまで、さまざまなサイズのモデルが学習された。LLaMA の開発者は、130億パラメータモデルがほとんどの自然言語処理 (in English: Natural language processing)ベンチマークにおいてGPT-3 (in English: GPT-3)（1750億パラメータ）の性能を上回ること、最大のモデルは PaLM (in English: PaLM) や Chinchilla などの最先端モデルに匹敵することを報告している。従来、ほとんどの強力な大規模言語モデルは限られた アプリケーションプログラミングインタフェース (in English: API) を通じてしかアクセスできなかったが、Meta は LLaMA のモデルのウェイトを非商用ライセンスで研究コミュニティに公開した。LLaMAのリリースから1週間で、そのウェイトがリークされた。",
            "Laminin subunit beta-3 is a protein that in humans is encoded by the LAMB3 gene. LAMB3 encodes the beta 3 subunit of laminin. Laminin is composed of three subunits (alpha, beta, and gamma), and refers to a family of basement membrane proteins. For example, LAMB3 serves as the beta chain in laminin-5. Mutations in LAMB3 have been identified as the cause of various types of epidermolysis bullosa. Two alternatively spliced transcript variants encoding the same protein have been found for this gene."
        ],
        "title": [
            "LLaMA",
            "LLaMA",
            "Laminin, beta 3"
        ],
        "full_section_title": [
            "LLaMA",
            "LLaMA",
            "Laminin, beta 3"
        ],
        "block_type": [
            "text",
            "text",
            "text"
        ],
        "language": [
            "pt",
            "ja",
            "en"
        ],
        "last_edit_date": [
            "2023-09-09T14:18:11Z",
            "2024-01-13T22:47:19Z",
            "2023-01-29T23:05:16Z"
        ],
        "prob": [
            0.34051134155877916,
            0.33221110342082927,
            0.32727755502039146
        ]
    }
]
```