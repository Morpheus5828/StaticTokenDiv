# Static Token Div

## Installation

Before compiling the report, you need to set up a Python environment and install the necessary dependencies.
```shell
conda create --name static_token_div python=3.10
```

```shell
conda activate static_token_div
```

```shell
pip install -r requirements.txt
```

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Demos files

```shell
python .\demos\demo_create_embedding.py
```

```shell
python .\demos\demos_word2vec.py
```
To use analogy demo, you need to install folder number 43 in this url <a href="http://vectors.nlpl.eu/repository" target="_blanck">Click here</a> and store it into <br>
resources folder, extract zip file and rename it model.txt
```shell
python .\demos\analogy\demos_analogy.py
```

## Authors

- Marius THORRE (Owner)
- Thomas CELESCHI (Follower)
