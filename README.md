# Image Classification by Relevance of Regions Using Machine Learning


## Setup
Para clonar e executar o projeto, siga as seguintes instruções

### Opção 1: Setup tradicional com pip

```bash
# Clone o projeto
git clone git@github.com:afmireski/tcc_image_classification_by_relevance_of_regions.git 
## or
git clone https://github.com/afmireski/tcc_image_classification_by_relevance_of_regions.git

# Configure a venv do projeto
python -m venv .venv    

# Ative a venv
source ./.venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Selecione a versão do python da venv como kernel do main.ipynb
```

### Opção 2: Setup com Python uv (Recomendado)

```bash
# Clone o projeto
git clone git@github.com:afmireski/tcc_image_classification_by_relevance_of_regions.git 
## or
git clone https://github.com/afmireski/tcc_image_classification_by_relevance_of_regions.git

# Entre no diretório do projeto
cd tcc_image_classification_by_relevance_of_regions

# Sincronize as dependências (isso criará automaticamente o ambiente virtual)
uv sync

# Ative o ambiente virtual
source .venv/bin/activate

# Selecione a versão do python da venv como kernel do main.ipynb
```

> **Nota:** O Python uv é uma ferramenta moderna e mais rápida para gerenciamento de dependências Python. Para instalá-lo, visite: https://docs.astral.sh/uv/getting-started/installation/