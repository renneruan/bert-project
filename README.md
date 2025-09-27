# Instalação de Ambiente Virtual

No nó de login conseguimos criar o ambiente e ativá-lo, recomenda-se criar o caminho do venv no diretório $WORK, uma vez que ele facilmente excede o tamanho do $HOME pelo uso de bibliotecas pesadas como o pytorch.

```
python -m venv nome-do-env
source nome-do-env/bin/activate

# Atualizamos o pip e adicionamos o packaging
pip install --upgrade pip packaging setuptools wheel
```

Devemos ter em mente que cada ambiente deverá possuir o flash attention correspondente a versão do ROCM que ele se destina, é necessário então termos um ambiente para placas da família MI200 e outro para MI300.

Carregamos a partir dos módulos disponíveis no cluster HPC os caminhos referentes ao ROCM. No caso do mais recente temos:

```
module load rocm/6.4.1
```

Importante termos essa versão em mente, pois o PyTorch instalado deve corresponder a ela. Executamos o seguinte comando para instalá-lo.

```
pip install --pre --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
```

Agora podemos instalar as demais dependências necessárias, aqui caso o torch esteja listado ele irá verificar que já está instalado na versão desejada. O requirements disponibilizado é enxuto possuindo as bibliotecas gerais necessárias até agora.

```
pip install -r requirements.txt
```

## Instalando Flash Attention

Com o requirements anterior instalamos o ninja, pacote essencial para a próxima etapa, a instalação do flash attention. Para instalarmos o flash attention precisamos clonar o repositório específico para o ROCM, lembrando o flash attention irá verificar a arquitetura que estamos desejando no caso MI200 ou MI300, para instalar para uma arquitetura diferente, sugere-se fazer um novo clone em outra pasta.

A instalação do flash_attn consumirá bastante tempo por possuir uma compilação mais pesada, se fizermos isso no nó de login sem utilizar paralelismo demoraria em cerca de mais de 6 horas, recomenda-se agora alocarmos um nó de processamento para sua instalação.

A melhor forma de fazer isso é alocar um nó interativo no próprio terminal (Desaloca automaticamente se fecharmos). Fazemos isso utilizando por exemplo:

```
# Demandando uma máquina simples com mais tempo para garantir a instalação
salloc -N 1 -n 1 -p mi2104x -t 3:00:00
```

Ao entrarmos no nó (no terminal, o indicativo do usuário irá mudar de user@login1 para o identificador do nó), precisamos ativar o virtual environment novamente (criado na pasta $WORK) e executar a instalação do flash_attn. 

Antes podemos verificar se o ninja está instalado em nosso environment.

```
pip list
```

Com o ninja instalado agora vemos a quantia de núcleos de processador presente no nó alocado.

```
nproc
```

No caso da mi2104x provavelmente teremos 128 núcleos, recomendo não utilizar todos os núcleos pois irá resultar em erros de comunicação do nó com o gerenciador/log. Para o caso de 128, utilizei 96.


Aqui é essencial definirmos a qual arquitetura o flash attention irá se destinar. Devemos definir o `GPU_ARCHS` para família MI200 igual a `gfx90a` e para MI300 igual a `gfx942`.


```
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention

# Alterar o 96 de acordo com nproc disponíveis
# Alterar GPU_ARCHS de acordo com arquitetura
MAX_JOBS=96 GPU_ARCHS=gfx942 python setup.py install
```

Como estamos utilizando o ninja com 96 núcleos a instalação do flash attention se dará em aproximadamente 15 minutos. Com isso podemos apenas verificar se os pacotes essenciais estão instalados (transformers, flash attention e torch), garantindo que todos tenham a mesma versão de rocm.

## Alterações no job.notebook

Para que tenhamos os links corretos e não sobrepormos nosso torch destinado ao rocm, podemos comentar no job.notebook a linha `module load pytorch` e adicionar `module load rocm/6.4.1`.

## Registrar o venv como kernel

Após a instalação precisamos executar apenas uma vez o registro do venv como kernel do Jupyter, caso formos utilizar notebooks. Para isso precisamos ativar o venv e executarmos:

``` 
python -m ipykernel install --user --name="nome-do-env" --display-name="nome-de-display"
```

Com isso registramos novo environment aos kerneis possíveis.

Para levarmos este Kernel ao Jupyter server desejado podemos adicionar a ativação do environment direto no job.notebook. Abaixo da declaração dos modules podemos inserir:

```
echo "Activating virtual environment from $WORK..."
source $WORK/nome-env/bin/activate
```

Com isso podemos submeter o job.notebook. Realizamos então o o tunneling com SSH em outro terminal.

```
ssh -t user@hpcfund.amd.com -L 7080:localhost:{porta fornecida no job}
```

Isso irá providenciar um Jupyer server na porta 7080 em nossa máquina. Conseguimos acessar esse jupyter server enquanto o tunneling estiver ativo apertando em Select Kernels na parte direita superior do nosso notebook (já em nossa máquina).

Selecionamos a opção Existing Jupyter Server, com isso inserimos a url http://localhost:7080/tree?token={TOKEN} em que o Token é disponibilizado no arquivo gerado pela submissão do Job.

Conseguimos agora selecionamos o kernel registrado e utilizar os recursos.

Podemos verificar se estamos no nó desejado executando em uma célula:

## Garantindo que GPUs estão sendo utiliziadas

Caso executarmos o treino utilizando a chamada apenas como `python train.py` mesmo com o accelerator instalado ainda iremos utilizar apenas um núcleo de GPU, isso também irá acontecer em execução do Train disponibilizado pelo huggingface em notebooks.

Para utilizarmos o accelerate devemos executar o treino em um script Python, chamando:

```
accelerate launch --multi_gpu --num_processes=4 train_entrypoint.py
```

Dúvidas consultar: https://github.com/huggingface/accelerate

No exemplo temos 4 núcleos de GPU, para placas mais potentes podemos alterar este valor.

OBS: Isso deve ser realizado em um nó alocado com o ambiente virtual já ativo, uma vez que esta é a etapa de treino.

Para garantirmos que a GPU está sendo utilziada em sua capacidade máxima, podemos monitorar o nó que o treinamento está sendo executado, em outro terminal podemos entrar no nó utilizando

Consultamos o job id com

```
squeue -u $USER
```

Executamos no nó uma instância de bash

```
srun --jobid {JOB_ID} --pty bash
```

Nesta instância podemos utilizar os comandos:

```
rocm-smi
```

Sendo este para monitorar GPUs, recomenda-se executar diversas vezes, pois há a presença de picos que o uso vai a 100% e 0% em intervalos pequenos, com isso conseguimos analisar se o accelerate distribuiu para todos os núcleos.

Podemos monitorar o uso de CPU com o comando `top`

É essencial garantirmos que estejamos logados ao nó que o treino está acontecendo.