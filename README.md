Funk Generator
===================

Decidi criar este projeto quando estava aprendendo sobre como modelos de linguagem usando Deep
Learning funcionam. Escrevi mais detalhadamente sobre este projeto neste
[post](https://medium.com/@lucasmoura_35920/mc-neural-o-funkeiro-artificial-ab6fbedc9771) no Medium.
Além disso, fiz também uma descrição mais técnica do projeto em um
[post](http://lmoura.me/blog/2018/05/07/funk-generator/) no meu blog. Por fim, o programa está
rodando em uma [página](http://lmoura.me/funk_generator/) do meu blog. Lá você pode gerar músicas em
tempo real.

Neste documento, irei mostrar como fazer para que você consiga rodar o projeto do zero, ou até usar
outras músicas para treinar o seu modelo.

Dependências do projeto
---------

Antes de tudo, instale as dependências necessárias para executar este projeto:

```sh
$ pip install -r requirements.txt
```

Coleta das músicas
----------

A lista dos artistas que usei pode ser encontrada na pasta *data*, com o nome de *artist_list.txt*.
Esse arquivo é usado para fazer um crawler na API do [vagalume](https://api.vagalume.com.br/).

Para começar o crawler, execute o seguinte comando:

```sh
$ ./scripts/run_vagalume_crawler.sh
```

Este script irá passar por todos os artistas da lista e irá criar um diretório para cada artista na
pasta *data*. Dentro de cada diretório, será criado dos arquivos distintos:

* *song_codes.txt*: Um arquivo contendo o código de todas as músicas do artista. Esse código é usado
  para fazer o download da música em si, usando a API do vagalume.
* *song_names.txt*: Um arquivo contendo o nome das músicas do artista.

Uma vez que este script tenha sido executado, pode-se então executar o seguinte script:

```sh
$ ./scripts/download_musics.sh
```

Este script vai entrar em cada diretório que representa um artista e baixar todas as músicas
presentes no *song_codes.txt* daquele diretório. Cada música é armazenada como um arquivo txt.


Formatar os dados
------------------

Após o download das músicas, é necessário converter os arquivos de texto para um formato que o
modelo entenda. Para isso, rode o seguinte script:

```sh
$ ./scripts/run_format_data.sh
```

Esse script irá criar um pasta chamada *song_dataset* dentro da pasta *data*. Dentro dessa pasta,
terá os arquivos já processados para serem treinados pelo modelo.

*OBS: Nesse meu projeto eu criei quatro modelos diferentes para tipos diferentes de funk (Kondzilla,
Proibidão, Ostentção e todas as músicas) Entretanto, essa separação foi feita manualmente. Eu tive
que decidir quais músicas eram só de Ostentação, por exemplo. Para isso, olhei músicas que tinham
certos termos característicos e as agrupei uma pasta diferente. Ou seja, aqui essa separação não
será feita de forma automática. Se você quiser treinar os 4 modelos como eu fiz, terá que fazer esta
etapa manualmente.*

Treinamento do modelo
----------------------

Uma vez com os dados formatados, basta rodar o seguinte script:

```sh
$ ./script/run_model.sh
```

Após o treinamento ser concluído, será criado um diretório chamado *checkpoint* na raiz do projeto.
Esse diretório é o modelo treinado. Caso queria continuar treinando esse modelo gerado, altere a
variável *USE_CHECKPOINT* no script *run_model.sh*


Gerar Músicas
--------------

Para testar o modelo e gerar algumas músicas, rode o seguinte script:

```sh
./scripts/create_sample.sh
```

Esse script irá gerar uma música por vez.


Criar API
---------------------

Ao final, rode o seguinte comando para subir a API do projeto:

```sh
python app.py
```

O aplicação rodará com o servidor default do Flask e permite que você teste o modelo por requisições
POST. Lembre-se que se você só treinar um modelo, o mesmo só reconhecerá o modelo com id 1. Logo,
lembre-se de setar o id da requisição POST como 1 sempre. Por default, o programa será executado na
porta 5000.

Além disso, como em produção eu queria que a geração das músicas fosse o mais rápido possível, eu
gerei 4 mil músicas e armazenei elas dentro da minha aplicação (Essas músicas não estão presentes
nesse repositório). Sendo assim, a sua requisição POST não pode ter a variável *sentence* vazia,
pois caso ela esteja vazia, o modelo vai pegar aleatoriamente uma das músicas já armazenadas.

Dessa forma, o modelo só gera músicas em tempo real se o valor da variável *sentence* não for vazio.

Recomendo que se você queria usar esse modelo em produção como eu fiz, usar outra aplicação para
fazer o servidor, como o [gunicorn](http://gunicorn.org/).


Container
--------------

Caso você queira apenas usar a aplicação sem criar o modelo do zero, pode usar o container que
criei. Para isso é necessário ter o [Docker](https://www.docker.com/) instalado.

Uma vez com ele instalado, execute o seguinte comando para baixar a imagem do container:

```sh
$ docker pull lucasmoura/funk-generator
```

E para executar este container:

```sh
$ docker run -d -p 5000:5000 lucasmoura/funk-generator
```

Com o container rodando, basta seguir os passos descritos na seção *Criar API* para usar o programa.
Entretanto, o container tem uma vantagem. Nele existem todos os 4 modelos de funk que criei e também
nele está presente as 4 mil músicas que criei. Dessa forma, você não está restrito a sempre deixar a
variável *id* como 1, e também pode deixar a variável *sentence* vazia, caso queira recuperar
algumas das músicas já criadas.

Gere seu próprio modelo com suas próprias músicas
---------------

Para gerar seu próprio modelo, basta com que você mude o arquivo *artist_list.txt* para conter os
artistas que você quiser e depois é só seguir todos os passos já listados.

Caso você já tenhas as músicas baixadas, garanta que cada artista tem um diretório próprio e que
todas as músicas desse artista estejam no diretório que o representa. Uma vez isso pronto, basta
continuar à partir da seção *Formatar dados*
