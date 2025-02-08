# Projeto - Previsão da Crise de Ansiedade


# Telemarketing analisys

Neste projeto são analisados os dados dos clientes de um banco fictício que deseja oferecer um novo plano a eles. Ou seja, está sendo analisada a proporção dos clientes que aceitam
(ou não) a oferta de acordo com suas faixas etárias, seus níveis de educação, emprego, entre outras características.

## Descrição do projeto

As características (variáveis explicativas) dos clientes analisadas neste projeto são:
* age: idade
* job: emprego
* marital: estado civil
* education: nível educacional
* default: se a pessoa é devedora ou não
* housing: posse de imóvel
* loan: empréstimo
* contact: meio de contato
* month: mês em que o cliente entrou em contato com o banco
* day_of_week: dia da semana em que houve o contato
* duration: duração da ligação
* y (variável resposta): informa se o cliente aceitou ou não a oferta

### Link para a aplicação (LIVE)
https://telemarketing-ebac-wzu2.onrender.com/

## Utilização

### Dependencias

* O código deste projeto pode ser lido e executado em qualquer sistema operacional, desde que tenha uma editor de códigos que reconheça a linguagem python
  (tal como Visual Studio Code), um software capaz de executar o código no formato py (por exemplo o Git Bash) e um leitor de planilhas no formato csv
  (pode utilizar o libre office).
* Os pacotes necessários para rodar o código são:
- pandas==2.2.0
- seaborn==0.13.1
- streamlit==1.30.0
- XlsxWriter==3.1.9
- matplotlib==3.8.2
- protobuf==4.25.2
o que pode ser confirmado no arquivo requirements.txt.

### Instalação

* Os arquivos necessários (código e planilha) podem ser obtidos neste repositório (Marketing_EBAC), sendo encontrados logo acima deste arquivo README.
* Para visualizar o código e os dados utilizados, faça o download dos arquivos "app_7.py" e "bank-additional.csv".

### Executando o projeto

* Caso queira apenas visualizar o site desenvolvido neste projeto, acesse o link presente na seção "Link para a aplicação (LIVE)" logo acima.
* Por outro lado, para editar o código abra o arquivo "app_7.py" pelo editor presente no seu SO.
* Em relação aos dados, caso queira alterá-los, abra o arquivo "bank-additional.csv" pelo office do seu SO.
* Para acessar o site desenvolvido pelo código diretamente, abra o terminal do seu sistema (Git Bash, CMD do Windows ou Conda)
  e digite os seguintes comandos:
  
```
cd [NOME DA PASTA NA QUAL O CÓDIGO FOI SALVO]
python -m streamlit run app_7.py
```

* Ao entrar no site, é necessário carregar os dados do banco. Isto pode ser feito de duas formas:
  1. Arrastando o arquivo csv para a área "Drag and drop files here";
  2. Clicando em "Browse files" e selecione a planilha csv na pasta em que ela foi salva.  

* Com o carregamento dos dados, aparecerá na tela principal duas tabelas, sendo uma com os dados originais e a outra
* com os dados filtrados. Abaixo delas encontram-se as proporções de pessoas que aceitaram (ou não) a oferta do banco
* em cada conjunto de dados, e os gráficos das proporções.
* A filtragem dos dados é feita atraavés dos filtros abaixo da área "Drag and drop files here", sendo eles:
  - O tipo de gráficos desejado, podendo ser "barras" ou "pizza";
  - Intervalo das idades dos clientes, sendo regulado pelo scrollbar "idade";
  - Seleção das características mencionadas em "Descrição do Projeto", podendo ser todas em cada uma (padrão),
    ou uma única resposta ou conjunto delas para cada característica dos clientes.
* Todas as tabelas mencionadas acima podem ser baixada por meio dos botões "download" presentes abaixo de cada uma.

## Ajuda

Caso restem dúvidas sobre o funcionamento do projeto, entre em contato com o autor. 

## Autores

Nomes dos desenvolvedores do projeto e informação para entrar em contato.

Lucas Ribeiro
[@LucasRibeiro](www.linkedin.com/in/lucas-data-science)

## Histórico de versões.

* 0.2
	* Ajustes de diversos bugs e otimização
* 0.1
    * Primeira versão

## Licença de uso

Esse projeto possui licença de uso - acesse o arquivo LICENSE.md para mais detalhes.

## Fontes de inspiração

Inspiração, trechos de códigos utilizados, etc.
* [readme-template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)

