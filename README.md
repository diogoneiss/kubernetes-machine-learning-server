## Entregáveis
#### The ML code responsible for generating association rules and a version of your model:
[./machine-learning](./machine-learning)

#### Code responsible for running the Web front-end

[./rest_api](./rest_api)

#### The client application, scripts, or Web front-end in charge of demonstrating access to your REST API.
Mapeamento de portas e rotas (se estiver fora do servidor será necessário um bind de portas para acessar o serviço)

* Página html para testes: http://localhost:31000/
* Frontend com swagger da api: http://localhost:31000/docs
* Endpoint POST da api: http://localhost:31000/api/recommend/

#### The Dockerfile to build your containers in charge of running the REST API and the other responsible to run the model generation, together with any additional code required to build your container images.

Tudo está nas pastas de código, porém aqui estão os Dockerfiles.

* [./rest_api/Dockerfile](./rest_api/Dockerfile)
* [./machine-learning](./machine-learning/Dockerfile)

#### The YAML files describing the Kubernetes deployment and service.

* [./kubernetes/deployment.yaml](./kubernetes/deployment.yaml)
* [./kubernetes/service.yaml](./kubernetes/service.yaml)
* [./kubernetes/pvc.yaml](./kubernetes/pvc.yaml)
* [./kubernetes/job.yaml](./kubernetes/job.yaml)

#### The YAML file describing the ArgoCD application. This file is called the "Manifest" in the Web interface. This can also be exported using the spec field/property in the output of argocd app get [appname] -o yaml.

[./argocd_manifest](./argocd_manifest.yaml)


## Arquitetura

Foi desenvolvido um serviço FastAPI vinculado ao PVC. Um arquivo é utilizado para polling de mudanças, permitindo verificar se as regras devem ser recarregadas periodicamente.

Para geração das regras foi feito um job simples, executado periodicamente. Cada execução usa um dataset diferente do utilizado na última vez, gera os arquivos necessários e grava no PVC.

## Playlist Rules Generator - Machine learning Job

Script python capaz de gerar as regras utilizando o algoritmo **FpGrowth**. Os datasets estão no repositório para facilitar a execução do job localmente, porém fora do [diretório do serviço.](./machine-learning)

### Datasets

O container **NÃO** possui os datasets dentro, eles estão fora do diretório do serviço. Para rodar no kubernetes, certifique-se que eles estão dentro do PVC e ajuste o deployment para apontar para o diretório.

### Execução

Tudo que o serviço precisa é das variáveis de ambiente apontando para diretórios válidos, o resto será gerado.

O job executa apenas uma vez, e é removido apos o tempo especificado no TTL. Como o ArgoCD foi configurado para sincronizar esse job com replace, ele será reexecutado, se tornando então um cron job.

## Playlist Recommendation API - REST API

Serviço FastAPI que recebe um JSON com uma lista de músicas e retorna uma lista de recomendações baseadas nas regras geradas anteriormente pelo job.

O serviço depende das regras geradas dentro do PVC e corretamente mapeadas nas variáveis de ambiente.

Na raiz do serviço há um arquivo ```index.html``` que pode ser utilizado para testar o serviço, acessível pelo navegador em ```/```.

Há também uma documentação OpenAPI no endpoint ```/docs```