## Arquitetura

Foi desenvolvido um serviço FastAPI vinculado ao PVC. Um arquivo é utilizado para polling de mudanças, permitindo verificar se as regras devem ser recarregadas periodicamente.

Para geração das regras foi feito um job simples, executado periodicamente. Cada execução usa um dataset diferente do utilizado na última vez, gera os arquivos necessários e grava no PVC.

## Playlist Rules Generator - Machine learning Job

Script python capaz de gerar as regras. Coloquei os datasets no git para facilitar a execução do job localmente.

### Datasets

O container **NÃO** possui os datasets dentro, eles estão fora do diretório do serviço. Para rodar no kubernetes, certifique-se que eles estão dentro do PVC e ajuste o deployment para apontar para o diretório.

### Execução

Tudo que o serviço precisa é das variáveis de ambiente apontando para diretórios válidos, todo o resto será gerado.

O job executa apenas uma vez, e é removido apos o tempo especificado no TTL. Como o ArgoCD foi configurado para sincronizar esse job com replace, ele será reexecutado, se tornando então um cron job.