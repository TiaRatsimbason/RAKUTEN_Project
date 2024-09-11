# ⚠️ TODO
- [ ] Il faudra mettre à jour toutes les partie `api` afin que les docker-compose fonctionnent et lancent l'application
- [ ] Faire les fichiers `Dockerfile.api`et `Dockerfile.test`

# Exécuter l'environnement de développement

`cd docker/docker-compose` <-- se déplacer dans le dossier docker-compose

`docker-compose -f docker-compose.dev.yaml up --build` <-- lancer le docker-compose de l'environnement de développement

# Exécuter l'environnement de staging

`cd docker/docker-compose` <-- se déplacer dans le dossier docker-compose

`docker-compose -f docker-compose.staging.yaml up --build` <-- lancer le docker-compose de l'environnement de staging


# Exécuter l'environnement de test

`cd docker/docker-compose` <-- se déplacer dans le dossier docker-compose

`docker-compose -f docker-compose.testapp.yaml up --build` <-- lancer le docker-compose de l'environnement de test

**le nom du fichier est *.testapp.yaml car docker-compose.test.yaml file is using prometheus.rules.test.json first (voir: [issue #949 vscode-yaml](https://github.com/redhat-developer/vscode-yaml/issues/949))**


# Exécuter l'environnement de production

`cd docker/docker-compose` <-- se déplacer dans le dossier docker-compose

`docker-compose -f docker-compose.prod.yaml up --build` <-- lancer le docker-compose de l'environnement de production
