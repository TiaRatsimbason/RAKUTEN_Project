# ⚠️ TODO
- [ ] Il faudra mettre à jour toutes les partie `api` afin que les docker-compose fonctionnent et lancent l'application
- [ ] Faire les fichiers `Dockerfile.api`et `Dockerfile.test`

# Exécuter l'environnement de développement

`docker-compose -f docker/docker-compose/docker-compose.dev.yaml up --build` <-- lancer le docker-compose de l'environnement de développement

# Exécuter l'environnement de staging

`docker-compose -f docker/docker-compose/docker-compose.staging.yaml up --build` <-- lancer le docker-compose de l'environnement de staging


# Exécuter l'environnement de test

`docker-compose -f docker/docker-compose/docker-compose.testapp.yaml up --build` <-- lancer le docker-compose de l'environnement de test

**le nom du fichier est *.testapp.yaml car docker-compose.test.yaml file is using prometheus.rules.test.json first (voir: [issue #949 vscode-yaml](https://github.com/redhat-developer/vscode-yaml/issues/949))**


# Exécuter l'environnement de production

`docker-compose -f docker/docker-compose/docker-compose.prod.yaml up --build` <-- lancer le docker-compose de l'environnement de production
