# First build images

> `docker-compose -f docker/docker-compose.yaml build`

# Exécuter l'environnement de développement

> `docker-compose -f docker/docker-compose.yaml --env-file .env.dev up`

# Exécuter l'environnement de staging

> `docker-compose -f docker/docker-compose.yaml --env-file .env.staging up`

# Exécuter l'environnement de production

> `docker-compose -f docker/docker-compose.yaml --env-file .env.prod up`
