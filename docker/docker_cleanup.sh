#!/bin/bash

echo "Pruning all unused Docker resources..."


echo "Stopping all containers..."
docker stop $(docker ps -aq)


echo "Removing all containers..."
docker rm $(docker ps -aq)


echo "Removing all images..."
docker rmi $(docker images -q) --force


echo "Removing all volumes..."
docker volume rm $(docker volume ls -q) --force


echo "Removing all unused networks..."
docker network prune -f


echo "Final general prune..."
docker system prune -a --volumes -f

echo "Docker cleanup completed!"
