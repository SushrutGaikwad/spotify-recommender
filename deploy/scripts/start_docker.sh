#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 872515288060.dkr.ecr.us-east-2.amazonaws.com

echo "Pulling Docker image..."
docker pull 872515288060.dkr.ecr.us-east-2.amazonaws.com/spotify_recommender:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=spotify_recommender)" ]; then
    echo "Stopping existing container..."
    docker stop spotify_recommender
fi

if [ "$(docker ps -aq -f name=spotify_recommender)" ]; then
    echo "Removing existing container..."
    docker rm spotify_recommender
fi

echo "Starting new container..."
docker run -d -p 8000:8000 --name spotify_recommender 872515288060.dkr.ecr.us-east-2.amazonaws.com/spotify_recommender:latest

echo "Container started successfully."
