FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json /app/
RUN npm ci

COPY . /app

ENV EXPO_PUBLIC_API_BASE=http://backend:8000

# Expo / Metro / Web ports
EXPOSE 8081 19000 19001 19002 19006

# Dev server for docker-compose (full dev stack)
CMD ["npm", "run", "start", "--", "--host", "0.0.0.0"]

