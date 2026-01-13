FROM node:20-slim
WORKDIR /app

COPY package.json package-lock.json* /app/
RUN npm install

COPY . /app

ENV EXPO_PUBLIC_API_BASE=http://backend:8000

EXPOSE 8081 19000 19001 19002 19006

ENV REACT_NATIVE_PACKAGER_HOSTNAME=0.0.0.0

CMD ["npx", "expo", "start", "--web", "--port", "19006"]
