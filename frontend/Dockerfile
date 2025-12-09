# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

# Install dependencies first (Caching layer)
# Change: Use package*.json to match package.json OR package-lock.json
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Expose Vite Port
EXPOSE 5173

# Start Vite with --host to expose it to the Docker network
CMD ["npm", "run", "dev", "--", "--host"]