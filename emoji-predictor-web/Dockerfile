# Stage 1: Install dependencies and build the application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package.json and lock file
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the application code
COPY . .

# Build the Next.js application
# Pass empty NEXT_PUBLIC_API_URL and NEXT_PUBLIC_WS_URL during build
# They will be set at runtime by Docker Compose / Coolify
ENV NEXT_PUBLIC_API_URL="/api" 
ENV NEXT_PUBLIC_WS_URL="/ws"
RUN npm run build

# Stage 2: Create the production image
FROM node:20-alpine

WORKDIR /app

# Set environment to production
ENV NODE_ENV=production

# Copy built artifacts from the builder stage
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./ 
COPY --from=builder /app/.next/static ./.next/static

# Expose port 3000
EXPOSE 3000

# Define environment variable (optional, good practice)
ENV NAME World

# Command to run the application
# Use the standalone output which includes the Node.js server
CMD ["node", "server.js"] 