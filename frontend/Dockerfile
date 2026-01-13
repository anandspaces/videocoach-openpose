FROM node:24-alpine AS builder

# Install pnpm globally
RUN npm install -g pnpm

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml* ./

# Install dependencies using pnpm
RUN pnpm install --frozen-lockfile

# Copy source code
COPY . .

# Build the application
RUN pnpm run build

FROM node:24-alpine AS production

# Install serve globally
RUN npm install -g serve

# Set working directory
WORKDIR /app

# Copy built assets from builder stage
COPY --from=builder /app/dist ./dist

# Expose port 4005
EXPOSE 4005

# Serve the built files
CMD ["serve", "-s", "dist", "-l", "4005"]