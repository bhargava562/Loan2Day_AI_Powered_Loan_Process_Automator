# Loan2Day Frontend Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM node:18-alpine as builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine as production

# Install curl for health checks
RUN apk add --no-cache curl

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user for security
RUN addgroup -g 1001 -S loan2day && \
    adduser -S loan2day -u 1001

# Set proper permissions
RUN chown -R loan2day:loan2day /usr/share/nginx/html && \
    chown -R loan2day:loan2day /var/cache/nginx && \
    chown -R loan2day:loan2day /var/log/nginx && \
    chown -R loan2day:loan2day /etc/nginx/conf.d

# Switch to non-root user
USER loan2day

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]