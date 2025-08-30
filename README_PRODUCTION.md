# Production-Hardened vLLM FastAPI Service

## Overview

This enhanced vLLM service provides a production-ready, OpenAI-compatible API optimized for Tesla T4 GPUs with CUDA 12.4. The service includes comprehensive observability, security features, and performance optimizations specifically tuned for T4 architecture.

## Key Features

### 🚀 Core Capabilities
- **OpenAI API Compatibility**: Full `/v1/chat/completions` and `/v1/completions` endpoints
- **Streaming Support**: Server-sent events (SSE) for real-time token generation
- **Model Aliasing**: Automatic mapping of OpenAI model names to internal models
- **Function Calling**: Support for tool use and function calling (compatible models)
- **Token Counting**: Accurate token usage statistics with tiktoken integration

### 🔒 Security & Reliability
- **API Key Authentication**: Configurable multi-key authentication system
- **Rate Limiting**: Per-key and global rate limiting with configurable windows
- **Circuit Breaker**: Automatic failure detection and recovery
- **Graceful Shutdown**: Connection draining with configurable timeout
- **Request Queue Management**: Backpressure handling with queue size limits
- **CORS Support**: Configurable cross-origin resource sharing

### 📊 Observability
- **Structured JSON Logging**: Request tracing with correlation IDs
- **Prometheus Metrics**: Comprehensive metrics for monitoring
- **GPU Monitoring**: Real-time GPU utilization, memory, and temperature tracking
- **Request Tracing**: End-to-end request tracking with timing information
- **Health Endpoints**: Liveness and readiness probes with graduated checks

### ⚡ Performance Optimizations
- **T4 GPU Optimization**: SDPA attention backend for Tesla T4
- **Model Warmup**: Pre-loading and cache warming on startup
- **Connection Pooling**: Efficient resource management
- **CUDA Graph Support**: Optimized inference with graph capture
- **AWQ Quantization**: 4-bit quantization for efficient memory usage

## Quick Start

### Running Locally

1. **Using Docker (Recommended)**:
```bash
# Build the enhanced production image
make docker-build-enhanced

# Run the enhanced service
make docker-run-enhanced

# Check service health
make health-check-enhanced

# View metrics
make metrics-enhanced
```

2. **Direct Python Execution**:
```bash
# Install dependencies
pip install -r requirements-cuda124.txt

# Set environment variables
export VLLM_ATTENTION_BACKEND=SDPA  # Required for T4
export MODEL_PATH=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
export CUDA_VISIBLE_DEVICES=0

# Run the service
python -m app.main_enhanced
```

### Testing

```bash
# Run unit tests
make test-unit-enhanced

# Run all tests
make test

# Run load tests
make test-load

# Run security scans
make test-security
```

## API Endpoints

### OpenAI-Compatible Endpoints

#### Chat Completions
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": false
  }'
```

#### Streaming Chat Completions
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

#### Text Completions
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "text-davinci-003",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

#### List Models
```bash
curl http://localhost:8080/v1/models
```

### Health & Monitoring Endpoints

#### Health Check (Liveness)
```bash
curl http://localhost:8080/healthz
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "model_loaded": true,
  "active_requests": 2,
  "total_requests": 100,
  "gpu_memory_usage_gb": 8.5,
  "gpu_utilization_percent": 75,
  "gpu_temperature_celsius": 65,
  "circuit_breaker_state": "closed",
  "warmup_cache": {
    "enabled": true,
    "current_key": "mistral-7b-awq-sdpa",
    "is_cached": true,
    "total_cached_configs": 3,
    "cache_dir": "/cache/vllm"
  }
}
```

#### Prometheus Metrics
```bash
curl http://localhost:8080/metrics
```

Key metrics exposed:
- `vllm_request_total`: Total requests by status, endpoint, and model
- `vllm_request_duration_seconds`: Request latency histogram
- `vllm_active_requests`: Currently active requests gauge
- `vllm_token_throughput`: Tokens per second histogram
- `vllm_ttft_seconds`: Time to first token histogram
- `vllm_queue_size`: Current queue depth
- `vllm_gpu_memory_usage_gb`: GPU memory usage
- `vllm_gpu_utilization_percent`: GPU compute utilization
- `vllm_gpu_temperature_celsius`: GPU temperature
- `vllm_circuit_breaker_state`: Circuit breaker status (0=closed, 1=open)
- `vllm_concurrent_connections`: Active HTTP connections
- `vllm_warm_cache_hits_total`: Warmup cache hit counter
- `vllm_oom_events_total`: Out of memory events

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Hugging Face model path | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` |
| `VLLM_ATTENTION_BACKEND` | Attention backend (use SDPA for T4) | `AUTO` |
| `CUDA_VISIBLE_DEVICES` | GPU device IDs | `0` |
| `MAX_MODEL_LEN` | Maximum sequence length | `4096` |
| `GPU_MEMORY_UTILIZATION` | GPU memory fraction to use | `0.9` |
| `MAX_NUM_SEQS` | Maximum concurrent sequences | `32` |
| `UVICORN_WORKERS` | Number of Uvicorn workers | `1` |
| `API_KEYS` | Comma-separated API keys | `` (disabled) |
| `DEFAULT_RATE_LIMIT` | Default requests per minute | `100` |
| `RATE_LIMIT_WINDOW_MINUTES` | Rate limit window size | `1` |
| `MAX_QUEUE_SIZE` | Maximum request queue size | `50` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |
| `VLLM_CACHE_DIR` | vLLM cache directory | `/tmp/vllm_cache` |
| `HF_HOME` | Hugging Face cache directory | `~/.cache/huggingface` |
| `ENABLE_REQUEST_TRACING` | Enable request tracing | `true` |
| `READINESS_STRICT` | Strict readiness checks | `true` |

### API Key Configuration

1. **Single Key**:
```bash
export API_KEYS="sk-your-secret-key"
```

2. **Multiple Keys**:
```bash
export API_KEYS="sk-key1,sk-key2,sk-key3"
```

3. **Per-Key Rate Limits**:
```bash
export API_KEY_LIMITS="sk-key1:200,sk-key2:500,sk-key3:1000"
```

### T4 GPU Optimization

For Tesla T4 GPUs, ensure these settings:

```bash
# Required for T4 (no FlashAttention support)
export VLLM_ATTENTION_BACKEND=SDPA

# Optimal batch size for T4 (16GB memory)
export MAX_NUM_SEQS=32

# Memory utilization
export GPU_MEMORY_UTILIZATION=0.9

# Use AWQ quantization for efficiency
export QUANTIZATION=awq
```

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm-service:enhanced-latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: 8
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 4
        env:
        - name: VLLM_ATTENTION_BACKEND
          value: "SDPA"
        - name: MODEL_PATH
          value: "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 120
          periodSeconds: 10
        volumeMounts:
        - name: cache
          mountPath: /cache
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: vllm-cache
      nodeSelector:
        nvidia.com/gpu.product: Tesla-T4
```

### Service Monitor for Prometheus

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-metrics
spec:
  selector:
    matchLabels:
      app: vllm
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

## Production Considerations

### Resource Requirements

**Minimum Requirements**:
- GPU: NVIDIA Tesla T4 (16GB) or equivalent
- CPU: 8 cores
- RAM: 32GB
- Storage: 100GB for model cache

**Recommended**:
- GPU: NVIDIA Tesla T4 or better
- CPU: 16 cores
- RAM: 64GB
- Storage: 200GB SSD for model cache

### Monitoring & Alerting

**Key Metrics to Monitor**:
1. **Request Latency**: `vllm_request_duration_seconds` > 5s
2. **GPU Memory**: `vllm_gpu_memory_usage_gb` > 14GB (for T4)
3. **Queue Depth**: `vllm_queue_size` > 40
4. **Circuit Breaker**: `vllm_circuit_breaker_state` = 1 (open)
5. **OOM Events**: `vllm_oom_events_total` increasing

**Sample Prometheus Alert**:
```yaml
groups:
- name: vllm_alerts
  rules:
  - alert: HighGPUMemoryUsage
    expr: vllm_gpu_memory_usage_gb > 14
    for: 5m
    annotations:
      summary: "High GPU memory usage on {{ $labels.instance }}"
      
  - alert: CircuitBreakerOpen
    expr: vllm_circuit_breaker_state == 1
    for: 1m
    annotations:
      summary: "Circuit breaker is open on {{ $labels.instance }}"
      
  - alert: HighRequestLatency
    expr: histogram_quantile(0.95, vllm_request_duration_seconds_bucket) > 5
    for: 5m
    annotations:
      summary: "P95 latency > 5s on {{ $labels.instance }}"
```

### Security Best Practices

1. **Enable API Key Authentication**:
   - Always use API keys in production
   - Rotate keys regularly
   - Use different keys for different clients

2. **Network Security**:
   - Use TLS/HTTPS in production
   - Implement network policies in Kubernetes
   - Restrict ingress to known sources

3. **Resource Limits**:
   - Set appropriate rate limits
   - Configure queue size limits
   - Implement timeout policies

4. **Monitoring**:
   - Enable request tracing
   - Monitor for anomalous patterns
   - Set up alerting for security events

### Performance Tuning

1. **Batch Size Optimization**:
   ```bash
   # For T4 (16GB memory)
   export MAX_NUM_SEQS=32  # Start conservative
   # Monitor GPU memory and increase if possible
   ```

2. **Memory Management**:
   ```bash
   # Adjust based on model size
   export GPU_MEMORY_UTILIZATION=0.9  # Use 90% of GPU memory
   export SWAP_SPACE=4  # GB of CPU swap space
   ```

3. **Cache Configuration**:
   ```bash
   # Pre-download models
   export HF_HOME=/cache/huggingface
   export VLLM_CACHE_DIR=/cache/vllm
   # Mount persistent volume for cache
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `MAX_NUM_SEQS`
   - Lower `GPU_MEMORY_UTILIZATION`
   - Use smaller model or more aggressive quantization

2. **Slow First Request**:
   - Model loading takes time
   - Enable warmup cache
   - Use persistent volume for model cache

3. **Circuit Breaker Opens Frequently**:
   - Check GPU health
   - Review error logs
   - Adjust circuit breaker thresholds

4. **High Latency**:
   - Check queue depth
   - Monitor GPU utilization
   - Consider scaling horizontally

### Debug Endpoints

```bash
# Detailed runtime information
curl http://localhost:8080/debug/runtime

# Check warmup cache status
curl http://localhost:8080/healthz | jq '.warmup_cache'

# View active connections
curl http://localhost:8080/metrics | grep concurrent_connections
```

## License

This service is optimized for production use with vLLM and is designed for Tesla T4 GPUs. Ensure compliance with model licenses when deploying.

## Support

For issues or questions:
1. Check logs: `docker logs vllm-enhanced`
2. Review metrics: `curl http://localhost:8080/metrics`
3. Examine health: `curl http://localhost:8080/healthz`