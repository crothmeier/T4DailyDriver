# vLLM Service for Tesla T4

Production-ready vLLM service optimized for Tesla T4 GPUs with Mistral-7B AWQ quantization, connection pooling, Prometheus metrics, and health checks.

## Features

- **Optimized for Tesla T4**: Configured for 16GB VRAM with AWQ quantization
- **Connection Pooling**: Efficient request handling with metrics
- **Prometheus Metrics**: Comprehensive monitoring including TTFT, token throughput, and GPU usage
- **Health Checks**: Liveness and readiness probes for production reliability
- **Kubernetes Ready**: Full K8s manifests with HPA, monitoring, and ingress
- **Docker Compose**: Local development and testing environment

## Quick Start

### Local Development (Docker Compose)

```bash
# Clone the repository
git clone <repository-url>
cd T4DailyDriver

# Build and start services
docker-compose up -d

# Check service health
curl http://localhost:8080/health

# Test generation
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Production Deployment (Kubernetes)

```bash
# Apply all manifests
kubectl apply -k k8s/

# Or apply individually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# Check deployment status
kubectl -n vllm get pods
kubectl -n vllm get svc

# Port forward for testing
kubectl -n vllm port-forward svc/vllm-service 8080:80
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key configurations for T4 optimization:
- `GPU_MEMORY_UTILIZATION=0.9` - Use 90% of T4's 16GB VRAM
- `MAX_NUM_SEQS=32` - Optimal batch size for T4
- `QUANTIZATION=awq` - 4-bit quantization for Mistral-7B

### Model Selection

Default model: `TheBloke/Mistral-7B-Instruct-v0.2-AWQ`

To use a different model, update:
1. `MODEL_PATH` environment variable
2. ConfigMap in Kubernetes
3. Docker Compose environment

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check with GPU metrics
- `GET /metrics` - Prometheus metrics
- `POST /generate` - Text generation (streaming supported)

### Generation Request

```json
{
  "prompt": "Your prompt here",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "stream": false,
  "stop": ["\\n"],
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0
}
```

## Monitoring

### Prometheus Metrics

- `vllm_request_count` - Total requests by status
- `vllm_request_duration_seconds` - Request duration histogram
- `vllm_active_requests` - Currently active requests
- `vllm_token_throughput` - Tokens per second histogram
- `vllm_time_to_first_token_seconds` - TTFT histogram
- `vllm_gpu_memory_usage_gb` - GPU memory usage

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

## Performance Targets

On Tesla T4 with Mistral-7B AWQ:
- **Throughput**: 50 tokens/second
- **TTFT p95**: < 200ms
- **Max concurrent requests**: 32
- **GPU utilization**: > 70%

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `GPU_MEMORY_UTILIZATION`
   - Decrease `MAX_NUM_SEQS`
   - Use smaller model or more aggressive quantization

2. **Slow Performance**
   - Check GPU utilization with `nvidia-smi`
   - Monitor metrics at `/metrics` endpoint
   - Verify CUDA graphs are enabled

3. **Model Loading Fails**
   - Ensure sufficient disk space for model cache
   - Check network connectivity to Hugging Face
   - Verify CUDA/driver compatibility

## Development

### Building Docker Image

```bash
docker build -t vllm-service:latest .
```

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## License

[Your License Here]
