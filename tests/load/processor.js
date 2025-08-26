module.exports = {
  logMetrics: function(requestParams, response, context, ee, next) {
    const latency = response.timings.phases.firstByte;
    const totalTime = response.timings.phases.total;
    const responseSize = JSON.stringify(response.body).length;

    console.log(`[COMPLETION] Latency: ${latency}ms, Total: ${totalTime}ms, Size: ${responseSize} bytes`);

    if (response.body && response.body.usage) {
      const usage = response.body.usage;
      console.log(`[TOKEN_USAGE] Prompt: ${usage.prompt_tokens}, Completion: ${usage.completion_tokens}, Total: ${usage.total_tokens}`);

      ee.emit('counter', 'tokens.prompt', usage.prompt_tokens);
      ee.emit('counter', 'tokens.completion', usage.completion_tokens);
      ee.emit('counter', 'tokens.total', usage.total_tokens);
    }

    ee.emit('histogram', 'response.size', responseSize);
    ee.emit('histogram', 'response.latency', latency);

    if (latency > 5000) {
      ee.emit('counter', 'slow_requests', 1);
      console.warn(`[WARNING] Slow request detected: ${latency}ms`);
    }

    return next();
  },

  logStreamingMetrics: function(requestParams, response, context, ee, next) {
    const timeToFirstByte = response.timings.phases.firstByte;
    const totalTime = response.timings.phases.total;

    console.log(`[STREAMING] Time to first byte: ${timeToFirstByte}ms, Total: ${totalTime}ms`);

    ee.emit('histogram', 'streaming.ttfb', timeToFirstByte);
    ee.emit('histogram', 'streaming.total', totalTime);

    if (timeToFirstByte > 2000) {
      ee.emit('counter', 'slow_streaming_start', 1);
      console.warn(`[WARNING] Slow streaming start: ${timeToFirstByte}ms`);
    }

    return next();
  },

  logBatchMetrics: function(requestParams, response, context, ee, next) {
    const latency = response.timings.phases.total;
    const batchSize = response.body.completions ? response.body.completions.length : 0;

    console.log(`[BATCH] Processed ${batchSize} requests in ${latency}ms`);

    ee.emit('histogram', 'batch.latency', latency);
    ee.emit('histogram', 'batch.size', batchSize);

    if (batchSize > 0) {
      const avgLatency = latency / batchSize;
      ee.emit('histogram', 'batch.avg_latency', avgLatency);
      console.log(`[BATCH] Average latency per request: ${avgLatency.toFixed(2)}ms`);
    }

    return next();
  },

  logSystemMetrics: function(requestParams, response, context, ee, next) {
    if (response.body) {
      const gpuMemory = response.body.gpu_memory_usage;
      const activeRequests = response.body.active_requests;

      console.log(`[SYSTEM] GPU Memory: ${gpuMemory}MB, Active Requests: ${activeRequests}`);

      ee.emit('histogram', 'system.gpu_memory', gpuMemory);
      ee.emit('histogram', 'system.active_requests', activeRequests);

      if (gpuMemory > 14000) {
        ee.emit('counter', 'high_memory_usage', 1);
        console.warn(`[WARNING] High GPU memory usage: ${gpuMemory}MB`);
      }

      if (activeRequests > 50) {
        ee.emit('counter', 'high_active_requests', 1);
        console.warn(`[WARNING] High number of active requests: ${activeRequests}`);
      }
    }

    return next();
  },

  beforeRequest: function(requestParams, context, ee, next) {
    requestParams.headers = requestParams.headers || {};
    requestParams.headers['X-Request-ID'] = `artillery-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    requestParams.headers['X-Test-Scenario'] = context.scenario ? context.scenario.name : 'unknown';

    return next();
  },

  afterResponse: function(requestParams, response, context, ee, next) {
    if (response.statusCode >= 400) {
      console.error(`[ERROR] Request failed with status ${response.statusCode}: ${JSON.stringify(response.body)}`);
      ee.emit('counter', `errors.${response.statusCode}`, 1);
    }

    if (response.statusCode === 429) {
      console.warn('[RATE_LIMIT] Rate limit hit, backing off...');
      context.think = 5;
    }

    return next();
  }
};
