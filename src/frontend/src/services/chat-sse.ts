export type ChatMessageRole = 'user' | 'assistant';

export interface ChatResponseMetrics {
  readonly ttftMs?: number;
  readonly generationMs?: number;
  readonly generationThroughputTokensPerSecond?: number;
  readonly outputTokens?: number;
  readonly outputTokenSource: 'usage' | 'chunks';
}

// ================================================================
// SSE

interface SSEOptions {
  url: string;
  onOpen?: () => void;
  onClose?: (metrics?: ChatResponseMetrics) => void;
  onError?: (error: Error, metrics?: ChatResponseMetrics) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onMessage?: (message: { event: string; id?: string; data: any }) => void;
}

interface SSEMetricsState {
  requestStartAt: number;
  firstTokenAt?: number;
  endAt?: number;
  outputChunks: number;
  usageCompletionTokens?: number;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object';

interface RequestMessage {
  readonly id: string;
  readonly role: ChatMessageRole;
  readonly content: string;
}

export const createSSE = (options: SSEOptions) => {
  const { url, onOpen, onClose, onError, onMessage } = options;

  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let abortController: AbortController | undefined;
  let metrics: SSEMetricsState | undefined;
  let closed = true;

  const resetMetrics = () => {
    metrics = {
      requestStartAt: performance.now(),
      outputChunks: 0,
    };
  };

  const finishMetrics = () => {
    if (!metrics?.endAt) {
      metrics = {
        ...(metrics || { requestStartAt: performance.now(), outputChunks: 0 }),
        endAt: performance.now(),
      };
    }
  };

  const trackMetrics = (data: unknown) => {
    if (!metrics || !isRecord(data)) {
      return;
    }

    const usageTokens = isRecord(data.usage) ? data.usage.completion_tokens : undefined;
    if (typeof usageTokens === 'number' && Number.isFinite(usageTokens)) {
      metrics.usageCompletionTokens = Math.max(metrics.usageCompletionTokens || 0, usageTokens);
    }

    const choices = Array.isArray(data.choices) ? data.choices : [];
    const contentfulChoices = choices.filter((choice) => {
      if (!isRecord(choice) || !isRecord(choice.delta)) {
        return false;
      }
      const content = choice.delta.content;
      const reasoning = choice.delta.reasoning;
      return (
        (typeof content === 'string' && content.length > 0)
        || (typeof reasoning === 'string' && reasoning.length > 0)
      );
    }).length;

    if (contentfulChoices > 0) {
      metrics.outputChunks += contentfulChoices;
      metrics.firstTokenAt = metrics.firstTokenAt ?? performance.now();
    }
  };

  const getMetrics = (): ChatResponseMetrics | undefined => {
    finishMetrics();
    if (!metrics) {
      return undefined;
    }

    const outputTokens = metrics.usageCompletionTokens || metrics.outputChunks || undefined;
    const outputTokenSource = metrics.usageCompletionTokens ? 'usage' : 'chunks';
    const ttftMs =
      metrics.firstTokenAt === undefined ?
        undefined
      : metrics.firstTokenAt - metrics.requestStartAt;
    const generationMs =
      metrics.firstTokenAt === undefined || metrics.endAt === undefined ?
        undefined
      : metrics.endAt - metrics.firstTokenAt;
    const generationThroughputTokensPerSecond =
      outputTokens === undefined || generationMs === undefined || generationMs <= 0 ?
        undefined
      : outputTokens / (generationMs / 1000);

    return {
      ttftMs,
      generationMs,
      generationThroughputTokensPerSecond,
      outputTokens,
      outputTokenSource,
    };
  };

  const closeOnce = () => {
    if (closed) {
      return;
    }
    closed = true;
    onClose?.(getMetrics());
  };

  const errorOnce = (error: Error) => {
    finishMetrics();
    if (!closed) {
      closed = true;
    }
    onError?.(error, getMetrics());
  };

  const connect = (model: string, messages: readonly RequestMessage[]) => {
    const decoder = new TextDecoder();
    closed = false;
    resetMetrics();
    abortController = new AbortController();

    onOpen?.();

    fetch(url, {
      method: 'POST',
      body: JSON.stringify({
        stream: true,
        model,
        messages,
        max_tokens: 2048,
        top_k: 3,
      }),
      signal: abortController.signal,
    })
      .then(async (response) => {
        const statusCode = response.status;
        const contentType = response.headers.get('Content-Type');
        if (statusCode !== 200) {
          errorOnce(new Error(`[SSE] Failed to connect: ${statusCode}`));
          return;
        }
        if (!contentType?.includes('text/event-stream')) {
          errorOnce(new Error(`[SSE] Invalid content type: ${contentType}`));
          return;
        }

        reader = response.body?.getReader();
        if (!reader) {
          errorOnce(new Error(`[SSE] Failed to get reader`));
          return;
        }

        let buffer = '';

        const processLines = (lines: string[]) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const message: { event: string; id?: string; data: any } = {
            event: 'message',
            data: undefined,
          };
          lines.forEach((line) => {
            const colonIndex = line.indexOf(':');
            if (colonIndex <= 0) {
              // No colon, skip
              return;
            }

            const field = line.slice(0, colonIndex).trim();
            const value = line.slice(colonIndex + 1).trim();

            if (value.startsWith(':')) {
              // Comment line
              return;
            }

            switch (field) {
              case 'event':
                message.event = value;
                break;
              case 'id':
                message.id = value;
                break;
              case 'data':
                try {
                  // Try to parse as JSON object
                  const data = JSON.parse(value);
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const walk = (data: any) => {
                    if (!data) {
                      return;
                    }
                    if (Array.isArray(data)) {
                      data.forEach((item, i) => {
                        if (item === null) {
                          data[i] = undefined;
                        } else {
                          walk(item);
                        }
                      });
                    } else if (typeof data === 'object') {
                      Object.keys(data).forEach((key) => {
                        if (data[key] === null) {
                          delete data[key];
                        } else {
                          walk(data[key]);
                        }
                      });
                    }
                  };
                  walk(data);
                  message.data = data;
                  trackMetrics(data);
                } catch (error) {
                  // Parse failed, use original data
                  message.data = value;
                }
                break;
            }

            if (message.data !== undefined) {
              onMessage?.(message);
            }
          });
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            closeOnce();
            return;
          }

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          processLines(lines);
        }
      })
      .catch((error: Error) => {
        if (error instanceof Error && error.name === 'AbortError') {
          closeOnce();
          return;
        }
        errorOnce(error);
      });
  };

  const disconnect = () => {
    reader?.cancel();
    reader = undefined;
    abortController?.abort('stop');
    abortController = undefined;

    closeOnce();
  };

  return { connect, disconnect };
};
