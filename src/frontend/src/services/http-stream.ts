import type { HttpCommonOptions } from './http-common';

export type HttpStreamClientStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface HttpStreamClientOptions {
  debugName?: string;

  /**
   * Whether the client can re-connect automatically when close or error.
   * @default false
   */
  autoReconnect?: boolean;

  /**
   * The interval time (ms) of auto-reconnect.
   * @default 2000
   */
  autoReconnectInterval?: number;

  /**
   * The callback function to handle the connection status change.
   * @param status The status of the client.
   */
  onStatusChange?: (status: HttpStreamClientStatus) => void;

  /**
   * The callback function to handle the message.
   * @param message The message data.
   */
  onMessage?: (message: any) => void;
  onError?: (error: unknown) => void;
}

export interface HttpStreamClientSendOptions {
  searchParams?: Record<string, string>;
  headers?: Record<string, string>;
  data?: any;
  body?: string | Blob | ArrayBuffer | ArrayBufferView;
}

export interface HttpStreamClient {
  /**
   * Connect HTTP stream and send data to the server.
   * @param data The data to send.
   */
  send: (data?: HttpStreamClientSendOptions) => void;

  /**
   * Abort the HTTP stream.
   */
  abort: () => void;
}

/**
 * Factory for stream client via HTTP (NDJSON format).
 */
export const createHttpStreamFactory =
  (commonOptions: HttpCommonOptions) =>
  (options: HttpStreamClientOptions): HttpStreamClient => {
    const { url: _url, method = 'GET', afterFetch = (data) => data } = commonOptions;
    const {
      debugName,
      autoReconnect = false,
      autoReconnectInterval = 2000,
      onStatusChange: onStatusChangeCallback,
      onMessage: onMessageCallback,
      onError: onErrorCallback,
    } = options;

    const decoder = new TextDecoder();
    let abortController: AbortController | undefined;
    let reader: ReadableStreamDefaultReader | undefined;

    const debugLog = (...args: any[]) => {
      console.log(
        `%c http-stream.ts ${method} ${_url} ${debugName ?? ''}`,
        'color: white; background: darkcyan; padding: 2px 4px; border-radius: 2px;',
        ...args,
      );
    };

    const onStatusChange = (status: HttpStreamClientStatus): void => {
      debugLog('onStatusChange', status);
      if (autoReconnect && (status === 'disconnected' || status === 'error')) {
        setTimeout(() => {
          debugLog('auto-reconnect');
          send(previousSendOptions);
        }, autoReconnectInterval);
      }
      try {
        onStatusChangeCallback?.(status);
      } catch (error) {
        debugLog('onStatusChange Error', error);
      }
    };

    const onMessage = (message: any): void => {
      // debugLog(
      //   `onMessage ${message.type}`,
      //   ...(Array.isArray(message.data) ? message.data : [message.data])
      // );
      try {
        onMessageCallback?.(message);
      } catch (error) {
        debugLog('onMessage Error', error);
      }
    };

    const cleanupReader = async () => {
      try {
        if (reader) {
          await reader.cancel();
          reader = undefined;
        }
      } catch (err) {
        debugLog('reader cancel error', err);
      }
    };

    let previousSendOptions: HttpStreamClientSendOptions | undefined;
    const send = (options?: HttpStreamClientSendOptions) => {
      previousSendOptions = options;
      const { headers, body, data } = options || {};

      onStatusChange('connecting');
      debugLog('send', data);

      abortController = new AbortController();
      let timeoutId: ReturnType<typeof setTimeout> | undefined;

      const isQueueInfo = data?.type === 'queue_info';
      const timeoutMs = isQueueInfo ? 0 : 5000;

      if (timeoutMs > 0) {
        timeoutId = setTimeout(() => {
          debugLog(`fetch timeout: triggering abort after ${timeoutMs}ms`);
          try {
            abortController?.abort();
          } catch (err) {
            debugLog('abort error during timeout', err);
          }
          onStatusChange('error');
          onErrorCallback?.(new Error('The request timed out. Please try again later.'));
        }, timeoutMs);
      }

      fetch(_url, {
        headers: {
          'Content-Type': 'application/json',
          ...headers,
        },
        method,
        body: (data && JSON.stringify(data)) || body,
        signal: abortController.signal,
      })
        .then(async (response) => {
          if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = undefined;
          }

          if (
            response.status !== 200
            || response.headers.get('content-type') !== 'application/x-ndjson'
          ) {
            debugLog('fetch status error or content-type error', response);
            onStatusChange('error');
            return;
          }

          reader = response.body?.getReader();
          if (!reader) {
            debugLog('getReader error', response);
            onStatusChange('error');
            return;
          }

          onStatusChange('connected');

          let buffer = '';
          while (true) {
            const { done, value } = await reader.read();

            if (done) {
              await cleanupReader();
              onStatusChange('disconnected');
              break;
            }

            const chunk = decoder.decode(value);
            buffer += chunk;
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            lines.forEach((line) => {
              try {
                const message = afterFetch(JSON.parse(line));
                onMessage(message);
              } catch (error) {
                debugLog('Parse Message Error', error);
              }
            });
          }
        })
        .catch(async (error) => {
          if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = undefined;
          }

          await cleanupReader();
          debugLog('fetch error', error);
          onStatusChange('error');
          onErrorCallback?.(error);
        })
        .finally(async () => {
          if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = undefined;
          }

          await cleanupReader();
          abortController = undefined;
        });
    };

    const abort = () => {
      try {
        abortController?.abort();
        abortController = undefined;
      } catch (error) {
        debugLog('abort error', error);
      }
      reader?.cancel();
      reader = undefined;
      onStatusChange('disconnected');
    };

    const instance: HttpStreamClient = { send, abort };
    return Object.freeze(instance);
  };

// export const createEventSourceClient = (options: HttpStreamClientOptions): HttpStreamClient => {
//   const { url, onMessage: onMessageCallback, onStatusChange: onStatusChangeCallback } = options;

//   let client: EventSource | undefined = undefined;

//   const onStatusChange = (status: HttpStreamClientStatus) => {
//     debugLog("onStatusChange", status);
//     onStatusChangeCallback?.(status);
//   };

//   const onMessage = (event: MessageEvent) => {
//     try {
//       const data = JSON.parse(event.data);
//       debugLog("onMessage", data);
//       onMessageCallback?.(data);
//     } catch (error) {
//       debugLog("onMessage Error", error);
//     }
//   };

//   const onOpen = () => {
//     onStatusChange('connected');
//   };

//   const onError = (event: Event) => {
//     onStatusChange('error');
//   };

//   const connect = () => {
//     if (client) {
//       return;
//     }

//     client = new EventSource(url);

//     client.onmessage = onMessage;
//     client.onopen = onOpen;
//     client.onerror = onError;
//     onStatusChange('connecting');
//   };

//   const disconnect = () => {
//     if (!client) {
//       return;
//     }

//     client.close();
//     client = undefined;
//     onStatusChange('disconnected');
//   };

//   return {
//     connect,
//     disconnect,
//   };
// }
