/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useContext,
  useMemo,
  useState,
  type Dispatch,
  type FC,
  type PropsWithChildren,
  type SetStateAction,
} from 'react';
import { API_BASE_URL } from './api';
import { useConst, useRefCallback } from '../hooks';
import { useCluster } from './cluster';

const debugLog = (...args: any[]) => {
  if (import.meta.env.DEV) {
    console.log('%c chat.tsx ', 'color: white; background: orange;', ...args);
  }
};

export type ChatMessageRole = 'user' | 'assistant';

export interface ChatMessage {
  readonly id: string;
  readonly role: ChatMessageRole;
  readonly content: string;
  readonly createdAt: number;
}

export type ChatStatus = 'closed' | 'opened' | 'generating' | 'error';

export interface ChatStates {
  readonly input: string;
  readonly status: ChatStatus;
  readonly messages: readonly ChatMessage[];
}

export interface ChatActions {
  readonly setInput: Dispatch<SetStateAction<string>>;
  readonly generate: (message?: ChatMessage) => void;
  readonly stop: () => void;
  readonly clear: () => void;
}

export const ChatProvider: FC<PropsWithChildren> = ({ children }) => {
  const [
    {
      modelName,
      clusterInfo: { status: clusterStatus },
    },
  ] = useCluster();

  const [input, setInput] = useState<string>('');

  const [status, setStatus] = useState<ChatStatus>('closed');
  const [messages, setMessages] = useState<readonly ChatMessage[]>([]);

  const sse = useConst(() =>
    createSSE({
      onOpen: () => setStatus('opened'),
      onClose: () => setStatus('closed'),
      onError: (error) => setStatus('error'),
      onMessage: (message) => {
        debugLog('onMessage', message);
        // const example = {
        //   id: 'd410014e-3308-450d-bbd2-0ec4e0c0a345',
        //   object: 'chat.completion.chunk',
        //   model: 'default',
        //   created: 1758842801.822061,
        //   choices: [
        //     {
        //       index: 0,
        //       logprobs: null,
        //       finish_reason: null,
        //       matched_stop: null,
        //       delta: { role: null, content: ' the' },
        //     },
        //   ],
        //   usage: null,
        // };
        setStatus('generating');
        const {
          data: { id, object, model, created, choices, usage },
        } = message;
        if (object === 'chat.completion.chunk') {
          setMessages((prev) => {
            let next = prev;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            choices.forEach(({ delta: { role, content } = {} }: any) => {
              if (typeof content !== 'string') {
                return;
              }
              role = role || 'assistant';
              let lastMessage = next[next.length - 1];
              if (lastMessage && lastMessage.role === role) {
                const nextContent = lastMessage.content + content;
                if (nextContent === lastMessage.content) {
                  return;
                }
                lastMessage = {
                  ...lastMessage,
                  content: lastMessage.content + content,
                };
                next = [...next.slice(0, -1), lastMessage];
              } else {
                lastMessage = { id, role, content, createdAt: created };
                next = [...next, lastMessage];
              }
              debugLog('onMessage', 'update last message', lastMessage.content);
            });
            return next;
          });
        }
      },
    }),
  );

  const generate = useRefCallback<ChatActions['generate']>((message) => {
    if (clusterStatus !== 'available' || status === 'opened') {
      return;
    }

    if (!modelName) {
      return;
    }

    let nextMessages: readonly ChatMessage[] = messages;
    if (message) {
      // Regenerate
      const finalMessageIndex = messages.findIndex((m) => m.id === message.id);
      const finalMessage = messages[finalMessageIndex];
      if (!finalMessage) {
        return;
      }
      nextMessages = nextMessages.slice(
        0,
        finalMessageIndex + (finalMessage.role === 'user' ? 1 : 0),
      );
      debugLog('generate', 'regenerate', nextMessages);
    } else {
      // Generate for new input
      const finalInput = input.trim();
      if (!finalInput) {
        return;
      }
      setInput('');
      const now = performance.now();
      nextMessages = [
        ...nextMessages,
        { id: now.toString(), role: 'user', content: finalInput, createdAt: now },
      ];
      debugLog('generate', 'new', nextMessages);
    }
    setMessages(nextMessages);

    sse.connect(modelName, nextMessages);
  });

  const stop = useRefCallback<ChatActions['stop']>(() => {
    if (status !== 'opened') {
      return;
    }
    debugLog('stop');
    sse.disconnect();
  });

  const clear = useRefCallback<ChatActions['clear']>(() => {
    stop();
    if (status === 'opened') {
      return;
    }
    debugLog('clear');
    setMessages([]);
  });

  const actions = useConst<ChatActions>({
    setInput,
    generate,
    stop,
    clear,
  });

  const value = useMemo<readonly [ChatStates, ChatActions]>(
    () => [
      {
        input,
        status,
        messages,
      },
      actions,
    ],
    [input, status, messages, actions],
  );

  return <context.Provider value={value}>{children}</context.Provider>;
};

const context = createContext<readonly [ChatStates, ChatActions] | undefined>(undefined);

export const useChat = (): readonly [ChatStates, ChatActions] => {
  const value = useContext(context);
  if (!value) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return value;
};

// ================================================================
// SSE

interface SSEOptions {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Error) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onMessage?: (message: { event: string; id?: string; data: any }) => void;
}

const createSSE = (options: SSEOptions) => {
  const { onOpen, onClose, onError, onMessage } = options;

  const decoder = new TextDecoder();
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let abortController: AbortController | undefined;

  const connect = (model: string, messages: readonly ChatMessage[]) => {
    abortController = new AbortController();
    fetch(`${API_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      body: JSON.stringify({
        model,
        messages,
        stream: true,
      }),
      signal: abortController.signal,
    })
      .then(async (response) => {
        const statusCode = response.status;
        const contentType = response.headers.get('Content-Type');
        if (statusCode !== 200) {
          onError?.(new Error(`[SSE] Failed to connect: ${statusCode}`));
          return;
        }
        if (!contentType?.includes('text/event-stream')) {
          onError?.(new Error(`[SSE] Invalid content type: ${contentType}`));
          return;
        }

        reader = response.body?.getReader();
        if (!reader) {
          onError?.(new Error(`[SSE] Failed to get reader`));
          return;
        }

        onOpen?.();

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
            onClose?.();
            return;
          }

          const chunk = decoder.decode(value);
          buffer += chunk;

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          processLines(lines);
        }
      })
      .catch((error: Error) => {
        if (error instanceof Error && error.name === 'AbortError') {
          onClose?.();
          return;
        }
        onError?.(error);
      });
  };

  const disconnect = () => {
    reader?.cancel();
    reader = undefined;
    abortController?.abort('stop');
    abortController = undefined;

    onClose?.();
  };

  return { connect, disconnect };
};
