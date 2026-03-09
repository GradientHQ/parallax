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
import { parseGenerationGpt, parseGenerationQwen } from './chat-helper';

const debugLog = async (...args: any[]) => {
  if (import.meta.env.DEV) {
    console.log('%c chat.tsx ', 'color: white; background: orange;', ...args);
  }
};

export type ChatMessageRole = 'user' | 'assistant';

export type ChatMessageStatus = 'waiting' | 'thinking' | 'generating' | 'done' | 'error';

export interface ChatMessage {
  readonly id: string;
  readonly role: ChatMessageRole;
  readonly status: ChatMessageStatus;

  /**
   * The content from user input or assistant generating.
   */
  readonly content: string;

  /**
   * The raw content from model response.
   */
  readonly raw?: string;

  /**
   * The thinking content in assistant generating.
   */
  readonly thinking?: string;
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
      clusterInfo: { status: clusterStatus, modelName },
    },
  ] = useCluster();

  const [input, setInput] = useState<string>('');

  const [status, _setStatus] = useState<ChatStatus>('closed');
  const setStatus = useRefCallback<typeof _setStatus>((value) => {
    _setStatus((prev) => {
      const next = typeof value === 'function' ? value(prev) : value;
      if (next !== prev) {
        debugLog('setStatus', 'status', next);
      }
      return next;
    });
  });

  const [messages, setMessages] = useState<readonly ChatMessage[]>([]);

  const sse = useConst(() =>
    createSSE({
      onOpen: () => {
        debugLog('SSE OPEN');
        setStatus('opened');
      },
      onClose: () => {
        debugLog('SSE CLOSE');
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          const { id, raw, thinking, content } = lastMessage;
          debugLog('GENERATING DONE', 'lastMessage:', lastMessage);
          debugLog('GENERATING DONE', 'id:', id);
          debugLog('GENERATING DONE', 'raw:', raw);
          debugLog('GENERATING DONE', 'thinking:', thinking);
          debugLog('GENERATING DONE', 'content:', content);
          return [
            ...prev.slice(0, -1),
            {
              ...lastMessage,
              status: 'done',
            },
          ];
        });
        setStatus('closed');
      },
      onError: (error) => {
        debugLog('SSE ERROR', error);
        // Set last message to done
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          const { id, raw, thinking, content } = lastMessage;
          debugLog('GENERATING ERROR', 'lastMessage:', lastMessage);
          debugLog('GENERATING ERROR', 'id:', id);
          debugLog('GENERATING ERROR', 'raw:', raw);
          debugLog('GENERATING ERROR', 'thinking:', thinking);
          debugLog('GENERATING ERROR', 'content:', content);
          return [
            ...prev.slice(0, -1),
            {
              ...lastMessage,
              status: 'done',
            },
          ];
        });
        debugLog('SSE ERROR', error);
        setStatus('error');
      },
      onMessage: (message) => {
        // debugLog('onMessage', message);
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
        const {
          data: { id, object, model, created, choices, usage },
        } = message;
        if (object === 'chat.completion.chunk' && choices?.length > 0) {
          if (choices[0].delta.content) {
            setStatus('generating');
          }
          setMessages((prev) => {
            let next = prev;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            choices.forEach(({ delta: { role, content: rawDelta } = {} }: any) => {
              if (typeof rawDelta !== 'string' || !rawDelta) {
                return;
              }
              role = role || 'assistant';
              let lastMessage = next[next.length - 1];
              if (lastMessage && lastMessage.role === role) {
                const raw = lastMessage.raw + rawDelta;
                lastMessage = {
                  ...lastMessage,
                  raw: raw,
                  content: raw,
                };
                next = [...next.slice(0, -1), lastMessage];
              } else {
                lastMessage = {
                  id,
                  role,
                  status: 'thinking',
                  raw: rawDelta,
                  content: rawDelta,
                  createdAt: created,
                };
                next = [...next, lastMessage];
              }
              // debugLog('onMessage', 'update last message', lastMessage.content);
            });

            // Parse generation and extract thinking and content
            if (next !== prev && typeof model === 'string') {
              let lastMessage = next[next.length - 1];
              let thinking = '';
              let content = '';
              const modelLowerCase = model.toLowerCase();
              if (modelLowerCase.includes('gpt-oss')) {
                ({ analysis: thinking, final: content } = parseGenerationGpt(
                  lastMessage.raw || '',
                ));
              } else if (modelLowerCase.includes('qwen')) {
                ({ think: thinking, content } = parseGenerationQwen(lastMessage.raw || ''));
              } else {
                content = lastMessage.raw || '';
              }
              lastMessage = {
                ...lastMessage,
                status: (content && 'generating') || 'thinking',
                thinking,
                content,
              };
              next = [...next.slice(0, -1), lastMessage];
            }

            return next;
          });
        }
      },
    }),
  );

  const generate = useRefCallback<ChatActions['generate']>((message) => {
    if (clusterStatus !== 'available' || status === 'opened' || status === 'generating') {
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
        { id: now.toString(), role: 'user', status: 'done', content: finalInput, createdAt: now },
      ];
      debugLog('generate', 'new', nextMessages);
    }
    setMessages(nextMessages);

    sse.connect(
      modelName,
      nextMessages.map(({ id, role, content }) => ({ id, role, content })),
    );
  });

  const stop = useRefCallback<ChatActions['stop']>(() => {
    debugLog('stop', 'status', status);
    if (status === 'closed' || status === 'error') {
      return;
    }
    sse.disconnect();
  });

  const clear = useRefCallback<ChatActions['clear']>(() => {
    debugLog('clear', 'status', status);
    stop();
    if (status === 'opened' || status === 'generating') {
      return;
    }
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

interface RequestMessage {
  readonly id: string;
  readonly role: ChatMessageRole;
  readonly content: string;
}

const createSSE = (options: SSEOptions) => {
  const { onOpen, onClose, onError, onMessage } = options;

  const decoder = new TextDecoder();
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let abortController: AbortController | undefined;

  const connect = (model: string, messages: readonly RequestMessage[]) => {
    abortController = new AbortController();
    const url = `${API_BASE_URL}/v1/chat/completions`;

    onOpen?.();

    fetch(url, {
      method: 'POST',
      body: JSON.stringify({
        stream: true,
        model,
        messages,
        max_tokens: 2048,
        sampling_params: {
          top_k: 3,
        },
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
