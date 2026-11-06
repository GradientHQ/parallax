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
import { createSSE, type ChatMessageRole, type ChatResponseMetrics } from './chat-sse';
export type { ChatMessageRole, ChatResponseMetrics } from './chat-sse';
import { useConst, useRefCallback } from '../hooks';
import { useCluster } from './cluster';

const debugLog = async (...args: unknown[]) => {
  if (import.meta.env.DEV) {
    console.log('%c chat.tsx ', 'color: white; background: orange;', ...args);
  }
};

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
  readonly metrics?: ChatResponseMetrics;
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
      url: `${API_BASE_URL}/v1/chat/completions`,
      onOpen: () => {
        debugLog('SSE OPEN');
        setStatus('opened');
      },
      onClose: (metrics) => {
        debugLog('SSE CLOSE');
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          if (!lastMessage || lastMessage.role !== 'assistant') {
            return prev;
          }
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
              metrics,
            },
          ];
        });
        setStatus('closed');
      },
      onError: (error, metrics) => {
        debugLog('SSE ERROR', error);
        // Set last message to done
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          if (!lastMessage || lastMessage.role !== 'assistant') {
            return prev;
          }
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
              metrics,
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
          data: { id, object, created, choices },
        } = message;
        if (object === 'chat.completion.chunk' && choices?.length > 0) {
          if (choices[0].delta?.content || choices[0].delta?.reasoning) {
            setStatus('generating');
          }
          setMessages((prev) => {
            let next = prev;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            choices.forEach(({ delta: { role, content, reasoning } = {} }: any) => {
              const contentDelta = typeof content === 'string' ? content : '';
              const reasoningDelta = typeof reasoning === 'string' ? reasoning : '';
              if (!contentDelta && !reasoningDelta) {
                return;
              }
              role = role || 'assistant';
              let lastMessage = next[next.length - 1];
              if (lastMessage && lastMessage.role === role) {
                const raw = (lastMessage.raw || '') + reasoningDelta + contentDelta;
                const nextContent = lastMessage.content + contentDelta;
                const nextThinking = (lastMessage.thinking || '') + reasoningDelta;
                lastMessage = {
                  ...lastMessage,
                  status: (nextContent && 'generating') || 'thinking',
                  raw,
                  thinking: nextThinking,
                  content: nextContent,
                };
                next = [...next.slice(0, -1), lastMessage];
              } else {
                lastMessage = {
                  id,
                  role,
                  status: (contentDelta && 'generating') || 'thinking',
                  raw: reasoningDelta + contentDelta,
                  thinking: reasoningDelta,
                  content: contentDelta,
                  createdAt: created,
                };
                next = [...next, lastMessage];
              }
              // debugLog('onMessage', 'update last message', lastMessage.content);
            });

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
