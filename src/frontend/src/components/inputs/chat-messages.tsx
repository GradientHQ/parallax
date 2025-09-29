import { memo, useEffect, useRef, useState, type FC, type UIEventHandler } from 'react';
import { useChat, type ChatMessage } from '../../services';
import { Box, Button, IconButton, Paper, Stack, Tooltip, Typography } from '@mui/material';
import {
  IconArrowDown,
  IconCopy,
  IconCopyCheck,
  IconRefresh,
} from '@tabler/icons-react';
import { useRefCallback } from '../../hooks';
import ChatMarkdown from './chat-markdown';
import { DotPulse } from './dot-pulse';

export const ChatMessages: FC = () => {
  const [{ status, messages }] = useChat();

  const refContainer = useRef<HTMLDivElement>(null);
  const refBottom = useRef<HTMLDivElement>(null);
  const [isBottom, setIsBottom] = useState(true);

  const userScrolledUpRef = useRef(false);
  const autoScrollingRef = useRef(false);
  const prevScrollTopRef = useRef(0);

  useEffect(() => {
    if (userScrolledUpRef.current) return;
    autoScrollingRef.current = true;
    refBottom.current?.scrollIntoView({ behavior: 'smooth' });
    const t = setTimeout(() => { autoScrollingRef.current = false; }, 200);
    return () => clearTimeout(t);
  }, [messages]);

  const onScroll = useRefCallback<UIEventHandler<HTMLDivElement>>(() => {
    const container = refContainer.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const bottomGap = scrollHeight - scrollTop - clientHeight;

    setIsBottom(bottomGap < 10);

    if (!autoScrollingRef.current) {
      if (scrollTop < prevScrollTopRef.current - 2) {
        userScrolledUpRef.current = true;
      }
    }
    prevScrollTopRef.current = scrollTop;

    if (bottomGap < 10) {
      userScrolledUpRef.current = false;
    }
  });

  const scrollToBottom = () => {
    userScrolledUpRef.current = false;
    autoScrollingRef.current = true;
    refBottom.current?.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => { autoScrollingRef.current = false; }, 200);
  };

  return (
    <Stack
      ref={refContainer}
      sx={{
        flex: 1,
        overflowX: 'hidden',
        overflowY: 'auto',
        gap: 4,
        '&::-webkit-scrollbar': { display: 'none' },
        scrollbarWidth: 'none',
        msOverflowStyle: 'none',
      }}
      onScroll={onScroll}
      // 额外保险：用户滚轮“向上”也算主动上滑
      onWheel={(e) => {
        if (e.deltaY < 0) userScrolledUpRef.current = true;
      }}
      onTouchMove={() => { userScrolledUpRef.current = true; }} // 移动端手势上滑
    >
      {messages.map((message, idx) => (
        <ChatMessage key={message.id} message={message} isLast={idx === messages.length - 1} />
      ))}

      {status === 'opened' && <DotPulse size="large" />}

      <Box ref={refBottom} sx={{ width: '100%', height: 0 }} />

      {!isBottom && (
        <IconButton
          onClick={scrollToBottom}
          size="small"
          sx={{
            position: 'sticky',
            bottom: 0,
            alignSelf: 'flex-end',
            mr: 1.5,
            width: 28,
            height: 28,
            bgcolor: 'white',
            border: '1px solid',
            borderColor: 'grey.300',
            '&:hover': { bgcolor: 'grey.100' },
          }}
          aria-label="Scroll to bottom"
        >
          <IconArrowDown />
        </IconButton>
      )}
    </Stack>
  );
};

const ChatMessage: FC<{ message: ChatMessage; isLast?: boolean }> = memo(({ message, isLast }) => {
  const { role, content } = message;

  const [{ status }, { generate }] = useChat();

  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const timeoutId = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timeoutId);
  }, [copied]);

  const onCopy = useRefCallback(() => {
    navigator.clipboard.writeText(content);
    setCopied(true);
  });

  const onRegenerate = useRefCallback(() => {
    generate(message);
  });

  const justifyContent = role === 'user' ? 'flex-end' : 'flex-start';

  const nodeContent =
    role === 'user' ?
      <Typography
        variant='body1'
        sx={{ px: 2, py: 1.5, borderRadius: '0.5rem', backgroundColor: 'background.default', fontSize: '0.95rem' }}
      >
        {content}
      </Typography>
    : <ChatMarkdown content={content} />;


  const assistantDone = !isLast || status === 'closed';


  const showCopy = role === 'user' || (role === 'assistant' && assistantDone);
  const showRegen = role === 'assistant' && assistantDone;

  const userHoverRevealSx =
    role === 'user' ?
      {
        '&:hover .actions-user': {
          opacity: 1,
          pointerEvents: 'auto',
        },
      }
    : {};

  return (
    <Stack direction='row' sx={{ width: '100%', justifyContent }}>
      {/* 父容器：承载 hover */}
      <Stack sx={{ maxWidth: '100%', gap: 1, ...userHoverRevealSx }}>
        {nodeContent}

        {(showCopy || showRegen) && (
          <Stack
            key='actions'
            direction='row'
            className={role === 'user' ? 'actions-user' : undefined}
            sx={{
              justifyContent,
              color: 'grey.600',
              gap: 0.5,
              ...(role === 'user' ?
                {
                  opacity: 0, 
                  pointerEvents: 'none', 
                  transition: 'opacity .15s ease',
                }
              : {}),
            }}
          >
            {showCopy && (
              <Tooltip
                title={copied ? 'Copied!' : 'Copy'}
                slotProps={{
                  tooltip: { sx: { bgcolor: 'primary.main', borderRadius: 1 } },
                  popper: { modifiers: [{ name: 'offset', options: { offset: [0, -8] } }] },
                }}
              >
                <IconButton
                  onClick={onCopy}
                  size='small'
                  sx={{
                    width: 24,
                    height: 24,
                    borderRadius: '8px',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  {copied ?
                    <IconCopyCheck />
                  : <IconCopy />}
                </IconButton>
              </Tooltip>
            )}

            {showRegen && (
              <Tooltip
                title='Regenerate'
                slotProps={{
                  tooltip: { sx: { bgcolor: 'primary.main', borderRadius: 1 } },
                  popper: { modifiers: [{ name: 'offset', options: { offset: [0, -8] } }] },
                }}
              >
                <IconButton
                  onClick={onRegenerate}
                  size='small'
                  sx={{
                    width: 24,
                    height: 24,
                    borderRadius: '8px',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  <IconRefresh />
                </IconButton>
              </Tooltip>
            )}
          </Stack>
        )}
      </Stack>
    </Stack>
  );
});
