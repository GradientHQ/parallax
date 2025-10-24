import { memo, useEffect, useRef, useState, type FC, type UIEventHandler } from 'react';
import { useChat, type ChatMessage } from '../../services';
import { Box, Button, IconButton, Paper, Stack, Tooltip, Typography } from '@mui/material';
import { IconArrowDown, IconCopy, IconCopyCheck, IconRefresh } from '@tabler/icons-react';
import { useRefCallback } from '../../hooks';
import ChatMarkdown from './chat-markdown';
import { DotPulse } from './dot-pulse';

export const ChatMessages: FC = () => {
  const [{ status, messages }] = useChat();

  const refContainer = useRef<HTMLDivElement>(null);
  // const refBottom = useRef<HTMLDivElement>(null);
  const [isBottom, setIsBottom] = useState(true);

  const userScrolledUpRef = useRef(false);
  const autoScrollingRef = useRef(false);
  const prevScrollTopRef = useRef(0);

  const scrollToBottom = useRefCallback(() => {
    const el = refContainer.current;
    if (!el) return;
    userScrolledUpRef.current = false;
    autoScrollingRef.current = true;
    requestAnimationFrame(() => {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
      // el.lastElementChild?.scrollIntoView({ behavior: 'smooth' });
    });
    setTimeout(() => {
      autoScrollingRef.current = false;
    }, 250);
  });

  useEffect(() => {
    if (userScrolledUpRef.current) return;
    autoScrollingRef.current = true;
    scrollToBottom();
    const t = setTimeout(() => {
      autoScrollingRef.current = false;
    }, 200);
    return () => clearTimeout(t);
  }, [messages]);

  const onScroll = useRefCallback<UIEventHandler<HTMLDivElement>>((event) => {
    event.stopPropagation();

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

  const nodeScrollToBottomButton = (
    <IconButton
      key='scroll-to-bottom'
      onClick={scrollToBottom}
      size='small'
      aria-label='Scroll to bottom'
      sx={{
        position: 'absolute',
        right: 12,
        bottom: 8,
        width: 28,
        height: 28,
        bgcolor: 'white',
        border: '1px solid',
        borderColor: 'grey.300',
        '&:hover': { bgcolor: 'grey.100' },
        opacity: isBottom ? 0 : 1,
        pointerEvents: isBottom ? 'none' : 'auto',
        transition: 'opacity .15s ease',
      }}
    >
      <IconArrowDown />
    </IconButton>
  );

  const nodeStream = (
    <Stack
      key='stream'
      ref={refContainer}
      sx={{
        width: '100%',
        height: '100%',

        overflowX: 'hidden',
        overflowY: 'scroll',
        '&::-webkit-scrollbar': { display: 'none' },
        scrollbarWidth: 'none',
        msOverflowStyle: 'none',

        display: 'flex',
        gap: 4,
      }}
      onScroll={onScroll}
      onWheel={(e) => {
        if (e.deltaY < 0) userScrolledUpRef.current = true;
      }}
      onTouchMove={() => {
        userScrolledUpRef.current = true;
      }}
    >
      {messages.map((message, idx) => (
        <ChatMessage key={message.id} message={message} isLast={idx === messages.length - 1} />
      ))}

      {status === 'opened' && <DotPulse size='large' />}

      {/* Last child for scroll to bottom */}
      <Box sx={{ width: '100%', height: 0 }} />
    </Stack>
  );

  return (
    <Box
      sx={{
        position: 'relative',
        flex: 1,
        overflow: 'hidden',
      }}
    >
      {nodeStream}
      {nodeScrollToBottomButton}
    </Box>
  );
};

const ChatMessage: FC<{ message: ChatMessage; isLast?: boolean }> = memo(({ message, isLast }) => {
  const { role, status: messageStatus, thinking, content } = message;

  const [, { generate }] = useChat();

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
        key='user-message'
        variant='body1'
        sx={{
          px: 2,
          py: 1.5,
          borderRadius: '0.5rem',
          backgroundColor: 'background.default',
          fontSize: '0.875rem',
        }}
      >
        {content}
      </Typography>
    : <>
        {thinking && <ChatMarkdown key='assistant-thinking' isThinking content={thinking} />}
        {content && <ChatMarkdown key='assistant-message' content={content} />}
      </>;

  const assistantDone = messageStatus === 'done';
  const showCopy = role === 'user' || (role === 'assistant' && assistantDone);
  const showRegenerate = role === 'assistant' && assistantDone;

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
      <Stack
        sx={{
          maxWidth: role === 'user' ? { xs: '100%', md: '80%' } : '100%',
          alignSelf: role === 'user' ? 'flex-end' : 'flex-start',
          gap: 1,
          ...userHoverRevealSx,
        }}
      >
        {nodeContent}

        {(showCopy || showRegenerate) && (
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
                key='copy'
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

            {showRegenerate && (
              <Tooltip
                key='regenerate'
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
