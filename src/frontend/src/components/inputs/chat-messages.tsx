import { memo, useEffect, useRef, useState, type FC, type UIEventHandler } from 'react';
import { useChat, type ChatMessage } from '../../services';
import { Box, Button, Paper, Stack, Typography } from '@mui/material';
import { IconCopy, IconCopyCheck, IconRefresh } from '@tabler/icons-react';
import { useRefCallback } from '../../hooks';
import ChatMarkdown from './chat-markdown';

export const ChatMessages: FC = () => {
  const [{ messages }] = useChat();

  const refContainer = useRef<HTMLDivElement>(null);
  const refBottom = useRef<HTMLDivElement>(null);
  const [isBottom, setIsBottom] = useState(true);

  useEffect(() => {
    refBottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const onScroll = useRefCallback<UIEventHandler<HTMLDivElement>>((event) => {
    const { current: container } = refContainer;
    if (!container) {
      return;
    }
    setIsBottom(container.scrollHeight - container.scrollTop - container.clientHeight < 10);
  });

  return (
    <Stack
      ref={refContainer}
      sx={{ flex: 1, overflowX: 'hidden', overflowY: 'auto', gap: 4 }}
      onScroll={onScroll}
    >
      {messages.map((message) => (
        <ChatMessage key={message.id} message={message} />
      ))}

      <Box
        ref={refBottom}
        sx={{
          width: '100%',
          height: 0,
        }}
      />
    </Stack>
  );
};

const ChatMessage: FC<{ message: ChatMessage }> = memo(({ message }) => {
  const { role, content } = message;

  const [{ status }, { generate }] = useChat();

  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setCopied(false);
    }, 2000);
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
        sx={{ px: 2, py: 1.5, borderRadius: '0.5rem', backgroundColor: 'background.default' }}
      >
        {content}
      </Typography>
    : <ChatMarkdown content={content} />;

  return (
    <Stack direction='row' sx={{ width: '100%', justifyContent }}>
      <Stack sx={{ width: '35rem', gap: 1 }}>
        {nodeContent}

        <Stack key='actions' direction='row' sx={{ justifyContent, gap: 2, color: 'grey.600' }}>
          <Button
            variant='text'
            startIcon={copied ? <IconCopyCheck /> : <IconCopy />}
            onClick={onCopy}
          >
            Copy
          </Button>
          {role === 'assistant' && (
            <Button variant='text' startIcon={<IconRefresh />} onClick={onRegenerate}>
              Regenerate
            </Button>
          )}
        </Stack>
      </Stack>
    </Stack>
  );
});
