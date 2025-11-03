import { Button, Stack, TextField } from '@mui/material';
import {
  useRef,
  type CompositionEventHandler,
  type FC,
  type KeyboardEventHandler,
  type MouseEventHandler,
} from 'react';
import { useRefCallback } from '../../hooks';
import { useChat, useCluster } from '../../services';
import { IconArrowBackUp, IconArrowUp, IconSquareFilled } from '@tabler/icons-react';
import { DotPulse } from './dot-pulse';

export const ChatInput: FC = () => {
  const [
    {
      modelName,
      clusterInfo: { status: clusterStatus },
    },
  ] = useCluster();
  const [{ input, status }, { setInput, generate, stop, clear }] = useChat();

  const compositionRef = useRef(false);

  const onCompositionStart = useRefCallback<CompositionEventHandler>((e) => {
    compositionRef.current = true;
  });

  const onCompositionEnd = useRefCallback<CompositionEventHandler>((e) => {
    compositionRef.current = false;
  });

  const onKeyDown = useRefCallback<KeyboardEventHandler>((e) => {
    if (e.key === 'Enter' && !e.shiftKey && !compositionRef.current) {
      e.preventDefault();
      generate();
    }
  });

  const onClickMainButton = useRefCallback<MouseEventHandler>((e) => {
    if (status === 'opened' || status === 'generating') {
      stop();
    } else if (status === 'closed' || status === 'error') {
      generate();
    }
  });

  const onClickClearButton = useRefCallback<MouseEventHandler>((e) => {
    clear();
  });

  return (
    <Stack data-status={status}>
      {/* <Stack direction='row' sx={{ gap: 1, p: 1 }}>
        {modelName}
      </Stack> */}
      <TextField
        value={input}
        onChange={(event) => setInput(event.target.value)}
        multiline
        maxRows={4}
        placeholder='Ask anything'
        fullWidth
        onCompositionStart={onCompositionStart}
        onCompositionEnd={onCompositionEnd}
        onKeyDown={onKeyDown}
        slotProps={{
          input: {
            sx: {
              border: '1px solid',
              borderColor: 'grey.300',
              borderRadius: 2,
              fontSize: '0.95rem',
              boxShadow: '2px 2px 4px rgba(0,0,0,0.05)',
              flexDirection: 'column',
              '& textarea': {
                fontSize: '0.875rem',
                scrollbarWidth: 'none', // Firefox
                msOverflowStyle: 'none', // IE, Edge
                '&::-webkit-scrollbar': {
                  display: 'none', // Chrome, Safari
                },
              },
            },
            endAdornment: (
              <Stack direction='row' sx={{ alignSelf: 'flex-end', alignItems: 'center', gap: 2 }}>
                <Button
                  variant='text'
                  sx={{ color: 'text.secondary' }}
                  startIcon={<IconArrowBackUp />}
                  disabled={status === 'opened' || status === 'generating'}
                  onClick={onClickClearButton}
                >
                  Clear
                </Button>
                <Button
                  size='small'
                  color='primary'
                  disabled={clusterStatus !== 'available' || status === 'opened'}
                  // loading={status === 'opened'}
                  onClick={onClickMainButton}
                >
                  {status === 'opened' ?
                    <DotPulse size='medium' />
                  : status === 'generating' ?
                    <IconSquareFilled size='1.25rem' />
                  : <IconArrowUp size='1.25rem' />}
                </Button>
              </Stack>
            ),
          },
        }}
      />
    </Stack>
  );
};
