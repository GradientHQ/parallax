import { Button, Stack, TextField } from '@mui/material';
import type { FC, KeyboardEventHandler } from 'react';
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

  const onKeyDown = useRefCallback<KeyboardEventHandler>((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      generate();
    }
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
                fontSize: '0.95rem',
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
                  onClick={clear}
                >
                  Clear
                </Button>
                <Button
                  size='small'
                  color='primary'
                  disabled={clusterStatus !== 'available'}
                  loading={status === 'opened'}
                  onClick={() => {
                    if (status === 'opened') {
                      stop();
                    } else if (status === 'closed') {
                      generate();
                    }
                  }}
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
