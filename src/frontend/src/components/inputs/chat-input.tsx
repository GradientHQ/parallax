import { Button, Stack, TextField } from '@mui/material';
import type { FC, KeyboardEventHandler } from 'react';
import { useRefCallback } from '../../hooks';
import { useChat, useCluster } from '../../services';
import {
  IconArrowBackUp,
  IconArrowUp,
  IconLoader,
  IconSquare,
  IconSquareFilled,
} from '@tabler/icons-react';

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
    <Stack>
      <Stack direction='row' sx={{ gap: 1, p: 1 }}>
        {modelName}
      </Stack>
      <TextField
        value={input}
        onChange={(event) => setInput(event.target.value)}
        multiline
        maxRows={4}
        placeholder='Enter your system prompt here...'
        fullWidth
        onKeyDown={onKeyDown}
        slotProps={{
          input: {
            sx: { flexDirection: 'column' },
            endAdornment: (
              <Stack direction='row' sx={{ alignSelf: 'flex-end', alignItems: 'center', gap: 1 }}>
                <Button variant='text' startIcon={<IconArrowBackUp />} onClick={clear}>
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
                    <IconLoader size='1.25rem' />
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
