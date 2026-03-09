import { useEffect, useState, type FC } from 'react';
import { IconButton, Paper, Stack, styled, Typography } from '@mui/material';
import { useCluster } from '../../services';
import { IconCopy, IconCopyCheck } from '@tabler/icons-react';
import { useRefCallback } from '../../hooks';

const LABEL_MAP: Record<string, string> = {
  'linux/mac': 'Linux/MacOS',
  windows: 'Windows',
};

const JoinCommandItem = styled('div')(({ theme }) => {
  const { palette, spacing } = theme;
  return {
    display: 'flex',
    flexFlow: 'row nowrap',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingInline: spacing(2),
    paddingBlock: spacing(1.5),
    gap: spacing(1),

    overflow: 'hidden',

    borderRadius: '0.7rem',
    backgroundColor: palette.background.area,
  };
});

export const JoinCommand: FC = () => {
  const [
    {
      clusterInfo: { nodeJoinCommand },
    },
  ] = useCluster();

  const [copiedKey, setCopiedKey] = useState<string>();

  useEffect(() => {
    if (copiedKey) {
      const timeoutId = setTimeout(() => {
        setCopiedKey(undefined);
      }, 2000);
      return () => clearTimeout(timeoutId);
    }
  }, [copiedKey]);

  const copy = useRefCallback(async (key: string) => {
    await navigator.clipboard.writeText(nodeJoinCommand[key]);
    setCopiedKey(key);
  });

  return (
    <Stack gap={1}>
      {Object.entries(nodeJoinCommand).map(([key, value], index, entries) => (
        <Stack key={key} gap={1}>
          {entries.length > 1 && (
            <Typography key='label' variant='subtitle2'>
              For {LABEL_MAP[key] || key}:
            </Typography>
          )}
          <JoinCommandItem key='command'>
            <Typography sx={{ flex: 1, lineHeight: '1.125rem', whiteSpace: 'wrap' }} variant='pre'>
              {value}
            </Typography>
            <IconButton
              sx={{ flex: 'none', fontSize: '1rem' }}
              size='em'
              onClick={() => copy(key)}
            >
              {(copiedKey === key && <IconCopyCheck />) || <IconCopy />}
            </IconButton>
          </JoinCommandItem>
        </Stack>
      ))}
    </Stack>
  );
};
