import { useEffect, useState, type FC } from 'react';
import { IconButton, Paper, styled, Typography } from '@mui/material';
import { useCluster } from '../../services';
import { IconCopy, IconCopyCheck } from '@tabler/icons-react';
import { useRefCallback } from '../../hooks';

const JoinCommandRoot = styled('div')(({ theme }) => {
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

  const [isCopied, setIsCopied] = useState(false);
  useEffect(() => {
    if (isCopied) {
      const timeoutId = setTimeout(() => {
        setIsCopied(false);
      }, 2000);
      return () => clearTimeout(timeoutId);
    }
  }, [isCopied]);

  const onCopy = useRefCallback(async () => {
    await navigator.clipboard.writeText(nodeJoinCommand);
    setIsCopied(true);
  });

  return (
    <JoinCommandRoot>
      <Typography sx={{ flex: 1, lineHeight: '1.125rem', whiteSpace: 'wrap' }} variant='pre'>
        {nodeJoinCommand}
      </Typography>
      <IconButton sx={{ flex: 'none', fontSize: '1.5rem' }} size='em' onClick={onCopy}>
        {(isCopied && <IconCopyCheck />) || <IconCopy />}
      </IconButton>
    </JoinCommandRoot>
  );
};
