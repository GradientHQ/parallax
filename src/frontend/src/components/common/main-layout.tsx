import type { FC, PropsWithChildren, ReactNode } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Link, Stack, styled, Typography } from '@mui/material';
import { LogoGradient } from '../brand';
import { useCluster } from '../../services';

export interface MainLayoutProps {
  contentStart?: ReactNode;
  contentEnd?: ReactNode;
}

const MainLayoutRoot = styled(Stack)(({ theme }) => {
  const { palette, spacing } = theme;
  return {
    width: '100%',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    gap: spacing(3),
    padding: spacing(3),
    overflow: 'hidden',
    backgroundColor: palette.grey[100],
  };
});

const MainLayoutHeader = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: '100%',
    flex: 'none',
    justifyContent: 'flex-start',
    alignItems: 'center',
    gap: spacing(1),
  };
});

const MainLayoutContainer = styled(Box)(({ theme }) => {
  const { spacing } = theme;
  return {
    position: 'relative',
    flex: '1',
    width: '100%',
    display: 'flex',
    flexFlow: 'column nowrap',
    justifyContent: 'center',
    alignItems: 'center',
    overflowY: 'hidden',
  };
});

const MainLayoutContent = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: '31rem',
    height: '100%',
    gap: spacing(7),
    paddingInline: spacing(1),
    overflowY: 'auto',
  };
});

const MainLayoutStart = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: 'calc((100% - 30rem) / 2 - 4rem)',
    height: '100%',
    overflow: 'auto',
    position: 'absolute',
    top: 0,
    left: '2rem',
    alignItems: 'flex-end',
    gap: spacing(2),
  };
});

const MainLayoutEnd = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: 'calc((100% - 30rem) / 2 - 4rem)',
    height: '100%',
    overflow: 'auto',
    position: 'absolute',
    top: 0,
    right: '2rem',
    alignItems: 'flex-start',
    gap: spacing(2),
  };
});

export const MainLayout: FC<PropsWithChildren<MainLayoutProps>> = ({
  children,
  contentStart,
  contentEnd = <DebugInfo />,
}) => {
  return (
    <MainLayoutRoot>
      <MainLayoutHeader direction='row'>
        <LogoGradient />
      </MainLayoutHeader>
      <MainLayoutContainer>
        <MainLayoutContent className='MainLayoutContent'>{children}</MainLayoutContent>
        {(contentStart && <MainLayoutStart>{contentStart}</MainLayoutStart>) || undefined}
        {(contentEnd && <MainLayoutEnd>{contentEnd}</MainLayoutEnd>) || undefined}
      </MainLayoutContainer>
    </MainLayoutRoot>
  );
};

const DebugInfo: FC = () => {
  const [{ initNodesNumber, networkType, modelName, clusterInfo }] = useCluster();

  const renderRecord = (title: string, record: Record<string, any>) => (
    <Stack gap={1}>
      <Typography variant='subtitle2'>{title}</Typography>
      {Object.entries(record).map(([key, value]) => (
        <Typography variant='body2' key={key}>
          {key}: {JSON.stringify(value)}
        </Typography>
      ))}
    </Stack>
  );

  return (
    (import.meta.env.DEV && (
      <Stack gap={4} data-debug-info='true'>
        <Typography variant='subtitle1'>Debug Info</Typography>

        {renderRecord('Init Parameters', { initNodesNumber, networkType, modelName })}

        {renderRecord('Status Info', clusterInfo)}

        <Stack>
          <Link component={RouterLink} to='/setup'>
            Setup
          </Link>
          <Link component={RouterLink} to='/join'>
            Join
          </Link>
          <Link component={RouterLink} to='/chat'>
            Chat
          </Link>
        </Stack>
      </Stack>
    ))
    || null
  );
};
