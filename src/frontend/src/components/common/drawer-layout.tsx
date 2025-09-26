import { useEffect, useState, type FC, type PropsWithChildren } from 'react';
import { Box, Button, IconButton, Stack, styled, Typography } from '@mui/material';
import { useCluster } from '../../services';
import { useAlertDialog } from '../mui';
import { IconBrandGradient } from '../brand';
import {
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarLeftExpand,
  IconLayoutSidebarRightCollapse,
  IconLayoutSidebarRightExpand,
  IconPlus,
} from '@tabler/icons-react';
import { NodeList } from '../inputs';

const DrawerLayoutRoot = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: '100%',
    height: '100%',
    justifyContent: 'flex-start',
    alignItems: 'stretch',
    overflow: 'hidden',
  };
});

const DrawerLayoutSide = styled(Stack)(({ theme }) => {
  const { palette, spacing } = theme;
  return {
    height: '100%',
    paddingBlock: spacing(3),
    paddingInline: spacing(2),
    gap: spacing(3),
    overflow: 'hidden',
    transition: 'width 0.3s ease-in-out',
    backgroundColor: palette.grey[200],
  };
});

const DrawerLayoutHeader = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: '100%',
    height: '4rem',
    flex: 'none',
    paddingBlock: spacing(2),
    paddingInline: spacing(4),
    overflow: 'hidden',
  };
});

const DrawerLayoutContainer = styled(Stack)(({ theme }) => {
  const { palette, spacing } = theme;
  return {
    flex: 1,
    alignItems: 'center',
    overflow: 'hidden',
    backgroundColor: palette.grey[100],
  };
});

const DrawerLayoutContent = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: '48.75rem',
    maxWidth: '100%',
    height: '100%',
    gap: spacing(4),
    padding: spacing(4),
    overflow: 'hidden',
  };
});

export const DrawerLayout: FC<PropsWithChildren> = ({ children }) => {
  const [
    {
      modelName,
      clusterInfo: { status: clusterStatus },
    },
  ] = useCluster();

  const [dialogRebalancing, { open, close }] = useAlertDialog({
    color: 'error',
    titleIcon: true,
    title: '',
    content: (
      <>
        <Typography variant='subtitle1'>Cluster rebalancing</Typography>
        <Typography variant='body1'>
          The cluster is rebalancing. Please wait for a moment.
        </Typography>
      </>
    ),
    confirmLabel: 'Finish',
  });
  useEffect(() => {
    if (clusterStatus === 'rebalancing') {
      open();
    }
  }, [clusterStatus, open]);

  const [sidebarExpanded, setMenuOpen] = useState(true);

  return (
    <DrawerLayoutRoot direction='row'>
      <DrawerLayoutSide sx={{ width: sidebarExpanded ? '21.875rem' : '3.5rem' }}>
        <Stack direction='row' sx={{ justifyContent: 'flex-end', alignItems: 'center', gap: 2 }}>
          {sidebarExpanded && <IconBrandGradient />}
          <Box sx={{ flex: 1 }} />
          <IconButton
            size='em'
            sx={{ fontSize: '1.5rem' }}
            onClick={() => setMenuOpen((prev) => !prev)}
          >
            {sidebarExpanded ?
              <IconLayoutSidebarLeftCollapse />
            : <IconLayoutSidebarLeftExpand />}
          </IconButton>
        </Stack>
        {sidebarExpanded && (
          <Stack sx={{ gap: 4 }}>
            <NodeList variant='menu' />
            <Button color='info' startIcon={<IconPlus />}>
              Add Nodes
            </Button>
          </Stack>
        )}
      </DrawerLayoutSide>
      <DrawerLayoutContainer>
        <DrawerLayoutHeader direction='row'>
          <Typography variant='h2' fontWeight={500}>
            {modelName}
          </Typography>
        </DrawerLayoutHeader>
        <DrawerLayoutContent>{children}</DrawerLayoutContent>
      </DrawerLayoutContainer>
      {dialogRebalancing}
    </DrawerLayoutRoot>
  );
};
