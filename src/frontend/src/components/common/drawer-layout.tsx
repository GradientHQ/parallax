import { useEffect, useState, type FC, type PropsWithChildren } from 'react';
import { Box, Button, IconButton, Stack, styled, Typography } from '@mui/material';
import { useCluster } from '../../services';
import { useAlertDialog } from '../mui';
import { IconBrandGradient } from '../brand';
import {
  IconCirclePlus,
  IconInfoCircle,
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarLeftExpand,
  IconLayoutSidebarRightCollapse,
  IconLayoutSidebarRightExpand,
  IconPlus,
} from '@tabler/icons-react';
import { JoinCommand, NodeList } from '../inputs';

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

  const [dialogWaiting, { open: openWaiting }] = useAlertDialog({
    color: 'primary',
    titleIcon: <IconInfoCircle />,
    title: 'Reconnect your nodes',
    content: (
      <Stack sx={{ gap: 7 }}>
        <Stack sx={{ gap: 1 }}>
          <Typography variant='body1'>Run join command on your new Node</Typography>
          <JoinCommand />
        </Stack>
        <Stack sx={{ gap: 1 }}>
          <Typography variant='body1'>Check your live node status</Typography>
          <Typography variant='body2' color='text.disabled'>
            After you successfully start the server on the nodes, you should see them show up on the
            below dashboard.
          </Typography>
          <NodeList />
        </Stack>
      </Stack>
    ),
    confirmLabel: 'Finish',
  });
  useEffect(() => {
    if (clusterStatus === 'waiting') {
      openWaiting();
    }
  }, [clusterStatus, openWaiting]);

  const [dialogRebalancing, { open: openRebalancing }] = useAlertDialog({
    color: 'primary',
    title: '',
    content: (
      <>
        <Typography variant='body1'>Cluster rebalancing</Typography>
        <Typography variant='body2' color='text.disabled'>
          We have noticed one of your nodes has been disconnected. We are now rebalancing your
          inference requests onto working nodes. Please wait a few seconds for the cluster to
          rebalance itself.
        </Typography>
        <NodeList variant='menu' />
      </>
    ),
    confirmLabel: 'Finish',
  });
  useEffect(() => {
    if (clusterStatus === 'rebalancing') {
      openRebalancing();
    }
  }, [clusterStatus, openRebalancing]);

  const [sidebarExpanded, setMenuOpen] = useState(true);

  const [dialogJoinCommand, { open: openJoinCommand }] = useAlertDialog({
    color: 'primary',
    titleIcon: <IconCirclePlus />,
    title: 'Add Nodes',
    content: (
      <Stack sx={{ gap: 7 }}>
        <Stack sx={{ gap: 1 }}>
          <Typography variant='body1'>Run join command on your new Node</Typography>
          <JoinCommand />
        </Stack>
        <Stack sx={{ gap: 1 }}>
          <Typography variant='body1'>Check your live node status</Typography>
          <Typography variant='body2' color='text.disabled'>
            After you successfully start the server on the nodes, you should see them show up on the
            below dashboard.
          </Typography>
          <NodeList />
        </Stack>
      </Stack>
    ),
  });

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
            <Button color='info' startIcon={<IconPlus />} onClick={openJoinCommand}>
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
      {dialogJoinCommand}
      {dialogWaiting}
      {dialogRebalancing}
    </DrawerLayoutRoot>
  );
};
