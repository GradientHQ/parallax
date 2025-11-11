import { useEffect, useState, type FC, type PropsWithChildren } from 'react';
import {
  Box,
  Button,
  Divider,
  IconButton,
  Stack,
  styled,
  Tooltip,
  Typography,
} from '@mui/material';
import { useCluster, useHost } from '../../services';
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
  IconTopologyStar3,
} from '@tabler/icons-react';
import { JoinCommand, ModelSelect, NodeList } from '../inputs';

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
    paddingBlock: spacing(2),
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
    height: '2.5rem',
    flex: 'none',
    marginTop: spacing(1),
    paddingBlock: spacing(1),
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
    gap: spacing(2),
    paddingBlock: spacing(1),
    paddingInline: spacing(4),
    overflow: 'hidden',
  };
});

export const DrawerLayout: FC<PropsWithChildren> = ({ children }) => {
  const [{ type: hostType }] = useHost();

  const [
    {
      modelInfo,
      clusterInfo: { status: clusterStatus, needMoreNodes },
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
    if (hostType === 'cluster' && clusterStatus === 'waiting') {
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

  const [dialogNeedMoreNodes, { open: openDialogNeedMoreNodes }] = useAlertDialog({
    color: 'primary',
    title: '',
    content: (
      <>
        <Typography variant='body1'>
          Your selected model requires more nodes.
          {(!!modelInfo
            && modelInfo.vram > 0 && [
              `You’ll need a `,
              <strong>{`minimum of ${modelInfo.vram} GB of total VRAM`}</strong>,
              ` to host this model.`,
            ])
            || ''}
        </Typography>
      </>
    ),
    confirmLabel: 'Finish',
  });
  useEffect(() => {
    if (needMoreNodes) {
      openDialogNeedMoreNodes();
    }
  }, [needMoreNodes, openDialogNeedMoreNodes]);

  const [dialogFailed, { open: openFailed }] = useAlertDialog({
    color: 'primary',
    title: '',
    content: (
      <>
        <Typography variant='body1'>Scheduler restart</Typography>
        <Typography variant='body2' color='text.disabled'>
          We have noticed that your scheduler has been disconnected (this would be the computer that
          ran the <strong>parallax run</strong> command). You would need to restart the scheduler,
          reconfigure the cluster, and your chat will be back up again!
        </Typography>
      </>
    ),
    confirmLabel: 'Finish',
  });
  useEffect(() => {
    if (clusterStatus === 'failed') {
      openFailed();
      return;
    }
    if (clusterStatus === 'idle') {
      // Delay trigger, due to the cluster init status is 'idle' before connecting to the scheduler.
      const timeoutId = setTimeout(() => openFailed(), 1000);
      return () => clearTimeout(timeoutId);
    }
  }, [clusterStatus, openFailed]);

  const [sidebarExpanded, setMenuOpen] = useState(true);

  const [dialogJoinCommand, { open: openJoinCommand }] = useAlertDialog({
    color: 'primary',
    titleIcon: <IconCirclePlus />,
    title: 'Add New Nodes',
    content: (
      <Stack sx={{ gap: 5 }}>
        <Stack sx={{ gap: 1 }}>
          <Typography variant='body1'>Run join command on all nodes</Typography>
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

  const IconCluster = () => (
    <svg width='1.5rem' height='1.5rem' viewBox='0 0 27 27' fill='currentColor'>
      <g
        fill='none'
        stroke='currentColor'
        stroke-linecap='round'
        stroke-linejoin='round'
        stroke-width='2'
      >
        <rect width='6' height='6' x='16' y='16' rx='1' />
        <rect width='6' height='6' x='2' y='16' rx='1' />
        <rect width='6' height='6' x='9' y='2' rx='1' />
        <path d='M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3m-7-4V8' />
      </g>
    </svg>
  );

  return (
    <DrawerLayoutRoot direction='row'>
      <DrawerLayoutSide
        sx={{
          width: sidebarExpanded ? '16.25rem' : '3.5rem',
          paddingInline: sidebarExpanded ? 2 : 2,
        }}
      >
        <Stack direction='row' sx={{ justifyContent: 'flex-end', alignItems: 'center', gap: 2 }}>
          {sidebarExpanded ?
            <>
              <IconBrandGradient />
              <Box sx={{ flex: 1 }} />
              <Tooltip
                title='Collapse Sidebar'
                placement='right'
                slotProps={{
                  tooltip: { sx: { bgcolor: 'primary.main', color: 'common.white' } },
                }}
              >
                <IconButton
                  size='em'
                  sx={{
                    fontSize: '1.5rem',
                    borderRadius: '8px',
                    color: '#808080FF',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                  onClick={() => setMenuOpen((prev) => !prev)}
                >
                  <IconLayoutSidebarLeftCollapse />
                </IconButton>
              </Tooltip>
            </>
          : <>
              <Box
                sx={{
                  position: 'relative',
                  width: 28,
                  height: 28,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  '&:hover .logo': { opacity: 0 },
                  '&:hover .toggle': { opacity: 1, pointerEvents: 'auto', transform: 'scale(1)' },
                }}
              >
                <Box
                  className='logo'
                  sx={{
                    position: 'absolute',
                    inset: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'opacity .15s ease',
                    opacity: 1,
                  }}
                >
                  <IconBrandGradient />
                </Box>

                <Tooltip
                  title='Expand Sidebar'
                  placement='right'
                  slotProps={{
                    tooltip: { sx: { bgcolor: 'primary.main', color: 'common.white' } },
                  }}
                >
                  <IconButton
                    className='toggle'
                    size='em'
                    sx={{
                      position: 'absolute',
                      opacity: 0,
                      pointerEvents: 'none',
                      fontSize: '1.5rem',
                      transition: 'opacity .15s ease, transform .15s ease',
                      '&:hover': { bgcolor: 'action.hover' },
                    }}
                    aria-label='Expand Sidebar'
                    onClick={() => setMenuOpen((prev) => !prev)}
                  >
                    <IconLayoutSidebarLeftExpand />
                  </IconButton>
                </Tooltip>
              </Box>
            </>
          }
        </Stack>
        {sidebarExpanded && (
          <Stack>
            <Stack direction='row' sx={{ gap: 1, color: 'text.primary' }}>
              {/* <IconCluster /> */}
              <Typography variant='body1' sx={{ mt: '1.5px', color: '#A7A7A7FF', fontWeight: 600 }}>
                Cluster topology
              </Typography>
            </Stack>
            <NodeList variant='menu' sx={{ py: '2rem' }} />
            <Button
              color='info'
              startIcon={<IconPlus />}
              onClick={openJoinCommand}
              // onClick={openRebalancing}
              // onClick={openWaiting}
            >
              Add Nodes
            </Button>
          </Stack>
        )}
      </DrawerLayoutSide>
      <DrawerLayoutContainer>
        <DrawerLayoutHeader direction='row'>
          <ModelSelect variant='text' />
        </DrawerLayoutHeader>
        <DrawerLayoutContent>{children}</DrawerLayoutContent>
      </DrawerLayoutContainer>
      {dialogJoinCommand}
      {dialogWaiting}
      {dialogRebalancing}
      {dialogFailed}
      {dialogNeedMoreNodes}
    </DrawerLayoutRoot>
  );
};
