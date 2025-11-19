import type { FC, ForwardRefExoticComponent, RefAttributes } from 'react';
import * as motion from 'motion/react-client';
import {
  IconCheck,
  IconCircleFilled,
  IconDevices2,
  IconLoader,
  IconX,
  type Icon,
  type IconProps,
} from '@tabler/icons-react';
import {
  Alert,
  List as MuiList,
  ListItem as MuiListItem,
  ListItemIcon as MuiListItemIcon,
  ListItemText as MuiListItemText,
  MenuList,
  Paper,
  Skeleton,
  styled,
  Typography,
  useTheme,
  Stack,
  Box,
  Divider,
  type ListProps,
  type StackProps,
} from '@mui/material';
import { useChat, useCluster, type NodeInfo, type NodeStatus } from '../../services';

const NodeListRoot = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    position: 'relative',
    flex: 1,
    gap: spacing(1.5),
    overflowX: 'hidden',
    overflowY: 'auto',
  };
});

const List = styled(MuiList)<{ variant: NodeListVariant }>(({ theme, variant }) => {
  const { spacing } = theme;
  return {
    // menu no need gap, use dash line to separate nodes
    gap: spacing(variant === 'list' ? 1.5 : 0),
  };
});

const ListItem = styled(MuiListItem)(({ theme }) => {
  const { spacing } = theme;
  return {
    flex: 'none',
    gap: spacing(1),
    backgroundColor: 'transparent',
    padding: spacing(2),
    overflow: 'visible',
  };
}) as typeof MuiListItem;

const ListItemIcon = styled(MuiListItemIcon)(({ theme }) => {
  return {
    color: 'inherit',
    fontSize: '1.5rem',
    width: '1em',
    height: '1em',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
  };
}) as typeof MuiListItemIcon;

const ListItemText = styled(MuiListItemText)(({ theme }) => {
  return {
    position: 'relative',
    display: 'block',
    height: '100%',
  };
}) as typeof MuiListItemText;

const ListItemStatus = styled(motion.div)<{ variant: NodeListVariant }>(({ theme, variant }) => {
  return {
    fontSize: variant === 'list' ? '1.5rem' : '1em',
    width: '1em',
    height: '1em',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    transformOrigin: 'center',
  };
});

const STATUS_COLOR_MAP: Record<NodeStatus, 'info' | 'success' | 'error'> = {
  waiting: 'info',
  available: 'success',
  failed: 'error',
};

const STATUS_ICON_MAP: Record<
  NodeStatus,
  ForwardRefExoticComponent<IconProps & RefAttributes<Icon>>
> = {
  waiting: IconLoader,
  available: IconCheck,
  failed: IconX,
};

const DashRoot = styled(Box)(({ theme }) => {
  const { spacing } = theme;
  return {
    position: 'relative',
    width: '1.5rem',
    height: '3.25rem', // For dash array last position, must to be minus 0.25rem(4px)
    overflow: 'hidden',
  };
});

const Dash: FC<{ animate?: boolean }> = ({ animate }) => {
  const width = 2;
  const height = 256;
  return (
    <DashRoot>
      <svg
        style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)' }}
        width={width}
        height={height}
        viewBox={`0 0 2 ${height}`}
        fill='none'
      >
        <line x1='1' y1='0' x2='1' y2={height} stroke='#9B9B9B' strokeWidth='2' strokeDasharray='4'>
          {animate && (
            <animate
              attributeName='stroke-dashoffset'
              from={height}
              to='0'
              dur={`${height / 32}s`}
              repeatCount='indefinite'
            ></animate>
          )}
        </line>
      </svg>
    </DashRoot>
  );
};

const Node: FC<{ variant: NodeListVariant; node?: NodeInfo }> = ({ variant, node }) => {
  const { id, status, gpuNumber, gpuName, gpuMemory } = node || { status: 'waiting' };
  const { palette } = useTheme();
  const { main, lighter } =
    status === 'waiting' ?
      { main: palette.grey[800], lighter: palette.grey[250] }
    : palette[STATUS_COLOR_MAP[status]];
  const opacity = status === 'failed' ? 0.2 : undefined;

  const IconStatus = STATUS_ICON_MAP[status];

  return (
    <ListItem
      component={variant === 'list' ? Paper : Box}
      variant='outlined'
      sx={{
        opacity,
        padding: variant === 'menu' ? 0 : undefined,
        height: variant === 'menu' ? '2.5rem' : undefined,
      }}
    >
      <ListItemIcon>
        <IconDevices2 />
      </ListItemIcon>

      <ListItemText>
        <Stack
          sx={
            variant === 'menu' ?
              {
                position: 'absolute',
                top: '50%',
                left: 0,
                right: 0,
                transform: 'translateY(-50%)',
              }
            : undefined
          }
        >
          {(node && (
            <>
              <Typography variant='body1' sx={{ fontWeight: 500 }}>
                {[
                  (gpuNumber && gpuNumber > 1 && `${gpuNumber}x`) || '',
                  gpuName,
                  (gpuMemory && `${gpuMemory}GB`) || '',
                ]
                  .filter(Boolean)
                  .join(' ')}
              </Typography>
              {/* <Typography variant='caption' color='text.disabled'>
                Rancho Cordova, United States
              </Typography> */}
            </>
          )) || (
            <>
              <Skeleton height='1lh' />
            </>
          )}
        </Stack>
      </ListItemText>

      {node && (
        <ListItemStatus
          sx={{ color: main }}
          {...(status === 'waiting' && {
            animate: { rotate: 360 },
            transition: {
              repeat: Infinity,
              ease: 'linear',
              duration: 2,
            },
          })}
          variant={variant}
        >
          {variant === 'list' && <IconStatus size={18} />}
          {variant === 'menu' && <IconCircleFilled size={10} />}
        </ListItemStatus>
      )}
    </ListItem>
  );
};

export type NodeListVariant = 'list' | 'menu';

export interface NodeListProps {
  variant?: NodeListVariant;
}

export const NodeList: FC<NodeListProps & StackProps> = ({ variant = 'list', ...rest }) => {
  const [
    {
      clusterInfo: { status: clusterStatus, initNodesNumber },
      nodeInfoList,
    },
  ] = useCluster();
  const [{ status: chatStatus }] = useChat();

  const { length: nodesNumber } = nodeInfoList;
  // const nodesNumber = 0;

  const generating = chatStatus === 'generating';
  let dashIndex = 0;
  const renderDash =
    variant === 'menu' ?
      (key: string) => {
        return dashIndex++ > 0 && <Dash key={key} animate={generating} />;
      }
    : () => undefined;

  return (
    <NodeListRoot {...rest}>
      <List variant={variant}>
        {nodeInfoList.map((node, index) => [
          renderDash(`${node.id}-dash`),
          <Node key={node.id} variant={variant} node={node} />,

          // renderDash(`${node.id}-dash-mock-0`),
          // <Node key={`${node.id}-mock-0`} variant={variant} node={node} />,

          // renderDash(`${node.id}-dash-mock-1`),
          // <Node key={`${node.id}-mock-1`} variant={variant} node={node} />,

          // renderDash(`${node.id}-dash-mock-2`),
          // <Node key={`${node.id}-mock-2`} variant={variant} node={node} />,
        ])}

        {clusterStatus !== 'idle'
          && initNodesNumber > nodesNumber
          && Array.from({ length: initNodesNumber - nodesNumber }).map((_, index) => [
            renderDash(`${index}-dash`),
            <Node key={index} variant={variant} />,
          ])}
      </List>
    </NodeListRoot>
  );
};
