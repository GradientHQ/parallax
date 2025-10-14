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
  ListItemText,
  MenuList,
  Paper,
  Skeleton,
  styled,
  Typography,
  useTheme,
  Stack,
  Box,
  Divider,
} from '@mui/material';
import { useChat, useCluster, type NodeInfo, type NodeStatus } from '../../services';

const NodeListRoot = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    position: 'relative',
    flex: 1,
    gap: spacing(1.5),
    overflow: 'hidden',
  };
});

const List = styled(MuiList)<{ variant: NodeListVariant }>(({ theme, variant }) => {
  const { spacing } = theme;
  return {
    // menu no need gap, use dash line to separate nodes
    gap: spacing(variant === 'list' ? 1.5 : 0),
    overflowY: 'auto',
  };
});

const ListItem = styled(MuiListItem)(({ theme }) => {
  const { spacing } = theme;
  return {
    flex: 'none',
    gap: spacing(1),
    backgroundColor: 'transparent',
    padding: spacing(2),
    overflow: 'hidden',
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
    height: '2.75rem', // For dash array last position, must to be minus 0.25rem(4px)
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
  const { id, status, gpuName, gpuMemory } = node || { status: 'waiting' };
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
      }}
    >
      <ListItemIcon>
        <IconDevices2 />
      </ListItemIcon>

      <ListItemText>
        {(node && (
            <Typography variant='body1' sx={{ fontWeight: 500 }}>
              {gpuName} {gpuMemory}GB
            </Typography>
        )) || <Skeleton width='8rem' height='1.25rem' />}
        {/* {(node && (
          <Typography
            variant='body2'
            color='text.disabled'
            overflow='hidden'
            textOverflow='ellipsis'
            whiteSpace='nowrap'
          >
            {id && id.substring(0, 4) + '...' + id.substring(id.length - 4)}
          </Typography>
        )) || <Skeleton width='14rem' height='1.25rem' />} */}
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

export const NodeList: FC<NodeListProps> = ({ variant = 'list' }) => {
  const [
    {
      clusterInfo: { initNodesNumber },
      nodeInfoList,
    },
  ] = useCluster();
  const [{ status: chatStatus }] = useChat();

  const { length: nodesNumber } = nodeInfoList;
  // const nodesNumber = 0;

  return (
    <NodeListRoot>
      <List variant={variant}>
        {nodeInfoList.map((node, index) => [
          variant === 'menu' && index > 0 && (
            <Dash key={`${node.id}-dash`} animate={chatStatus === 'generating'} />
          ),
          <Node key={node.id} variant={variant} node={node} />,
          <Dash key={`${node.id}-dash-mock-0`} animate={chatStatus === 'generating'} />,
          <Node key={`${node.id}-mock-0`} variant={variant} node={node} />,
          <Dash key={`${node.id}-dash-mock-1`} animate={chatStatus === 'generating'} />,
          <Node key={`${node.id}-mock-1`} variant={variant} node={node} />,
          <Dash key={`${node.id}-dash-mock-2`} animate={chatStatus === 'generating'} />,
          <Node key={`${node.id}-mock-2`} variant={variant} node={node} />,
        ])}
        {initNodesNumber > nodesNumber
          && Array.from({ length: initNodesNumber - nodesNumber }).map((_, index) => (
            <Node key={index} variant={variant} />
          ))}
      </List>
    </NodeListRoot>
  );
};
