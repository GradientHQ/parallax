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
import { useCluster, type NodeInfo, type NodeStatus } from '../../services';

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
    gap: spacing(variant === 'list' ? 1.5 : 5.5),
    overflowY: 'auto',
  };
});

const ListItem = styled(MuiListItem)(({ theme }) => {
  const { spacing } = theme;
  return {
    flex: 'none',
    padding: spacing(2),
    overflow: 'hidden',
  };
}) as typeof MuiListItem;

const ListItemIcon = styled(MuiListItemIcon)(({ theme }) => {
  return {
    fontSize: '1.5rem',
    width: '2.75rem',
    height: '2.75rem',
    borderRadius: '50%',
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
        backgroundColor: 'transparent',
        gap: 1,
      }}
    >
      <IconDevices2 size={'1.5rem'}/>

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
          {variant === 'list' && <IconStatus />}
          {variant === 'menu' && <IconCircleFilled />}
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

  const { length: nodesNumber } = nodeInfoList;
  // const nodesNumber = 0;

  return (
    <NodeListRoot>
      {variant === 'menu' && (
        <Box
          sx={{
            position: 'absolute',
            top: '1.375rem',
            bottom: '1.375rem',
            left: '1.375rem',
            borderLeft: '2px dashed',
            borderColor: 'divider',
          }}
        />
      )}
      <List variant={variant}>
        {nodeInfoList.map((node) => (
          <Node key={node.id} variant={variant} node={node} />
        ))}
        {initNodesNumber > nodesNumber
          && Array.from({ length: initNodesNumber - nodesNumber }).map((_, index) => (
            <Node key={index} variant={variant} />
          ))}
      </List>
    </NodeListRoot>
  );
};
