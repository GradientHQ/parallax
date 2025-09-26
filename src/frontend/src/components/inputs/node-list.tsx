import type { FC, ForwardRefExoticComponent, RefAttributes } from 'react';
import * as motion from 'motion/react-client';
import {
  IconCheck,
  IconCircle,
  IconCircleFilled,
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

const IconNode = () => (
  <svg width='1em' height='1em' viewBox='0 0 27 27'>
    <path
      d='M20.0235 2.88477H6.92829C6.49415 2.88477 6.0778 3.05722 5.77082 3.3642C5.46384 3.67118 5.29138 4.08754 5.29138 4.52167V22.5276C5.29138 22.9618 5.46384 23.3781 5.77082 23.6851C6.0778 23.9921 6.49415 24.1645 6.92829 24.1645H20.0235C20.4577 24.1645 20.874 23.9921 21.181 23.6851C21.488 23.3781 21.6604 22.9618 21.6604 22.5276V4.52167C21.6604 4.08754 21.488 3.67118 21.181 3.3642C20.874 3.05722 20.4577 2.88477 20.0235 2.88477ZM13.4759 20.0723C13.2331 20.0723 12.9957 20.0003 12.7938 19.8654C12.592 19.7305 12.4346 19.5387 12.3417 19.3144C12.2488 19.0901 12.2244 18.8432 12.2718 18.6051C12.3192 18.3669 12.4361 18.1482 12.6078 17.9765C12.7795 17.8048 12.9983 17.6879 13.2364 17.6405C13.4745 17.5931 13.7214 17.6174 13.9457 17.7104C14.17 17.8033 14.3618 17.9606 14.4967 18.1625C14.6316 18.3644 14.7036 18.6018 14.7036 18.8446C14.7036 19.1702 14.5742 19.4825 14.344 19.7127C14.1138 19.9429 13.8015 20.0723 13.4759 20.0723ZM16.7497 11.8877H10.2021C9.98503 11.8877 9.77685 11.8015 9.62336 11.648C9.46987 11.4945 9.38364 11.2864 9.38364 11.0693C9.38364 10.8522 9.46987 10.644 9.62336 10.4906C9.77685 10.3371 9.98503 10.2508 10.2021 10.2508H16.7497C16.9668 10.2508 17.175 10.3371 17.3285 10.4906C17.4819 10.644 17.5682 10.8522 17.5682 11.0693C17.5682 11.2864 17.4819 11.4945 17.3285 11.648C17.175 11.8015 16.9668 11.8877 16.7497 11.8877ZM16.7497 8.61393H10.2021C9.98503 8.61393 9.77685 8.5277 9.62336 8.37421C9.46987 8.22072 9.38364 8.01255 9.38364 7.79548C9.38364 7.57841 9.46987 7.37024 9.62336 7.21675C9.77685 7.06326 9.98503 6.97703 10.2021 6.97703H16.7497C16.9668 6.97703 17.175 7.06326 17.3285 7.21675C17.4819 7.37024 17.5682 7.57841 17.5682 7.79548C17.5682 8.01255 17.4819 8.22072 17.3285 8.37421C17.175 8.5277 16.9668 8.61393 16.7497 8.61393Z'
      fill='currentColor'
    />
  </svg>
);

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
      sx={{ opacity, padding: variant === 'menu' ? 0 : undefined }}
    >
      <ListItemIcon
        sx={{
          color: main,
          backgroundColor: lighter,
        }}
      >
        <IconNode />
      </ListItemIcon>

      <ListItemText>
        {(node && (
          <Typography variant='body1'>
            {gpuName} {gpuMemory}GB
          </Typography>
        )) || <Skeleton width='8rem' height='0.75rem' sx={{ my: 0.5 }} />}
        {(node && (
          <Typography
            variant='body1'
            color='text.disabled'
            overflow='hidden'
            textOverflow='ellipsis'
            whiteSpace='nowrap'
          >
            {id}
          </Typography>
        )) || <Skeleton width='14rem' height='0.75rem' sx={{ my: 0.5 }} />}
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

  return (
    <NodeListRoot>
      {/* <Alert key='info' severity='info' variant='standard'>
        If your nodes cannot connect properly, retry the above join command to restart the server.
      </Alert>
      <Alert key='error' severity='error' variant='standard'>
        Your selected model requires more nodes. Please go back to the previous step to add more
        nodes, or choose a smaller model.
      </Alert> */}
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
        {[...nodeInfoList, ...nodeInfoList].map((node) => (
          <Node key={node.id} variant={variant} node={node} />
        ))}
      </List>
    </NodeListRoot>
  );
};
