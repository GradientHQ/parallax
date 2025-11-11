import { useMemo, useState } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  Alert,
  Button,
  ButtonGroup,
  FormControl,
  FormControlLabel,
  FormLabel,
  MenuItem,
  Select,
  Stack as MuiStack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  styled,
} from '@mui/material';
import { IconArrowLeft } from '@tabler/icons-react';
import { MainLayout } from '../components/common';
import { JoinCommand, ModelSelect, NodeList } from '../components/inputs';
import { useCluster } from '../services';

const Stack = styled(MuiStack)(({ theme }) => {
  const { spacing } = theme;
  return {
    overflowY: 'auto',
  };
});

export default function PageJoin() {
  const [
    {
      modelInfo,
      clusterInfo: { status: clusterStatus, initNodesNumber, needMoreNodes },
      nodeInfoList,
    },
  ] = useCluster();

  const isError = useMemo(() => {
    if (
      initNodesNumber > 0
      && nodeInfoList.length >= initNodesNumber
      && nodeInfoList.every((node) => node.status === 'available')
      && clusterStatus === 'waiting'
    ) {
      return true;
    }
    return false;
  }, [clusterStatus, initNodesNumber, nodeInfoList]);

  return (
    <MainLayout
      contentStart={
        <Button
          component={RouterLink}
          to='/setup'
          size='medium'
          color='secondary'
          startIcon={<IconArrowLeft />}
        >
          Back
        </Button>
      }
    >
      <Typography variant='h1'>Get Your Nodes Running</Typography>

      <Stack gap={6} sx={{ overflow: 'hidden' }}>
        <Stack gap={2}>
          <Stack gap={1}>
            <Typography variant='body1'>Step 1 - Run join command on all nodes</Typography>
          </Stack>
          <JoinCommand />
        </Stack>

        <Stack gap={2} flex={1}>
          <Stack gap={1}>
            <Typography variant='body1'>Step 2 - Check your node status</Typography>
            <Typography variant='body2' color='text.secondary' fontWeight='regular'>
              After you successfully start your nodes, you should see them start to show up below
              with their status. Once all nodes are connected, you will automatically be directed to
              the chat interface.
            </Typography>
          </Stack>

          {(isError && (
            <Alert key='error' severity='error' variant='standard'>
              Your selected model requires more nodes. Please go back to the previous step to add
              more nodes, or choose a smaller model.
            </Alert>
          )) || (
            <Alert key='info' severity='info' variant='standard'>
              If your nodes cannot connect properly, retry the above join command to restart the
              server.
            </Alert>
          )}

          {!!modelInfo && modelInfo.vram > 0 && needMoreNodes && (
            <Alert key='vram-warning' severity='warning' variant='standard'>
              <Typography variant='inherit'>
                {[
                  `Your selected model requires more nodes.`,
                  `You’ll need a `,
                  <strong>{`minimum of ${modelInfo.vram} GB of total VRAM`}</strong>,
                  ` to host this model.`,
                ]}
              </Typography>
            </Alert>
          )}

          <NodeList key='node-list' />
        </Stack>
      </Stack>
    </MainLayout>
  );
}
