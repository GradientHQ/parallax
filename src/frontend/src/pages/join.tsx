import { useState } from 'react';
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
    { networkType, initNodesNumber, modelName, modelInfoList },
    { setNetworkType, setInitNodesNumber, setModelName },
  ] = useCluster();

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
            <Typography variant='body1'>
              Step 1 - On all worker nodes, run command line join
            </Typography>
          </Stack>
          <JoinCommand />
        </Stack>

        <Stack gap={2} flex={1}>
          <Stack gap={1}>
            <Typography variant='body1'>Step 2 - Check your live node status</Typography>
            <Typography variant='body2' color='text.secondary' fontWeight='regular'>
              After you successfully start the server on each nodes, you should see them show up on
              the below dashboard.
            </Typography>
          </Stack>

          <Alert key='info' severity='info' variant='standard'>
            If your nodes cannot connect properly, retry the above join command to restart the
            server.
          </Alert>

          <NodeList key='node-list' />
        </Stack>
      </Stack>
    </MainLayout>
  );
}
