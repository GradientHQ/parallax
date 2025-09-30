import { useState } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import {
  Button,
  ButtonGroup,
  FormControl,
  FormControlLabel,
  FormLabel,
  MenuItem,
  Select,
  Stack,
  styled,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import { MainLayout } from '../components/common';
import { ModelSelect, NumberInput } from '../components/inputs';
import { useCluster } from '../services';
import { useRefCallback } from '../hooks';

export default function PageSetup() {
  const [
    { networkType, initNodesNumber, modelName, modelInfoList },
    { setNetworkType, setInitNodesNumber, setModelName, init },
  ] = useCluster();

  const navigate = useNavigate();

  const [loading, setLoading] = useState(false);

  const onContinue = useRefCallback(async () => {
    setLoading(true);
    Promise.resolve()
      .then(() => init())
      .then(() => navigate('/join'))
      .catch((e) => console.error(e))
      .finally(() => setLoading(false));
  });

  return (
    <MainLayout>
      <Typography variant='h1'>Build Your Own AI Cluster</Typography>

      <Stack gap={2.5}>
        <Stack gap={0.5}>
          <Typography variant='body1'>Step 1 - Specify the initial number of nodes</Typography>
          <Typography variant='body2' color='text.secondary' fontWeight='regular'>
            Parallax runs and hosts model distributedly on your everyday hardware. Select the number
            of nodes you would like to add to your cluster with their connection types.{' '}
          </Typography>
        </Stack>

        <Stack direction='row' justifyContent='space-between' alignItems='center' gap={2}>
          <Typography color='text.secondary'>Node Number</Typography>
          <NumberInput
            sx={{ width: '10rem', boxShadow: 'none', bgcolor: 'transparent' }}
            slotProps={{
              root: {
                sx: {
                  bgcolor: 'transparent',
                  '&:hover': { bgcolor: 'transparent' },
                  '&:focus-within': { bgcolor: 'transparent' },
                },
              },
              input: {
                sx: {
                  bgcolor: 'transparent !important',
                  '&:focus': { outline: 'none' },
                },
              },
            }}
            value={initNodesNumber}
            onChange={(e) => setInitNodesNumber(Number(e.target.value))}
          />
        </Stack>

        <Stack direction='row' justifyContent='space-between' alignItems='center' gap={2}>
          <Typography color='text.secondary'>
            Are you nodes within the same local network?
          </Typography>
          <ToggleButtonGroup
            sx={{ width: '10rem', textTransform: 'none' }}
            exclusive
            value={networkType}
            onChange={(_, value) => value && setNetworkType(value)}
          >
            <ToggleButton value='local' sx={{ textTransform: 'none' }}>
              Local
            </ToggleButton>
            <ToggleButton value='remote' sx={{ textTransform: 'none' }}>
              Remote
            </ToggleButton>
          </ToggleButtonGroup>
        </Stack>
      </Stack>

      <Stack gap={2.5}>
        <Stack gap={0.5}>
          <Typography variant='body1'>Step 2 - Select the model you would like to host</Typography>
          <Typography variant='body2' color='text.secondary' fontWeight='regular'>
            Currently we support a handful of state-of-the-art open source models. Do keep in mind
            that larger models require more nodes to host, so If this is your first time trying
            Parallax, we suggest you to start with smaller models.
          </Typography>
        </Stack>

        <ModelSelect />
      </Stack>

      <Stack direction='row' justifyContent='flex-end' alignItems='center' gap={2}>
        <Button loading={loading} onClick={onContinue}>
          Continue
        </Button>
      </Stack>
    </MainLayout>
  );
}
