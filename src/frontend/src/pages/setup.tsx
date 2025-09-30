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

  const onContinue = useRefCallback(async () => {
    await init();
    await navigate('/join');
  });

  return (
    <MainLayout>
      <Typography variant='h1'>Build Your Own AI Cluster</Typography>

      <Stack gap={2.5}>
        <Stack gap={0.5}>
          <Typography variant='body1'>
            Step 1 - Specify the initial number of worker nodes to join
          </Typography>
          <Typography variant='body2' color='text.secondary' fontWeight='regular'>
            XXX Explainations here
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
            <ToggleButton value='local' sx={{ textTransform: 'none' }}>Local</ToggleButton>
            <ToggleButton value='remote' sx={{ textTransform: 'none' }}>Remote</ToggleButton>
          </ToggleButtonGroup>
        </Stack>
      </Stack>

      <Stack gap={2.5}>
        <Stack gap={0.5}>
          <Typography variant='body1'>Step 2 - Select the model you would like to host</Typography>
          <Typography variant='body2' color='text.secondary' fontWeight='regular'>
            XXX Explainations here
          </Typography>
        </Stack>

        <ModelSelect />
      </Stack>

      <Stack direction='row' justifyContent='flex-end' alignItems='center' gap={2}>
        <Button onClick={onContinue}>Continue</Button>
      </Stack>
    </MainLayout>
  );
}
