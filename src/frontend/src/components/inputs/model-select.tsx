import type { FC, ReactNode } from 'react';
import {
  InputBase,
  MenuItem,
  OutlinedInput,
  Select,
  selectClasses,
  Stack,
  styled,
  Typography,
} from '@mui/material';

import { useCluster, type ModelInfo } from '../../services';
import { useRefCallback } from '../../hooks';
import { useAlertDialog } from '../mui';
import { IconRestore } from '@tabler/icons-react';

const ModelSelectRoot = styled(Select)<{ ownerState: ModelSelectProps }>(({
  theme,
  ownerState: { variant },
}) => {
  const { spacing, typography } = theme;
  return {
    height: (variant === 'outlined' && '4rem') || '1lh',
    ...(variant === 'text' && {
      ...typography.h3,
      fontWeight: typography.fontWeightMedium,

      [`& .${selectClasses.select}`]: {
        fontSize: 'inherit',
        fontWeight: 'inherit',
        lineHeight: 'inherit',
      },
    }),
  };
});

const ModelSelectOption = styled(MenuItem)(({ theme }) => {
  const { spacing } = theme;
  return {
    height: '3.25rem',
    gap: '0.5rem',
  };
});

const ModelLogo = styled('img')(({ theme }) => {
  const { palette } = theme;
  return {
    width: '2.25rem',
    height: '2.25rem',
    borderRadius: '0.5rem',
    border: `1px solid ${palette.divider}`,
    objectFit: 'cover',
  };
});

const ModelDisplayName = styled('span')(({ theme }) => {
  const { palette, typography } = theme;
  return {
    ...typography.subtitle2,
    fontWeight: typography.fontWeightLight,
    color: palette.text.primary,
  };
});

const ModelName = styled('span')(({ theme }) => {
  const { palette, typography } = theme;
  return {
    ...typography.body2,
    fontWeight: typography.fontWeightLight,
    color: palette.text.secondary,
  };
});

const renderOption = (model: ModelInfo, selected?: boolean): ReactNode => (
  <ModelSelectOption key={model.name} value={model.name}>
    <ModelLogo src={model.logoUrl} />
    <Stack gap={0.25}>
      <ModelDisplayName>{model.displayName}</ModelDisplayName>
      <ModelName>{model.name}</ModelName>
    </Stack>
  </ModelSelectOption>
);

export interface ModelSelectProps {
  /**
   * The variant style of the select component.
   * @default 'outlined'
   */
  variant?: 'outlined' | 'text';
}

export const ModelSelect: FC<ModelSelectProps> = ({ variant = 'outlined' }) => {
  const [
    {
      modelName,
      modelInfoList,
      clusterInfo: { status: clusterStatus },
    },
    { setModelName },
  ] = useCluster();

  const [nodeDialog, { open: openDialog }] = useAlertDialog({
    titleIcon: <IconRestore />,
    title: 'Switch model',
    content: (
      <Typography variant='body2' color='text.secondary'>
        The current version of parallax only support hosting one model at once, so switching model
        will terminate your existing chat service. You may restart your current scheduler by going
        to your terminal, terminate and start the server by running parallax run. We will add node
        rebalancing and dynamic model allocation in the coming updates!
      </Typography>
    ),
    confirmLabel: 'Continue',
  });

  const onChange = useRefCallback((e) => {
    if (clusterStatus !== 'idle') {
      openDialog();
      return;
    }
    setModelName(String(e.target.value));
  });

  return (
    <>
      <ModelSelectRoot
        ownerState={{ variant }}
        input={variant === 'outlined' ? <OutlinedInput /> : <InputBase />}
        value={modelName}
        onChange={onChange}
        renderValue={(value) => {
          const model = modelInfoList.find((model) => model.name === value);
          return (
            (model && ((variant === 'outlined' && renderOption(model)) || model.name)) || undefined
          );
        }}
      >
        {modelInfoList.map((model) => renderOption(model, model.name === modelName))}
      </ModelSelectRoot>
      {nodeDialog}
    </>
  );
};
