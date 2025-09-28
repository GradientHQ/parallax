import type { ReactNode } from 'react';
import { MenuItem, Select, Stack, styled, Typography } from '@mui/material';

import { useCluster, type ModelInfo } from '../../services';

const ModelSelectRoot = styled(Select)(({ theme }) => {
  const { spacing } = theme;
  return {
    height: '4rem',
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

export const ModelSelect = () => {
  const [{ modelName, modelInfoList }, { setModelName }] = useCluster();

  return (
    <ModelSelectRoot
      value={modelName}
      onChange={(e) => setModelName(String(e.target.value))}
      renderValue={(value) => {
        const model = modelInfoList.find((model) => model.name === value);
        return (model && renderOption(model)) || undefined;
      }}
    >
      {modelInfoList.map((model) => renderOption(model, model.name === modelName))}
    </ModelSelectRoot>
  );
};
