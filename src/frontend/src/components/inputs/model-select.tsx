import type { ReactNode } from 'react';
import { MenuItem, Select, Stack, styled } from '@mui/material';
import { useCluster, type ModelInfo } from '../../services';

const ModelSelectRoot = styled(Select)(({ theme }) => ({
  height: '4rem',
  paddingInline: theme.spacing(0.5),
  '&:hover': { bg: 'action.hover' },
}));

const ModelSelectOption = styled(MenuItem)(({ theme }) => ({
  height: '3.25rem',
  gap: '0.5rem',
}));

const ValueRow = styled(Stack)(({ theme }) => ({
  flexDirection: 'row',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(1),
  '&:hover': { backgroundColor: 'transparent' },
  pointerEvents: 'none',
}));

const ModelLogo = styled('img')(({ theme }) => ({
  width: '2.25rem',
  height: '2.25rem',
  borderRadius: '0.5rem',
  border: `1px solid ${theme.palette.divider}`,
  objectFit: 'cover',
}));

const ModelDisplayName = styled('span')(({ theme }) => ({
  ...theme.typography.subtitle2,
  fontSize: '0.875rem', 
  lineHeight: '1rem',
  fontWeight: theme.typography.fontWeightLight,
  color: theme.palette.text.primary,
}));

const ModelName = styled('span')(({ theme }) => ({
  ...theme.typography.body2,
  fontSize: '0.75rem',
  fontWeight: theme.typography.fontWeightLight,
  color: theme.palette.text.secondary,
}));

const renderOption = (model: ModelInfo): ReactNode => (
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
        const model = modelInfoList.find((m) => m.name === value);
        if (!model) return undefined;
        return (
          <ValueRow>
            <ModelLogo src={model.logoUrl} />
            <Stack gap={0.25}>
              <ModelDisplayName>{model.displayName}</ModelDisplayName>
              <ModelName>{model.name}</ModelName>
            </Stack>
          </ValueRow>
        );
      }}
    >
      {modelInfoList.map((model) => renderOption(model))}
    </ModelSelectRoot>
  );
};
