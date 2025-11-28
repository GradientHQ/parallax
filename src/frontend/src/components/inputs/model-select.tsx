import { useEffect, useState, type FC, type ReactNode } from 'react';
import * as motion from 'motion/react-client';
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
import { useCluster, useHost, type ModelInfo } from '../../services';
import { useRefCallback } from '../../hooks';
import { useAlertDialog } from '../mui';
import { IconCheck, IconLoader, IconRestore } from '@tabler/icons-react';

const ModelSelectRoot = styled(Select)<{ ownerState: ModelSelectProps }>(({
  theme,
  ownerState,
}) => {
  const { spacing, typography, palette } = theme;
  const { variant = 'outlined' } = ownerState;

  return {
    height: variant === 'outlined' ? '4rem' : '1lh',
    paddingInline: spacing(0.5),
    borderRadius: 12,
    '&:hover': {
      backgroundColor: 'action.hover',
    },

    [`.${selectClasses.select}:hover`]: {
      backgroundColor: 'transparent',
    },

    ...(variant === 'text' && {
      ...typography.h3,
      fontWeight: typography.fontWeightMedium,
      [`.${selectClasses.select}`]: {
        fontSize: 'inherit',
        fontWeight: 'inherit',
        lineHeight: 'inherit',
        padding: 0,
      },
      '&:hover': { backgroundColor: 'transparent' },
    }),
  };
});

const ModelSelectOption = styled(MenuItem)(({ theme }) => ({
  height: '3.25rem',
  gap: theme.spacing(1),
  borderRadius: 10,
}));

const ValueRow = styled(Stack)(({ theme }) => ({
  flexDirection: 'row',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(1),
  '&:hover': { backgroundColor: 'transparent' },
  pointerEvents: 'none',
}));

const ModelExtraStatus = styled(motion.div)(({ theme }) => ({
  width: '1rem',
  height: '1rem',
  '& > .tabler-icon': {
    width: '1rem',
    height: '1rem',
  },
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
  lineHeight: '1.125rem',
  fontWeight: theme.typography.fontWeightLight,
  color: theme.palette.text.primary,
}));

const ModelName = styled('span')(({ theme }) => ({
  ...theme.typography.body2,
  fontSize: '0.75rem',
  lineHeight: '1rem',
  fontWeight: theme.typography.fontWeightLight,
  color: theme.palette.text.secondary,
}));

const renderOption = (
  { name, displayName, logoUrl }: ModelInfo,
  { selected, loading, disabled }: { selected?: boolean; loading?: boolean; disabled?: boolean },
): ReactNode => (
  <ModelSelectOption key={name} value={name}>
    <ModelExtraStatus
      {...(loading && {
        animate: { rotate: 360 },
        transition: {
          repeat: Infinity,
          ease: 'linear',
          duration: 2,
        },
      })}
    >
      {(loading && <IconLoader />) || (selected && <IconCheck />)}
    </ModelExtraStatus>
    <ModelLogo src={logoUrl} />
    <Stack gap={0.25}>
      <ModelDisplayName>{displayName}</ModelDisplayName>
      <ModelName>{name}</ModelName>
    </Stack>
  </ModelSelectOption>
);

export interface ModelSelectProps {
  variant?: 'outlined' | 'text';
  autoCommit?: boolean;
}

export const ModelSelect: FC<ModelSelectProps> = ({ variant = 'outlined', autoCommit = false }) => {
  const [{ type: hostType }] = useHost();
  const [
    {
      config: { modelName: configModelName, modelInfoList },
      clusterInfo: { status: clusterStatus, modelName: clusterModelName },
    },
    {
      config: { setModelName },
      init,
    },
  ] = useCluster();

  // const [nodeDialog, { open: openDialog }] = useAlertDialog({
  //   titleIcon: <IconRestore />,
  //   title: 'Switch model',
  //   content: (
  //     <Typography variant='body2' color='text.secondary'>
  //       The current version of parallax only supports hosting one model at once. Switching the model
  //       will terminate your existing chat service. You can restart the current scheduler in your
  //       terminal. We will add node rebalancing and dynamic model allocation soon.
  //     </Typography>
  //   ),
  //   confirmLabel: 'Continue',
  // });

  const onChange = useRefCallback((e) => {
    // if (clusterStatus !== 'idle') {
    //   openDialog();
    //   return;
    // }
    setModelName(String(e.target.value));
  });

  const [canAutoCommit, setCanAutoCommit] = useState(false);
  useEffect(() => {
    if (autoCommit) {
      setCanAutoCommit(autoCommit);
    }
  }, [autoCommit]);
  useEffect(() => {
    if (canAutoCommit && configModelName !== clusterModelName) {
      init();
    }
  }, [canAutoCommit, configModelName]);

  return (
    <>
      <ModelSelectRoot
        ownerState={{ variant }}
        readOnly={hostType === 'node'}
        input={variant === 'outlined' ? <OutlinedInput /> : <InputBase />}
        value={configModelName}
        onChange={onChange}
        renderValue={(value: unknown) => {
          const model = modelInfoList.find((m) => m.name === value);
          if (!model) return value as string;

          return variant === 'outlined' ?
              <ValueRow>
                <ModelLogo src={model.logoUrl} />
                <Stack gap={0.25}>
                  <ModelDisplayName>{model.displayName}</ModelDisplayName>
                  <ModelName>{model.name}</ModelName>
                </Stack>
              </ValueRow>
            : model.name;
        }}
        IconComponent={hostType === 'node' ? () => null : undefined}
      >
        {modelInfoList.map((model) => {
          const { name } = model;
          const selected = name === configModelName || name === clusterModelName;
          const loading =
            clusterStatus !== 'idle'
            && name === configModelName
            && configModelName !== clusterModelName;
          return renderOption(model, { selected, loading });
        })}
      </ModelSelectRoot>

      {/* {nodeDialog} */}
    </>
  );
};
