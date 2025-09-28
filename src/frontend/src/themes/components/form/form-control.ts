import type { Components, Theme } from '@mui/material';

export const MuiFormControl = (theme: Theme): Components<Theme>['MuiFormControl'] => {
  const { spacing } = theme;
  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        gap: '0.25rem',
      },
    },
  };
};

export const MuiFormLabel = (theme: Theme): Components<Theme>['MuiFormLabel'] => {
  const { typography } = theme;
  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        ...typography.subtitle2,
      },
    },
  };
};
