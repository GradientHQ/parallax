import type { Components, OutlinedInputProps, Theme } from '@mui/material';

export const MuiTextField = (theme: Theme): Components<Theme>['MuiTextField'] => {
  return {
    defaultProps: {
      size: 'small',
      variant: 'outlined',
      slotProps: {
        input: {} as Partial<OutlinedInputProps>,
        inputLabel: { shrink: true },
      },
    },
    styleOverrides: {
      root: {},
    },
  };
};
