import type { Components, Theme } from '@mui/material';

export const MuiInputAdornment = (theme: Theme): Components<Theme>['MuiInputAdornment'] => {
  const { palette, typography } = theme;
  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        ...typography.body2,
        color: palette.text.disabled,
        variants: [
          {
            props: {
              position: 'start',
            },
            style: {
              marginRight: 0,
            },
          },
          {
            props: {
              position: 'end',
            },
            style: {
              marginLeft: 0,
            },
          },
        ],
      },
    },
  };
};
