import type { Components, Theme } from '@mui/material';
import { dialogContentClasses, dialogTitleClasses, Stack } from '@mui/material';

export const MuiDialogActions = (theme: Theme): Components<Theme>['MuiDialogActions'] => {
  return {
    styleOverrides: {
      root: {
        padding: '1rem',
        [`.${dialogContentClasses.root}:not(.${dialogContentClasses.dividers}) + &`]: {
          marginTop: '-0.25rem',
        },
        variants: [
          {
            props: ({ ownerState }) => !ownerState.disableSpacing,
            style: {
              // gap: '0.75rem',
              '& > :not(style) ~ :not(style)': {
                marginLeft: 0,
              },
            },
          },
        ],
      },
    },
  };
};

export const MuiDialogTitle = (theme: Theme): Components<Theme>['MuiDialogTitle'] => {
  const { typography } = theme;
  return {
    styleOverrides: {
      root: {
        display: 'flex',
        flexFlow: 'row nowrap',
        justifyContent: 'space-between',
        alignContent: 'flex-start',
        alignItems: 'center',

        gap: '0.75rem',
        padding: '1rem',
        ...typography.subtitle1,
      },
    },
  };
};

export const MuiDialogContent = (theme: Theme): Components<Theme>['MuiDialogContent'] => {
  const { palette } = theme;

  return {
    styleOverrides: {
      root: {
        display: 'flex',
        flexFlow: 'column nowrap',
        justifyContent: 'flex-start',
        alignItems: 'stretch',

        gap: '0.5rem',
        padding: '1rem',

        variants: [
          {
            props: ({ ownerState }) => !!ownerState.dividers,
            style: {
              padding: '1rem',
              borderTop: `1px solid ${palette.divider}`,
              borderBottom: `1px solid ${palette.divider}`,
            },
          },
          {
            props: ({ ownerState }) => !ownerState.dividers,
            style: {
              [`.${dialogTitleClasses.root} + &`]: {
                paddingTop: 0,
              },
              [`& + &`]: {
                paddingTop: 0,
                marginTop: '-0.25rem',
              },
            },
          },
        ],
      },
    },
  };
};

export const MuiDialogContentText = (theme: Theme): Components<Theme>['MuiDialogContentText'] => {
  const { palette, spacing, typography } = theme;
  return {
    defaultProps: {
      component: Stack,
    },
    styleOverrides: {
      root: {
        gap: spacing(1),

        color: palette.text.primary,
        ...typography.body2,
      },
    },
  };
};

export const MuiDialog = (theme: Theme): Components<Theme>['MuiDialog'] => {
  const { palette } = theme;
  return {
    defaultProps: {
      slotProps: {
        paper: {
          variant: 'overlay',
          style: {
            backgroundColor: palette.background.default,
          },
        },
      },
    },
    styleOverrides: {
      root: {},
    },
  };
};
