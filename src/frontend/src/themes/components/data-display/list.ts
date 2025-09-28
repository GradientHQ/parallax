import type { Components, Theme } from '@mui/material';
import { INPUT_ICON_SIZE_REM_MAP, INPUT_SIZE_REM_MAP } from '../constants';

export const MuiList = (theme: Theme): Components<Theme>['MuiList'] => {
  const { palette, typography, spacing } = theme;

  return {
    defaultProps: {
      disablePadding: true,
    },
    styleOverrides: {
      root: {
        ...typography.subtitle2,
        fontWeight: typography.fontWeightMedium,
        color: palette.text.primary,

        display: 'flex',
        flexFlow: 'column nowrap',
        justifyContent: 'flex-start',
        alignItems: 'stretch',
        gap: spacing(0.5),

        variants: [
          {
            props: ({ ownerState }) => !ownerState.disablePadding,
            style: {
              paddingTop: spacing(1),
              paddingBottom: spacing(1),
            },
          },
        ],
      },
    },
  };
};

export const MuiListItem = (theme: Theme): Components<Theme>['MuiListItem'] => {
  const { spacing } = theme;
  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        padding: spacing(1),
        gap: spacing(0.75),
        variants: [
          {
            props: { disablePadding: true },
            style: {
              paddingBlock: 0,
            },
          },
        ],
      },
    },
  };
};

export const MuiListItemButton = (theme: Theme): Components<Theme>['MuiListItemButton'] => {
  const { palette, spacing } = theme;
  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        padding: spacing(1),
        gap: spacing(0.75),
        '&:hover': {
          backgroundColor: palette.grey[100],
        },
      },
    },
  };
};

export const MuiListItemIcon = (theme: Theme): Components<Theme>['MuiListItemIcon'] => {
  const { palette } = theme;
  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        fontSize: `${INPUT_ICON_SIZE_REM_MAP.small}rem`,
        minWidth: `${INPUT_ICON_SIZE_REM_MAP.small}rem`,
        color: palette.text.secondary,
        '& .tabler-icon': {
          width: '1em',
          height: '1em',
        },
      },
    },
  };
};

export const MuiListItemText = (theme: Theme): Components<Theme>['MuiListItemText'] => {
  const { palette, spacing, typography } = theme;
  return {
    defaultProps: {
      slotProps: {
        primary: {
          component: 'span',
          variant: 'subtitle2',
          fontWeight: 'medium',
          color: 'text.primary',
        },
        secondary: {
          component: 'span',
          variant: 'body2',
          fontWeight: 'medium',
          color: 'grey.600',
        },
      },
    },
    styleOverrides: {
      root: {
        margin: 0,
        variants: [
          {
            props: { inset: true },
            style: {
              padding: 0,
              paddingInlineStart: spacing(3),
            },
          },
        ],
      },
    },
  };
};

export const MuiListItemAvatar = (theme: Theme): Components<Theme>['MuiListItemAvatar'] => {
  return {
    defaultProps: {},
    styleOverrides: {},
  };
};
