import { toggleButtonClasses, type Components, type Theme } from '@mui/material';
import { INPUT_SIZE_REM_MAP } from '../constants';

export const MuiToggleButton = (theme: Theme): Components<Theme>['MuiToggleButton'] => {
  const { palette, typography } = theme;

  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        ...typography.subtitle2,

        flex: 1,
        height: `${INPUT_SIZE_REM_MAP.small}rem`,

        borderColor: palette.divider,

        color: palette.grey[400],
        backgroundColor: palette.grey[200],
        '&:hover': {
          color: palette.text.secondary,
          backgroundColor: palette.background.area,
          '@media (hover: none)': {
            backgroundColor: palette.grey[200],
          },
        },
        variants: [
          {
            props: { color: 'standard' },
            style: {
              [`&.${toggleButtonClasses.selected}`]: {
                color: palette.text.primary,
                backgroundColor: palette.background.default,
                '&:hover': {
                  backgroundColor: palette.background.default,
                  // Reset on touch devices, it doesn't add specificity
                  '@media (hover: none)': {
                    backgroundColor: palette.background.default,
                  },
                },
              },
            },
          },
        ],
      },
    },
  };
};

export const MuiToggleButtonGroup = (theme: Theme): Components<Theme>['MuiToggleButtonGroup'] => {
  return {
    defaultProps: {},
    styleOverrides: {
      root: {},
    },
  };
};
