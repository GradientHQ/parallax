import type { AlertProps, Components, Theme } from '@mui/material';
import { alertClasses } from '@mui/material';
import { IconAlertCircle, IconCircleCheck, IconInfoCircle, IconX } from '@tabler/icons-react';

declare module '@mui/material/Alert' {
  interface AlertPropsVariantOverrides {
    notification: true;
  }
}

const COLORS: NonNullable<AlertProps['severity']>[] = ['error', 'warning', 'info', 'success'];

export const MuiAlertTitle = (theme: Theme): Components<Theme>['MuiAlertTitle'] => {
  const { palette, typography } = theme;
  return {
    styleOverrides: {
      root: {
        ...typography.subtitle1,
        padding: 0,
        margin: 0,
        color: palette.text.primary,
      },
    },
  };
};

export const MuiAlert = (theme: Theme): Components<Theme>['MuiAlert'] => {
  const { palette, spacing, typography } = theme;

  return {
    defaultProps: {
      variant: 'standard',
      iconMapping: {
        info: <IconInfoCircle />,
        success: <IconCircleCheck />,
        warning: <IconAlertCircle />,
        error: <IconAlertCircle />,
      },
      slots: {
        closeIcon: () => <IconX fontSize='1.25rem' />,
      },
      slotProps: {
        root: {
          // set paper to outlined
          variant: 'outlined',
        },
      },
    },
    styleOverrides: {
      root: {
        ...typography.subtitle2,
        fontWeight: typography.fontWeightRegular,

        flex: 'none',
        position: 'relative',
        display: 'flex',
        flexFlow: 'row nowrap',
        justifyContent: 'flex-start',
        justifyItems: 'flex-start',
        alignItems: 'center',
        gap: '0.75rem',
        padding: 0,
        paddingBlock: '0.75rem',
        paddingInline: '0.75rem 1.25rem',
        overflow: 'hidden',

        variants: [
          ...COLORS.map((color) => ({
            props: { variant: 'outlined' as const, severity: color },
            style: {
              color: palette.text.secondary,
              backgroundColor: palette.background.default,
              borderColor: palette.divider,
            },
          })),
          ...COLORS.map((color) => ({
            props: { variant: 'standard' as const, severity: color },
            style: {
              alignItems: 'flex-start',
              gap: spacing(0.5),

              color: (color === 'info' && palette.text.disabled) || palette[color].main,
              backgroundColor:
                (color === 'info' && palette.background.area) || palette[color].lighter,
              border: 'none',

              [`& .${alertClasses.icon}`]: {
                color: 'inherit',
              },
            },
          })),
          {
            props: { variant: 'notification' },
            style: {
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              gap: 0,
              padding: 0,
              paddingBlock: 0,
              paddingInline: 0,
              alignItems: 'stretch',
            },
          },
        ],
      },
      icon: {
        fontSize: 'inherit',
        lineHeight: 'inherit',

        flex: 'none',
        width: '1em',
        height: '1lh',

        display: 'inline-flex',
        alignItems: 'center',
        marginRight: 0,
        padding: 0,

        '& .tabler-icon': {
          width: '1em',
          height: '1em',
        },

        variants: [
          {
            props: { variant: 'notification' },
            style: {
              order: 1,
              padding: '1rem',
            },
          },
        ],
      },
      message: {
        flex: 1,
        display: 'inline-flex',
        flexFlow: 'column nowrap',
        justifyContent: 'flex-start',
        alignItems: 'stretch',
        gap: '0.25rem',
        padding: 0,
        margin: 0,
        variants: [
          {
            props: { variant: 'notification' },
            style: {
              order: 3,
              flex: '2 0 100%',
              padding: '0 1rem 1rem',
            },
          },
        ],
      },
      action: {
        flex: 'none',
        display: 'inline-flex',
        flexFlow: 'row nowrap',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: '0.75rem',
        padding: 0,
        margin: 0,
        marginInlineStart: 'auto',
        color: palette.grey[700],
        variants: [
          {
            props: { variant: 'notification' },
            style: {
              order: 2,
              padding: '1rem',
            },
          },
        ],
      },
    },
  };
};
