import type { ButtonProps, Components, Palette, PaletteColor, Theme } from '@mui/material';
import { buttonClasses } from '@mui/material';
import { INPUT_ICON_SIZE_REM_MAP, INPUT_RADIUS, INPUT_SIZE_REM_MAP } from '../constants';

declare module '@mui/material/Button' {
  interface ButtonPropsColorOverrides {
    brand: true;
  }

  interface ButtonPropsVariantOverrides {
    containedRounded: true;
    containedSquare: true;
    containedCircle: true;
  }
}

declare module '@mui/material/IconButton' {
  interface IconButtonPropsColorOverrides {
    brand: true;
  }

  interface IconButtonPropsSizeOverrides {
    /**
     * Set the size to 1rem (the font size of root element).
     */
    mini: true;

    /**
     * Set the size to `1em` (the font size of parent element).
     */
    em: true;

    /**
     * Set the size to `1lh` (the line height of parent element).
     */
    lh: true;
  }
}

const COLORS: readonly NonNullable<ButtonProps['color']>[] = [
  'brand',
  'primary',
  'secondary',
  'error',
  'info',
  'success',
  'warning',
];

const SIZES: readonly NonNullable<ButtonProps['size']>[] = ['small', 'medium', 'large'];

const SIZE_REM_MAP: Record<NonNullable<ButtonProps['size']>, number> = INPUT_SIZE_REM_MAP;

const MIN_WIDTH_MULTIPLY = 3 / 2.25;

const SIZE_PADDING_INLINE_REM_MAP: Record<NonNullable<ButtonProps['size']>, number> = {
  small: 0.875,
  medium: 1.125,
  large: 1.5,
};

export const MuiButton = (theme: Theme): Components<Theme>['MuiButton'] => {
  const { overlays } = theme;

  const colorShadowMap: Partial<Record<NonNullable<ButtonProps['color']>, string>> = {
    primary: overlays.buttonShadeActiveDark,
    secondary: overlays.buttonShadeActiveLight,
    brand: overlays.buttonShadeBrand,
    error: overlays.buttonShadeError,
  };

  const containedBoxShadowVariants = COLORS.map((color) =>
    (
      ['contained', 'containedRounded', 'containedSquare', 'containedCircle'] as NonNullable<
        ButtonProps['variant']
      >[]
    ).map((variant) => ({
      props: { variant, color },
      style: {
        boxShadow: overlays.buttonShadeDefault,
        '&:hover': {
          boxShadow: overlays.buttonShadeDefault,
        },
        '&:active': {
          boxShadow: colorShadowMap[color] || colorShadowMap.secondary,
        },
        '&.Mui-disabled': {
          boxShadow: overlays.buttonShadeDefault,
        },
      },
    })),
  ).flat(1);

  const containedColorVariants = COLORS.map((color) => {
    const paletteColor = theme.palette[color as keyof Palette] as PaletteColor;
    return (
      ['contained', 'containedRounded', 'containedSquare', 'containedCircle'] as NonNullable<
        ButtonProps['variant']
      >[]
    ).map((variant) => ({
      props: { variant: variant as ButtonProps['variant'], color },
      style: {
        color: paletteColor.contrastText,
        border: '1px solid',
        borderColor: paletteColor.darker,
        backgroundColor: paletteColor.main,
        '&:hover': {
          backgroundColor: paletteColor.light,
        },
        '&:active': {
          backgroundColor: paletteColor.main,
        },
        '&.Mui-disabled': {
          color: theme.palette.text.disabled,
          borderColor: paletteColor.lighter,
          backgroundColor: paletteColor.lighter,
        },
      },
    }));
  }).flat(1);

  const iconFontSizeVariants = SIZES.map((size) => ({
    props: { size },
    style: {
      '&>*:nth-of-type(1)': {
        fontSize: `${INPUT_ICON_SIZE_REM_MAP[size]}rem`,
      },
    },
  }));

  return {
    defaultProps: {
      size: 'small',
      variant: 'contained',

      disableRipple: true,
    },
    styleOverrides: {
      root: {
        gap: '0.25rem',
        borderRadius: `${INPUT_RADIUS}rem`,

        fontWeight: 400,
        letterSpacing: 0,
        textTransform: 'none',

        variants: [...containedBoxShadowVariants, ...containedColorVariants],

        [`&.${buttonClasses.text}`]: {
          fontFamily: 'inherit',
          fontWeight: 'inherit',
          fontSize: 'inherit',
          lineHeight: 'inherit',
          color: 'inherit',
          background: 'transparent',
          '&:hover': {
            background: 'transparent',
          },
        },
      },

      startIcon: {
        marginLeft: 0,
        marginRight: 0,
        '& .tabler-icon': {
          width: '1em',
          height: '1em',
        },
        variants: iconFontSizeVariants,
      },
      endIcon: {
        '& .table-icon': {
          width: '1em',
          height: '1em',
        },
        variants: iconFontSizeVariants,
      },

      sizeSmall: {
        minWidth: `${SIZE_REM_MAP.small * MIN_WIDTH_MULTIPLY}rem`,
        height: `${SIZE_REM_MAP.small}rem`,
        fontSize: theme.typography.subtitle2.fontSize,
        lineHeight: theme.typography.subtitle2.lineHeight,
        padding: 0,
        paddingBlock: 0,
        paddingInline: `${SIZE_PADDING_INLINE_REM_MAP.small}rem`,
        [`&.${buttonClasses.text}`]: {
          minWidth: 0,
          height: '1lh',
          paddingInline: 0,
        },
        '&.MuiButton-containedRounded': {
          borderRadius: `${SIZE_REM_MAP.small / 2}rem`,
        },
        '&.MuiButton-containedSquare': {
          width: `${SIZE_REM_MAP.small}rem`,
          minWidth: `${SIZE_REM_MAP.small}rem`,
          maxWidth: `${SIZE_REM_MAP.small}rem`,
          paddingInline: 0,
        },
        '&.MuiButton-containedCircle': {
          width: `${SIZE_REM_MAP.small}rem`,
          minWidth: `${SIZE_REM_MAP.small}rem`,
          maxWidth: `${SIZE_REM_MAP.small}rem`,
          paddingInline: 0,
          borderRadius: '50%',
        },
      },
      sizeMedium: {
        minWidth: `${SIZE_REM_MAP.medium * MIN_WIDTH_MULTIPLY}rem`,
        height: `${SIZE_REM_MAP.medium}rem`,
        fontSize: theme.typography.h3.fontSize,
        lineHeight: theme.typography.h3.lineHeight,
        padding: 0,
        paddingBlock: 0,
        paddingInline: `${SIZE_PADDING_INLINE_REM_MAP.medium}rem`,
        [`&.${buttonClasses.text}`]: {
          minWidth: 0,
          height: '1lh',
          paddingInline: 0,
        },
        '&.MuiButton-containedRounded': {
          borderRadius: `${SIZE_REM_MAP.medium / 2}rem`,
        },
        '&.MuiButton-containedSquare': {
          width: `${SIZE_REM_MAP.medium}rem`,
          minWidth: `${SIZE_REM_MAP.medium}rem`,
          maxWidth: `${SIZE_REM_MAP.medium}rem`,
          paddingInline: 0,
        },
        '&.MuiButton-containedCircle': {
          width: `${SIZE_REM_MAP.medium}rem`,
          minWidth: `${SIZE_REM_MAP.medium}rem`,
          maxWidth: `${SIZE_REM_MAP.medium}rem`,
          paddingInline: 0,
          borderRadius: '50%',
        },
      },
      sizeLarge: {
        minWidth: `${SIZE_REM_MAP.large * MIN_WIDTH_MULTIPLY}rem`,
        height: `${SIZE_REM_MAP.large}rem`,
        fontSize: theme.typography.h3.fontSize,
        lineHeight: theme.typography.h3.lineHeight,
        padding: 0,
        paddingBlock: 0,
        paddingInline: `${SIZE_PADDING_INLINE_REM_MAP.large}rem`,
        [`&.${buttonClasses.text}`]: {
          minWidth: 0,
          height: '1lh',
          paddingInline: 0,
        },
        '&.MuiButton-containedRounded': {
          borderRadius: `${SIZE_REM_MAP.large / 2}rem`,
        },
        '&.MuiButton-containedSquare': {
          width: `${SIZE_REM_MAP.large}rem`,
          minWidth: `${SIZE_REM_MAP.large}rem`,
          maxWidth: `${SIZE_REM_MAP.large}rem`,
          paddingInline: 0,
        },
        '&.MuiButton-containedCircle': {
          width: `${SIZE_REM_MAP.large}rem`,
          minWidth: `${SIZE_REM_MAP.large}rem`,
          maxWidth: `${SIZE_REM_MAP.large}rem`,
          paddingInline: 0,
          borderRadius: '50%',
        },
      },
    },
  };
};

export const MuiButtonBase = (theme: Theme): Components<Theme>['MuiButtonBase'] => {
  return {
    defaultProps: {
      disableRipple: true,
    },
    styleOverrides: {
      root: {
        borderRadius: '6px',
        textTransform: 'none',
      },
    },
  };
};

export const MuiButtonGroup = (theme: Theme): Components<Theme>['MuiButtonGroup'] => {
  return {
    defaultProps: {
      color: 'primary',
      variant: 'contained',
      size: 'small',
    },
  };
};

export const MuiIconButton = (theme: Theme): Components<Theme>['MuiIconButton'] => {
  const { palette } = theme;

  return {
    defaultProps: {
      size: 'mini',
      color: 'inherit',
    },
    styleOverrides: {
      root: {
        display: 'inline-flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 0,
        borderRadius: `${INPUT_RADIUS}rem`,

        textTransform: 'none',

        '&:hover': {
          backgroundColor: palette.grey[200],
          color: palette.grey[800],
        },

        '& .tabler-icon': {
          width: '1em',
          height: '1em',
        },

        variants: [
          {
            props: { size: 'mini' },
            style: {
              width: '1em',
              height: '1em',
              fontSize: '1rem',
              borderRadius: '0.25rem',
            },
          },
          {
            props: { size: 'em' },
            style: {
              width: '1em',
              height: '1em',
              fontSize: 'inherit',
              lineHeight: 'inherit',
              borderRadius: '0.25rem',
            },
          },
          {
            props: { size: 'lh' },
            style: {
              width: '1lh',
              height: '1lh',
              fontSize: '1lh',
              lineHeight: '1lh',
              borderRadius: '0.25rem',
            },
          },
          {
            props: { color: 'primary' },
            style: {
              color: palette.grey[800],
            },
          },
          {
            props: { color: 'secondary' },
            style: {
              color: palette.grey[500],
            },
          },
          {
            props: { color: 'info' },
            style: {
              color: palette.grey[700],
            },
          },
        ],
      },
      sizeSmall: {
        width: `${SIZE_REM_MAP.small}rem`,
        height: `${SIZE_REM_MAP.small}rem`,
        fontSize: `${INPUT_ICON_SIZE_REM_MAP.small}rem`,
      },
      sizeMedium: {
        width: `${SIZE_REM_MAP.medium}rem`,
        height: `${SIZE_REM_MAP.medium}rem`,
        fontSize: `${INPUT_ICON_SIZE_REM_MAP.medium}rem`,
      },
      sizeLarge: {
        width: `${SIZE_REM_MAP.large}rem`,
        height: `${SIZE_REM_MAP.large}rem`,
        fontSize: `${INPUT_ICON_SIZE_REM_MAP.large}rem`,
      },
    },
  };
};
