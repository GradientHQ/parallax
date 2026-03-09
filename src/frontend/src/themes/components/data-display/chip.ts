import type { ChipProps, Components, Theme, Palette, PaletteColor } from '@mui/material';

const SIZE_PADDING_INLINE_REM_MAP: Record<NonNullable<ChipProps['size']>, number> = {
  small: 0.625,
  medium: 0.75,
};

const SIZE_PADDING_RADIUS_REM_MAP: Record<NonNullable<ChipProps['size']>, number> = {
  small: 0.375,
  medium: 0.25,
};

const COLORS: readonly NonNullable<ChipProps['color']>[] = [
  'secondary',
  'info',
  'brand',
  'primary',
  'error',
  'success',
  'warning',
];

export const MuiChip = (theme: Theme): Components<Theme>['MuiChip'] => {
  const containedColorVariants = COLORS.map((color) => {
    const paletteColor = theme.palette[color as keyof Palette] as PaletteColor;
    return (['filled', 'outlined'] as NonNullable<ChipProps['variant']>[]).map((variant) => ({
      props: { variant: variant as ChipProps['variant'], color },
      style: {
        backgroundColor: paletteColor.chip,
        color: paletteColor.chipText,
        // '&:hover': {
        //   backgroundColor: paletteColor.light,
        // },
        // '&:active': {
        //   backgroundColor: paletteColor.main,
        // },
        // '&.Mui-disabled': {
        //   color: theme.palette.text.disabled,
        //   borderColor: paletteColor.lighter,
        //   backgroundColor: paletteColor.lighter,
        // },
      },
    }));
  }).flat(1);

  return {
    defaultProps: {
      variant: 'filled',
    },
    styleOverrides: {
      root: {
        fontWeight: 700,
        letterSpacing: 0,
        textTransform: 'capitalize',
        variants: [...containedColorVariants],
      },
      deleteIcon: {
        marginLeft: '0.375rem',
        marginRight: 0,
        color: theme.palette.grey[700],
        '&:hover': {
          color: theme.palette.grey[700],
        },
      },
      sizeSmall: {
        minWidth: `auto`,
        height: '1.625rem',
        paddingInline: `${SIZE_PADDING_INLINE_REM_MAP.small}rem`,
        borderRadius: `${SIZE_PADDING_RADIUS_REM_MAP.small}rem`,
        '.MuiChip-label': {
          padding: '0 0',
        },
        fontSize: theme.typography.body2.fontSize,
      },
      sizeMedium: {
        minWidth: `auto`,
        height: '1.375rem',
        paddingInline: `${SIZE_PADDING_INLINE_REM_MAP.medium}rem`,
        borderRadius: `${SIZE_PADDING_RADIUS_REM_MAP.medium}rem`,
        '.MuiChip-label': {
          padding: '0 0',
        },
      },
    },
  };
};
