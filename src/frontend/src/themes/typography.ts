import type { TypographyVariantsOptions } from '@mui/material';

/**
 * DO NOT CHANGE THIS VALUE
 */
export const HTML_FONT_SIZE = 16;

// export const pxToRem = (px: number) => `${px / HTML_FONT_SIZE}rem`;

export const typography = (): TypographyVariantsOptions => ({
  fontFamily: [
    'Inter',
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    'Roboto',
    '"Helvetica Neue"',
    'Arial',
    'sans-serif',
    '"Apple Color Emoji"',
    '"Segoe UI Emoji"',
    '"Segoe UI Symbol"',
  ].join(','),

  h1: {
    // 28/16=1.75
    fontSize: '1.75rem',
    fontStyle: 'normal',
    fontWeight: 700,
    // 48/32=1.5
    lineHeight: 1.5,
  },
  h2: {
    // 20/16=1.25
    fontSize: '1.25rem',
    fontStyle: 'normal',
    fontWeight: 700,
    // 30/20=1.5
    lineHeight: 1.5,
  },
  h3: {
    // 16/16=1
    fontSize: '1rem',
    fontStyle: 'normal',
    fontWeight: 700,
    // 24/16=1.5
    lineHeight: 1.5,
  },
  h4: undefined,
  h5: undefined,
  h6: undefined,
  subtitle1: {
    // 14/16=0.875
    fontSize: '0.875rem',
    fontStyle: 'normal',
    fontWeight: 700,
    // 22/14=1.5714285714285714
    lineHeight: 1.5714285714285714,
  },
  subtitle2: {
    // 13/16=0.8125
    fontSize: '0.8125rem',
    fontStyle: 'normal',
    fontWeight: 700,
    // 20/13=1.5384615384615385
    lineHeight: 1.5384615384615385,
  },
  body1: {
    // 14/16=0.875
    fontSize: '0.875rem',
    fontStyle: 'normal',
    fontWeight: 500,
    // 22/14=1.5714285714285714
    lineHeight: 1.5714285714285714,
  },
  body2: {
    // 12/16=0.75
    fontSize: '0.75rem',
    fontStyle: 'normal',
    fontWeight: 500,
    // 18/12=1.5
    lineHeight: 1.5,
  },

  fontSize: 12,

  fontWeightLight: 400,
  fontWeightRegular: 400,
  fontWeightMedium: 500,
  fontWeightBold: 700,

  // DO NOT CHANGE THIS VALUE
  htmlFontSize: HTML_FONT_SIZE,
});
