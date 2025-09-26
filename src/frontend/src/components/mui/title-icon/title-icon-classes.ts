import { generateUtilityClass, generateUtilityClasses } from '@mui/material';

export interface TitleIconClasses {
  /** Styles applied to the root element. */
  root: string;
  /** Styles applied to the root element if `color="inherit"`. */
  colorInherit: string;
  /** Styles applied to the root element if `color="primary"`. */
  colorPrimary: string;
  /** Styles applied to the root element if `color="secondary"`. */
  colorSecondary: string;
  /** Styles applied to the root element if `color="error"`. */
  colorError: string;
  /** Styles applied to the root element if `color="info"`. */
  colorInfo: string;
  /** Styles applied to the root element if `color="success"`. */
  colorSuccess: string;
  /** Styles applied to the root element if `color="warning"`. */
  colorWarning: string;
  // /** Styles applied to the root element if `size="small"`. */
  // sizeSmall: string;
  // /** Styles applied to the root element if `size="medium"`. */
  // sizeMedium: string;
  // /** Styles applied to the root element if `size="large"`. */
  // sizeLarge: string;
}

export type TitleIconClassKey = keyof TitleIconClasses;

export function getTitleIconUtilityClass(slot: string): string {
  return generateUtilityClass('MuiTitleIcon', slot);
}

export const titleIconClasses: TitleIconClasses = generateUtilityClasses('MuiTitleIcon', [
  'root',
  'colorInherit',
  'colorPrimary',
  'colorSecondary',
  'colorError',
  'colorInfo',
  'colorSuccess',
  'colorWarning',
  // 'sizeSmall',
  // 'sizeMedium',
  // 'sizeLarge',
]);
