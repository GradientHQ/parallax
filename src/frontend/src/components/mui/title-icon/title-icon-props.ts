/* eslint-disable @typescript-eslint/no-empty-object-type */
import type { Theme, SxProps } from '@mui/material';
import type { OverridableComponent, OverridableStringUnion, OverrideProps } from '@mui/types';
import type { TitleIconClasses } from './title-icon-classes';

export interface TitleIconOwnerState extends Omit<TitleIconProps, 'slots' | 'slotProps'> {}

export interface TitleIconPropsVariantOverrides {}

export interface TitleIconPropsColorOverrides {}

// export interface TitleIconSizeOverrides {}

export interface TitleIconOwnProps {
  /**
   * The icon to display.
   */
  children?: React.ReactNode;

  /**
   * Override or extend the styles applied to the component.
   */
  classes?: Partial<TitleIconClasses>;

  /**
   * The variant to use.
   * @default 'circle'
   */
  variant?: OverridableStringUnion<'circle' | 'square', TitleIconPropsVariantOverrides>;

  /**
   * The color of the component.
   * It supports both default and custom theme colors, which can be added as shown in the
   * [palette customization guide](https://mui.com/material-ui/customization/palette/#custom-colors).
   * @default 'default'
   */
  color?: OverridableStringUnion<
    'inherit' | 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning',
    TitleIconPropsColorOverrides
  >;

  // /**
  //  * The size of the component.
  //  * `small` is equivalent to the dense icon styling.
  //  * @default 'medium'
  //  */
  // size?: OverridableStringUnion<'small' | 'medium' | 'large', TitleIconSizeOverrides>;

  /**
   * The system prop that allows defining system overrides as well as additional CSS styles.
   */
  sx?: SxProps<Theme>;
}

export type TitleIconTypeMap<
  AdditionalProps = {},
  RootComponent extends React.ElementType = 'div',
> = {
  props: AdditionalProps & TitleIconOwnProps;
  defaultComponent: RootComponent;
};

export type TitleIconProps<
  RootComponent extends React.ElementType = TitleIconTypeMap['defaultComponent'],
  AdditionalProps = {},
> = OverrideProps<TitleIconTypeMap<AdditionalProps, RootComponent>, RootComponent> & {
  component?: React.ElementType;
};
