import type { Ref } from 'react';
import { forwardRef } from 'react';
import { clsx } from 'clsx';
import composeClasses from '@mui/utils/composeClasses';
import { capitalize } from '@mui/material/utils';
import type { CSSInterpolation } from '@mui/material';
import { iconClasses, styled, useThemeProps } from '@mui/material';

import { getTitleIconUtilityClass, type TitleIconClassKey } from './title-icon-classes';
import type {
  TitleIconOwnerState,
  TitleIconOwnProps,
  TitleIconProps,
  TitleIconTypeMap,
} from './title-icon-props';

const useUtilityClasses = (ownerState: TitleIconOwnerState) => {
  const { classes, variant, color } = ownerState;
  const slots = {
    root: [
      'root',
      variant && `variant${capitalize(variant)}`,
      color && `color${capitalize(color)}`,
    ],
    outlineOuter: ['outlineOuter'],
    outlineInner: ['outlineInner'],
  };
  return composeClasses(slots, getTitleIconUtilityClass, classes);
};

const COLORS: NonNullable<NonNullable<TitleIconProps['color']>>[] = [
  'inherit',
  'primary',
  'secondary',
  'error',
  'info',
  'success',
  'warning',
];

const TitleIconRoot = styled('div', {
  name: 'MuiTitleIcon',
  slot: 'Root',
  overridesResolver: (
    props: { ownerState: TitleIconOwnerState },
    styles: Record<TitleIconClassKey, CSSInterpolation>,
  ) => {
    const {
      ownerState: { color, variant },
    } = props;
    return [
      styles.root,
      color && styles[`color${capitalize(color)}` as TitleIconClassKey],
      variant && styles[`variant${capitalize(variant)}` as TitleIconClassKey],
    ];
  },
})<{ ownerState: TitleIconOwnerState }>(({ theme }) => {
  const { palette } = theme;
  return {
    position: 'relative',
    width: '2.25rem',
    height: '2.25rem',

    display: 'inline-flex',
    justifyContent: 'center',
    alignItems: 'center',

    color: 'inherit',
    backgroundColor: 'transparent',
    border: 'none',
    outline: 'none',

    fontSize: '1.25rem',
    [`& .${iconClasses.root}, & .tabler-icon, & svg`]: {
      width: '1em',
      height: '1em',
    },

    variants: [
      ...COLORS.map((color) => ({
        props: { color },
        style: {
          color:
            color === 'inherit' ? 'inherit'
            : color === 'default' ? palette.text.primary
            : palette[color].main,
        },
      })),
    ],
  };
});

const TitleIconOutlineOuter = styled('div', {
  name: 'MuiTitleIcon',
  slot: 'OutlineOuter',
})<{ ownerState: TitleIconOwnerState }>(() => {
  return {
    boxSizing: 'border-box',
    width: '2.25rem',
    height: '2.25rem',

    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',

    opacity: 0.1,
    borderWidth: '1.667px',
    borderStyle: 'solid',
    borderColor: 'currentcolor',

    variants: [
      {
        props: { variant: 'circle' },
        style: {
          borderRadius: '50%',
        },
      },
      {
        props: { variant: 'square' },
        style: {
          borderRadius: '0.38rem',
        },
      },
    ],
  };
});

const TitleIconOutlineInner = styled('div', {
  name: 'MuiTitleIcon',
  slot: 'OutlineInner',
})<{ ownerState: TitleIconOwnerState }>(() => {
  return {
    boxSizing: 'border-box',
    width: '1.75rem',
    height: '1.75rem',

    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',

    opacity: 0.3,
    borderWidth: '1.667px',
    borderStyle: 'solid',
    borderColor: 'currentcolor',

    variants: [
      {
        props: { variant: 'circle' },
        style: {
          borderRadius: '50%',
        },
      },
      {
        props: { variant: 'square' },
        style: {
          borderRadius: '0.25em',
        },
      },
    ],
  };
});

export const TitleIcon = forwardRef<HTMLDivElement, TitleIconProps>(
  function TitleIcon(inProps, ref) {
    const props = useThemeProps({ props: inProps, name: 'MuiTitleIcon' });
    const {
      children,
      classes: classesProp,
      variant = 'circle',
      color = 'default',
      className,
      ...rest
    } = props;
    const ownerState: TitleIconOwnerState = {
      ...props,
      variant,
      color,
    };

    const classes = useUtilityClasses(ownerState);

    return (
      <TitleIconRoot
        ref={ref}
        className={clsx(classes.root, className)}
        ownerState={ownerState}
        role='img'
        {...rest}
      >
        <TitleIconOutlineOuter className={classes.outlineOuter} ownerState={ownerState} />
        <TitleIconOutlineInner className={classes.outlineInner} ownerState={ownerState} />
        {children}
      </TitleIconRoot>
    );
  },
);
