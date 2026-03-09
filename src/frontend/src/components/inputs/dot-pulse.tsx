import { forwardRef, type FC, type HTMLAttributes } from 'react';
import type { Transition, Variants } from 'motion';
import * as motion from 'motion/react-client';
import { Box, styled } from '@mui/material';

const SIZE_MAP: Record<DotPulseSize, number> = {
  small: 1,
  medium: 1.25,
  large: 2.25,
};

const DotPulseRoot = styled(motion.div)<{ size: DotPulseSize }>(({ theme, size }) => {
  const length = `${SIZE_MAP[size]}rem`;

  return {
    position: 'relative',
    width: length,
    height: length,

    display: 'inline-flex',
    flexFlow: 'row nowrap',
    justifyContent: 'center',
    alignItems: 'center',
  };
});

const Dot = styled(motion.div)(({ theme }) => {
  return {
    flex: 1,
    aspectRatio: 1,
    borderRadius: '50%',
    backgroundColor: 'currentColor',
  };
});

const VARIANTS: Variants = {
  pulse: {
    scale: [0, 0.6, 0, 0],
    keyTimes: [0, 0.3, 0.6, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: 'linear',
    },
  },
};

const TRANSITION: Transition = {
  staggerChildren: 0.25,
  staggerDirection: 1,
};

export type DotPulseSize = 'small' | 'medium' | 'large';

export interface DotPulseProps {
  /**
   * The size of the dot pulse.
   * @default 'medium'
   */
  size?: DotPulseSize;
}

export const DotPulse = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement> & DotPulseProps>(
  ({ size = 'medium' }, ref) => {
    return (
      <DotPulseRoot ref={ref} size={size} animate='pulse' transition={TRANSITION}>
        <Dot key={1} variants={VARIANTS} />
        <Dot key={2} variants={VARIANTS} />
        <Dot key={3} variants={VARIANTS} />
      </DotPulseRoot>
    );
  },
);
