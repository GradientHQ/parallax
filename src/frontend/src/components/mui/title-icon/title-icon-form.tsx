import type { FC } from 'react';
import { styled } from '@mui/material';

const TitleIconFormRoot = styled('div', {
  name: 'MuiTitleIconForm',
  slot: 'Root',
})(({ theme }) => {
  const { palette } = theme;

  return {
    flex: 'none',
    width: '3rem',
    height: '3rem',
    aspectRatio: '1 / 1',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '0.375rem',
    backgroundColor: palette.grey[250],
    overflow: 'hidden',
  };
});

const TitleIconFormImage = styled('div', {
  name: 'MuiTitleIconForm',
  slot: 'Image',
})(({ theme }) => {
  const { palette } = theme;

  return {
    width: '100%',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: `url(/assets/common/form-half-one.png) center / cover no-repeat`,
    mixBlendMode: 'multiply',
  };
});

export const TitleIconForm: FC = () => {
  return (
    <TitleIconFormRoot>
      <TitleIconFormImage />
    </TitleIconFormRoot>
  );
};
