import type { FC, Ref } from 'react';
import { forwardRef, useRef } from 'react';
import { IconButton, InputAdornment, OutlinedInput, type OutlinedInputProps } from '@mui/material';
import { IconMinus, IconPlus } from '@tabler/icons-react';

export interface NumberInputProps extends OutlinedInputProps {
  value: number;
}

export const NumberInput: FC<NumberInputProps> = (props) => {
  return (
    <OutlinedInput
      type='number'
      {...props}
      // startAdornment={
      //   <InputAdornment position='start'>
      //     <IconButton>
      //       <IconMinus />
      //     </IconButton>
      //   </InputAdornment>
      // }
      // endAdornment={
      //   <InputAdornment position='end'>
      //     <IconButton>
      //       <IconPlus />
      //     </IconButton>
      //   </InputAdornment>
      // }
      inputRef={props.inputRef}
      slotProps={{
        input: {
          step: 1,
          sx: { textAlign: 'center', ...props.slotProps?.input?.sx },
          ...props.slotProps?.input,
        },
        ...props.slotProps,
      }}
      data-debug-typeof={typeof props.value}
    />
  );
};
