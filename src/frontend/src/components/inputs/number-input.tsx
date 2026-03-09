import type { ChangeEventHandler, FC, Ref } from 'react';
import { forwardRef, useRef } from 'react';
import {
  IconButton,
  InputAdornment,
  InputBase,
  FilledInput,
  OutlinedInput,
  useForkRef,
  type OutlinedInputProps,
} from '@mui/material';
import { IconMinus, IconPlus } from '@tabler/icons-react';
import { useRefCallback } from '../../hooks';

export interface NumberInputProps extends OutlinedInputProps {
  value: number;
}

export const NumberInput: FC<NumberInputProps> = ({
  inputRef: inputRefProp,
  slotProps,
  onChange: onChangeProp,
  ...rest
}) => {
  const inputRefInner = useRef<HTMLInputElement>(null);
  const inputRef = useForkRef(inputRefInner, inputRefProp);

  const onChange = useRefCallback<ChangeEventHandler<HTMLTextAreaElement | HTMLInputElement>>(
    (event) => {
      event.target.value = `${Math.max(1, parseInt(event.target.value) || 1)}`;
      onChangeProp?.(event);
    },
  );

  const triggerChange = useRefCallback((newValue: number) => {
    if (onChangeProp) {
      // Create a synthetic event that mimics a real input change
      const syntheticEvent = {
        target: { value: newValue.toString() },
        currentTarget: { value: newValue.toString() },
        type: 'change',
        bubbles: true,
        cancelable: true,
        preventDefault: () => {},
        stopPropagation: () => {},
      } as React.ChangeEvent<HTMLInputElement>;

      onChangeProp(syntheticEvent);
    }
  });

  const onMinus = useRefCallback(() => {
    const { current: input } = inputRefInner;
    if (!input) {
      return;
    }
    const prev = Number(input.value);
    const next = Math.max(1, prev - 1);
    input.value = next.toString();
    triggerChange(next);
  });

  const onPlus = useRefCallback(() => {
    const { current: input } = inputRefInner;
    if (!input) {
      return;
    }
    const prev = Number(input.value);
    const next = Math.max(1, prev + 1);
    input.value = next.toString();
    triggerChange(next);
  });

  return (
    <OutlinedInput
      {...rest}
      type='number'
      sx={{
        '& input': {
          textAlign: 'center',
          width: '2.5rem',
        },
      }}
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
      inputRef={inputRef}
      onChange={onChange}
      startAdornment={
        <InputAdornment position='start'>
          <IconButton onClick={onMinus}>
            <IconMinus />
          </IconButton>
        </InputAdornment>
      }
      endAdornment={
        <InputAdornment position='end'>
          <IconButton onClick={onPlus}>
            <IconPlus />
          </IconButton>
        </InputAdornment>
      }
      slotProps={{
        input: {
          step: 1,
          sx: { textAlign: 'center', ...slotProps?.input?.sx },
          min: 1,
          ...slotProps?.input,
        },
        ...slotProps,
      }}
    />
  );
};
