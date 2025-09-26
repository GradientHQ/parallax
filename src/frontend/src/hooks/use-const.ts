import { useRef } from 'react';

export const useConst = <T>(value: T | (() => T)): T => {
  const ref = useRef<{ value: T }>(undefined);
  if (!ref.current) {
    ref.current = { value: typeof value === 'function' ? (value as () => T)() : value };
  }
  return ref.current.value;
};
