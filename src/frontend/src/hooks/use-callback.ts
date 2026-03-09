import { useRef } from 'react';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const useConstCallback = <F extends (...args: any[]) => any>(callback: F) => {
  const ref = useRef<{ callback: F }>(undefined);
  if (!ref.current) {
    ref.current = { callback };
  }
  return ref.current.callback;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const useRefCallback = <F extends (...args: any[]) => any>(callback: F) => {
  const ref = useRef<F>(undefined);
  ref.current = callback;
  const invoke = useConstCallback<F>(((...args) => ref.current?.(...args)) as F);
  return invoke;
};
