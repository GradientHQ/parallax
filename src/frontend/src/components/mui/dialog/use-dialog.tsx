'use client';

import type { JSX } from 'react';
import { useMemo, useState } from 'react';
import { AlertDialog, type AlertDialogProps } from './alert-dialog';

export interface AlertDialogActions {
  open: () => void;
  close: () => void;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface AlertDialogOptions {}

/**
 * A hook to manage the `open` state of `AlertDialog` and forward the render node.
 * @see AlertDialog
 * @example
 * ```tsx
 * const [node, actions] = useAlertDialog({
 *   title: 'Title',
 *   content: 'Content',
 *   cancelLabel: 'OK'
 * });
 *
 * <div>
 *   <Button onClick={actions.open}>Open</Button>
 *   <Button onClick={actions.close}>Close</Button>
 *   {node}
 * </div>
 * ```
 */
export const useAlertDialog = (
  props: Omit<AlertDialogProps, 'open'>,
  options?: AlertDialogOptions,
): [JSX.Element, AlertDialogActions] => {
  const [open, setOpen] = useState(false);

  const node = (
    <AlertDialog
      {...props}
      open={open}
      onClose={() => {
        setOpen(false);
        props.onClose?.();
      }}
    />
  );

  const actions: AlertDialogActions = useMemo(
    () => ({
      open: () => setOpen(true),
      close: () => setOpen(false),
    }),
    [],
  );

  return [node, actions];
};
