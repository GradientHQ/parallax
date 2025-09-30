/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import type { DOMAttributes, FC, ForwardRefExoticComponent, ReactNode, RefAttributes } from 'react';
import { useState } from 'react';

import {
  Button,
  type ButtonProps,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  type DialogProps,
  DialogTitle,
  IconButton,
  Typography,
  Box,
} from '@mui/material';
import {
  type Icon,
  IconAlertCircle,
  IconCircleCheck,
  IconInfoCircle,
  type IconProps,
  IconX,
} from '@tabler/icons-react';
import type { SubmitErrorHandler, SubmitHandler, UseFormHandleSubmit } from 'react-hook-form';
import { TitleIcon, TitleIconForm } from '../title-icon';

const COLOR_ICON_MAP: Record<
  NonNullable<AlertDialogProps['color']>,
  ForwardRefExoticComponent<IconProps & RefAttributes<Icon>>
> = {
  primary: IconInfoCircle,
  secondary: IconInfoCircle,
  info: IconInfoCircle,
  error: IconAlertCircle,
  warning: IconAlertCircle,
  success: IconCircleCheck,
};

export type AlertDialogControl = boolean | { closeDialog?: boolean };

export type AlertDialogColor = 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning';

export interface AlertDialogProps extends Omit<DialogProps, 'onClose' | 'title' | 'content'> {
  /**
   * Whether the dialog is open.
   */
  open: boolean;

  /**
   * The callback function to be called when the dialog is closed.
   */
  onClose?: () => void;

  /**
   * The id of the title element that label the dialog.
   */
  titleId?: string;

  /**
   * The title of the dialog.
   */
  title: ReactNode;

  /**
   * The id of the content element that describe the dialog.
   */
  contentId?: string;

  /**
   * The content of the dialog.
   */
  content: ReactNode;

  /**
   * The color for title icon and confirm button.
   */
  color?: AlertDialogColor;

  /**
   * The color for title icon.
   * @default 'warning'
   */
  titleIconColor?: AlertDialogColor;

  /**
   * Whether to show the title icon.
   * If 'form', the icon will be a special form icon.
   */
  titleIcon?: boolean | 'form' | ReactNode;

  /**
   * The content to be rendered in the cancel button.
   */
  cancelLabel?: ReactNode;

  /**
   * The callback function to be called when the cancel button is clicked.
   * If the function returns a promise, the dialog will be loading status
   * and wait for the promise to be resolved before closing the dialog.
   * Return `false` or `{ closeDialog: false }` to prevent dialog from closing.
   */
  onCancel?: () => void | Promise<void> | AlertDialogControl | Promise<AlertDialogControl>;

  /**
   * The content to be rendered in the confirm button.
   */
  confirmLabel?: ReactNode;

  /**
   * The callback function to be called when the confirm button is clicked.
   * If the function returns a promise, the dialog will be loading status
   * and wait for the promise to be resolved before closing the dialog.
   * Return `false` or `{ closeDialog: false }` to prevent dialog from closing.
   */
  onConfirm?: () => void | Promise<void> | AlertDialogControl | Promise<AlertDialogControl>;

  /**
   * The content to be rendered in the submit button.
   * If `submitLabel` is provided, the dialog will be a form dialog.
   */
  submitLabel?: ReactNode;

  /**
   * The callback function to be called when the form is submitted and the form is valid.
   * Return `false` or `{ closeDialog: false }` to prevent dialog from closing.
   */
  onSubmitValid?: SubmitHandler<any>;

  /**
   * The callback function to be called when the form is submitted and the form is invalid.
   * The dialog will not close when the form invalid.
   */
  onSubmitInvalid?: SubmitErrorHandler<any>;

  /**
   * Come from `react-hook-form`.
   * If `submitLabel` is provided, the dialog will be a form dialog.
   * The `handleSubmit` function will be used to handle the form submission,
   * it accepts two functions: `onSubmitValid` and `onSubmitInvalid`,
   * and return a `submit` event handler function to be used in the element `<form>`.
   * The `onSubmitValid` function will be called when the form is submitted and the form is valid.
   * The `onSubmitInvalid` function will be called when the form is submitted and the form is invalid.
   *
   * @example
   * ```tsx
   * import { useForm } from 'react-hook-form';
   *
   * const { handleSubmit } = useForm();
   *
   * const onSubmit = (data: any) => {
   *   console.log(data);
   * };
   *
   * <AlertDialog
   *  handleSubmit={handleSubmit}
   *  onSubmitValid={onSubmit}
   *  onSubmitInvalid={onSubmitInvalid}
   * />
   * ```
   */
  handleSubmit?: UseFormHandleSubmit<any>;

  secondaryAction?: ReactNode;
}

export const AlertDialog: FC<AlertDialogProps> = (props) => {
  const {
    open,
    onClose = () => {},

    titleId,
    title,
    contentId,
    content,

    color = 'primary',
    titleIconColor = 'warning',
    titleIcon = true,

    cancelLabel,
    onCancel = () => {},

    confirmLabel,
    onConfirm = () => {},

    submitLabel,
    onSubmitValid,
    onSubmitInvalid,
    handleSubmit,
    secondaryAction,

    ...rest
  } = props;

  if (import.meta.env.DEV) {
    if (submitLabel && (!handleSubmit || !onSubmitValid)) {
      throw new Error('handleSubmit and onSubmitValid are required when submitLabel is provided');
    }
  }

  const SlotIcon = COLOR_ICON_MAP[color];

  const [loading, setLoading] = useState(false);

  // The function to wrap the action callbacks
  const handleActionResult = (result: any): boolean => {
    // If return `false` or `{ closeDialog: false }`, do not close the dialog
    if (result === false || (typeof result === 'object' && result.closeDialog === false)) {
      return false;
    }
    // If return `true`, close the dialog
    return true;
  };

  const pipeHandlers =
    <F extends (...args: any[]) => any>(handler: F) =>
    async (...args: Parameters<F>): Promise<void> => {
      if (loading) {
        return;
      }

      try {
        const result = handler(...args);

        if (result instanceof Promise) {
          setLoading(true); // Dialog switch to loading status.
          const resolvedResult = await result;
          setLoading(false);

          if (handleActionResult(resolvedResult)) {
            onClose();
          }
        } else {
          if (handleActionResult(result)) {
            onClose();
          }
        }
      } catch (error) {
        setLoading(false);
        // Do not close the dialog when error
        console.error('Error in dialog action:', error);
      }
    };

  let formProps: DOMAttributes<HTMLFormElement> | undefined;
  const actionButtonPropsList: ButtonProps[] = [];

  if (cancelLabel) {
    actionButtonPropsList.push({
      key: 'cancel',
      children: cancelLabel,
      color: 'secondary',
      onClick: pipeHandlers(onCancel),
      loading,
    });
  }

  if (confirmLabel) {
    actionButtonPropsList.push({
      key: 'confirm',
      children: confirmLabel,
      color,
      onClick: pipeHandlers(onConfirm),
      loading,
    });
  }

  if (submitLabel) {
    actionButtonPropsList.push({
      key: 'submit',
      children: submitLabel,
      type: 'submit',
      loading,
    });

    if (handleSubmit && onSubmitValid) {
      formProps = {
        onSubmit: (e) => {
          // Prevent the page from being refreshed when the form is submitted.
          e.preventDefault();
          e.stopPropagation();

          if (loading) {
            return;
          }

          setLoading(true);
          handleSubmit(
            async (...args) => {
              try {
                const result = await onSubmitValid(...args);
                setLoading(false);

                // 处理提交结果，决定是否关闭
                if (handleActionResult(result)) {
                  onClose();
                }
              } catch (error) {
                setLoading(false);
                // The dialog will not close when error
                console.error('Error in form submission:', error);
              }
            },
            async (...args) => {
              try {
                // The dialog will not close when the form is invalid.
                await onSubmitInvalid?.(...args);
              } catch (error) {
                console.error('Error in form validation:', error);
              } finally {
                setLoading(false);
              }
            },
          )();
        },
      };
    }
  }

  const safeOnClose = () => {
    if (loading) {
      return;
    }
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={safeOnClose}
      aria-labelledby={titleId}
      aria-describedby={contentId}
      {...rest}
      component={(formProps && 'form') || undefined}
      {...(formProps as any)}
      slotProps={{
      paper: {
        sx: {
          p: 1.5,
        },
      },
    }}
    >
      <DialogTitle id={(!SlotIcon && titleId) || undefined}>
        {(titleIcon === 'form' && <TitleIconForm />)
          || (titleIcon === true && (
            <TitleIcon variant='circle' color={titleIconColor}>
              <SlotIcon />
            </TitleIcon>
          )) || (
            <TitleIcon variant='circle' color={titleIconColor}>
              {titleIcon}
            </TitleIcon>
          )
          || undefined}

        {!titleIcon && title}

        <IconButton size='small' onClick={safeOnClose}>
          <IconX size='1.25rem' />
        </IconButton>
      </DialogTitle>

      <DialogContent>
        {titleIcon && (
          <Typography variant='subtitle1' id={titleId} sx={{ fontSize: '1.125rem', fontWeight: 600, mr: 'auto' }}>
            {title}
          </Typography>
        )}

        <DialogContentText id={contentId}>{content}</DialogContentText>
      </DialogContent>

      {actionButtonPropsList.length > 0 && (
        <DialogActions>
          {secondaryAction}
          {actionButtonPropsList.map((props) => (
            <Button
              {...props}
              key={props.key}
              {...(actionButtonPropsList.length === 1 && { fullWidth: true })}
            />
          ))}
        </DialogActions>
      )}
    </Dialog>
  );
};
