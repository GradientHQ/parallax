import type { EventHandler, FC, ReactNode, SyntheticEvent } from 'react';
import { useEffect, useState } from 'react';
import { forwardRef } from 'react';
import type { AlertProps } from '@mui/material';
import { Alert, AlertTitle, Button, LinearProgress } from '@mui/material';
import { styled } from '@mui/material';

const NotificationActions = styled('div')(({ theme }) => {
  const { palette } = theme;
  return {
    display: 'flex',
    flexFlow: 'row nowrap',
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: '0.75rem',
    paddingBlockStart: '0.75rem',
    color: palette.grey[700],
  };
});

const NotificationCountdown = styled('div')(({ theme }) => {
  const { palette } = theme;
  return {
    position: 'absolute',
    bottom: 0,
    left: '-1rem',
    right: '-1rem',
  };
});

const Compensation = 150;

const Countdown: FC<NotificationProps> = (props) => {
  const { severity, autoHideDuration = 0 } = props;
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (autoHideDuration <= 0) {
      return;
    }

    const step = autoHideDuration / 100;
    let frameId: number | undefined;

    let startTime: number;
    const animate = () =>
      (frameId = requestAnimationFrame((currentTime) => {
        if (!startTime) {
          startTime = currentTime;
        }

        const progress = Math.min(100, Math.ceil((currentTime + Compensation - startTime) / step));
        setProgress(progress);

        if (progress < 100) {
          animate();
        }
      }));
    animate();

    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
    };
  }, [autoHideDuration]);

  return (
    <NotificationCountdown>
      <LinearProgress variant='determinate' color={severity} value={progress} />
    </NotificationCountdown>
  );
};

export interface NotificationProps extends Omit<AlertProps, 'title'> {
  /**
   * The title of the notification.
   */
  title?: ReactNode;

  /**
   * @default 'Dismiss'
   */
  dismissLabel?: ReactNode;

  /**
   * If provided, a dismiss button will be displayed in the bottom.
   */
  onDismiss?: EventHandler<SyntheticEvent>;

  /**
   * Display a progress bar to countdown the `autoHideDuration`.
   */
  autoHideDuration?: number;
}

export const Notification = forwardRef<HTMLDivElement, NotificationProps>((props, ref) => {
  const {
    autoHideDuration,
    variant,
    title: titleProp,
    children: childrenProp,
    onClose,
    action,
    dismissLabel = 'Dismiss',
    onDismiss,
    ...rest
  } = props;

  let title = titleProp;
  let children = childrenProp;

  if (!title && children) {
    title = children;
    children = undefined;
  }

  // onClose = onClose || onDismiss;

  const nodeDismiss = onDismiss && (
    <Button variant='text' color='inherit' onClick={onDismiss}>
      {dismissLabel}
    </Button>
  );

  const nodeActions =
    ((!!action || !!nodeDismiss) && (
      <>
        {nodeDismiss}
        {action}
      </>
    ))
    || undefined;

  // if (!variant) {
  //   if (
  //     (title && children)
  //     || ((nodeDismiss || action) && onClose)
  //     || (autoHideDuration && autoHideDuration > 0)
  //   ) {
  //     variant = 'notification';
  //   }
  // }

  if (variant === 'notification') {
    return (
      <Alert {...rest} {...{ variant, onClose }} ref={ref}>
        {!!title && <AlertTitle>{title}</AlertTitle>}
        {children}
        {nodeActions && <NotificationActions>{nodeActions}</NotificationActions>}
        {autoHideDuration && autoHideDuration > 0 && <Countdown {...props} />}
      </Alert>
    );
  }

  return (
    <Alert {...rest} {...{ variant, onClose, action: nodeActions }} ref={ref}>
      {!!title && <AlertTitle>{title}</AlertTitle>}
      {children}
    </Alert>
  );
});
