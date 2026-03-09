import type { FC, PropsWithChildren, ReactNode } from 'react';
import { forwardRef } from 'react';
import type { InternalSnack, SnackbarProviderProps } from 'notistack';
import { SnackbarProvider, useSnackbar } from 'notistack';
import { Notification } from '../../../components/mui';

declare module 'notistack' {
  export interface OptionsObject<V extends VariantType = VariantType> extends SharedProps<V> {
    title?: ReactNode;
    message?: ReactNode;

    /**
     * Set the variant of alert to notification.
     * It has header with icon and close button, title and content,
     * bottom action buttons and progress bar.
     * @default true
     */
    notification?: boolean;

    /**
     * Show the close button in the header or not.
     * @default depends on the prop notification
     */
    closable?: boolean;

    /**
     * Show the dismiss action button or not.
     */
    dismissAble?: boolean;
  }
}

const Simple = forwardRef<HTMLDivElement, InternalSnack>((props, ref) => {
  const {
    id,
    variant,
    title,
    message,
    action: propsAction,
    notification = true,
    closable = notification,
    dismissAble,
    persist,
    autoHideDuration: propsAutoHideDuration,
    hideIconVariant,
    className,
    style,
  } = props;

  const { closeSnackbar } = useSnackbar();

  const severity = variant === 'default' ? 'info' : variant;
  const action = typeof propsAction === 'function' ? propsAction(id) : propsAction;
  const onClose = (closable && (() => closeSnackbar(id))) || undefined;
  const onDismiss = (dismissAble && (() => closeSnackbar(id))) || undefined;
  const autoHideDuration = propsAutoHideDuration || undefined;

  return (
    <Notification
      variant={(notification && 'notification') || 'outlined'}
      severity={severity}
      icon={hideIconVariant ? false : undefined}
      onClose={onClose}
      onDismiss={onDismiss}
      action={action}
      title={title}
      autoHideDuration={autoHideDuration}
      ref={ref}
      className={className}
      style={style}
    >
      {message}
    </Notification>
  );
});

const Components: SnackbarProviderProps['Components'] = {
  default: Simple,
  info: Simple,
  success: Simple,
  warning: Simple,
  error: Simple,
};

const Provider: FC<PropsWithChildren> = ({ children }) => {
  return (
    <SnackbarProvider
      anchorOrigin={{
        vertical: 'bottom',
        horizontal: 'right',
      }}
      maxSnack={5}
      autoHideDuration={5000}
      Components={Components}
      style={{
        width: '21.25rem',
      }}
    >
      {children}
    </SnackbarProvider>
  );
};

export { Provider as SnackbarProvider };
