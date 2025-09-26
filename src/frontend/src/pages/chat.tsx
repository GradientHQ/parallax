import { Box } from '@mui/material';
import { DrawerLayout } from '../components/common';
import { ChatInput, ChatMessages } from '../components/inputs';

export default function PageChat() {
  return (
    <DrawerLayout>
      <ChatMessages />
      <ChatInput />
    </DrawerLayout>
  );
}
