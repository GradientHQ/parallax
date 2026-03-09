/* eslint-disable @typescript-eslint/no-explicit-any */
import { memo } from 'react';
import { styled } from '@mui/material/styles';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import 'katex/dist/katex.min.css';
import {
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

const ChatMarkdownRoot = styled('article', {
  name: 'MuiMarkdownRenderer',
  slot: 'Root',
})<{ isThinking?: boolean }>(({ theme, isThinking }) => {
  const { palette, spacing, typography } = theme;
  const preStyles = {
    '& pre': {
      overflowX: 'auto',
    },
  };

  return {
    ...typography.body1,
    overflowWrap: 'break-word',
    maxWidth: '100%',

    display: 'flex',
    flexFlow: 'column nowrap',
    justifyContent: 'flex-start',
    alignItems: 'stretch',
    gap: spacing(1),

    color: (isThinking && palette.text.disabled) || palette.text.primary,

    ...preStyles,

    '& think': {
      color: palette.text.disabled,
    },
    '& ul, & ol': {
      margin: 0,
    },
  };
});

interface Props {
  isThinking?: boolean;
  content: string;
}

const preprocessThink = (input: string) => {
  return input.replace(/<think>/, '<think>\n\n').replace(/<\/think>/, '\n\n</think>\n\n');
};

/**
 * Convert MathJax-style \(...\) and \[...\] into KaTeX-compatible $...$ and $$...$$
 */
const preprocessMath = (input: string) => {
  return input
    .replace(/\\\[/g, '$$')
    .replace(/\\\]/g, '$$')
    .replace(/\\\(/g, '$')
    .replace(/\\\)/g, '$');
};

const components: Components = {
  h1: (props) => <Typography {...props} variant='h1' />,
  h2: (props) => <Typography {...props} variant='h2' />,
  h3: (props) => <Typography {...props} variant='h3' />,
  h4: (props) => <Typography {...props} variant='h4' />,
  h5: (props) => <Typography {...props} variant='h5' />,
  h6: (props) => <Typography {...props} variant='h6' />,
  p: (props) => <Typography {...props} variant='body1' />,
  caption: (props) => <Typography {...props} variant='caption' />,
  hr: (props) => <Divider {...props} />,
  table: (props) => (
    <TableContainer>
      <Table {...props} />
    </TableContainer>
  ),
  thead: TableHead,
  tbody: TableBody,
  tr: TableRow,
  th: TableCell as any,
  td: TableCell as any,
};

const blockList = ['script', 'iframe', 'style', 'form', 'input', 'textarea'];
const schema = {
  ...defaultSchema,
  tagNames: (defaultSchema.tagNames || []).filter((tag) => !blockList.includes(tag)),
};

const ChatMarkdown = memo<Props>(({ isThinking, content }) => {
  content = preprocessThink(content);
  content = preprocessMath(content);

  return (
    <ChatMarkdownRoot className='ChatMarkdown' isThinking={isThinking}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeRaw, [remarkGfm, remarkMath, rehypeSanitize, schema]]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </ChatMarkdownRoot>
  );
});

export default ChatMarkdown;
