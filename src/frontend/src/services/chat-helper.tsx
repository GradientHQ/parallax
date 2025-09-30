/**
 * Mapping GPT generation massages.
 * The key is the channel name, the value is the message.
 */
export interface GptGenerationMap {
  analysis: string;
  final: string;
  [key: string]: string;
}

export function parseGenerationGpt(buffer: string): GptGenerationMap {
  buffer = buffer.trim();

  const map: GptGenerationMap = { analysis: '', final: '' };

  const regex = /<\|channel\|>([^<]+)<\|message\|>(.*?)(<\|end\|>|$)/gs;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(buffer)) !== null) {
    map[match[1]] = match[2]?.trim() || '';
  }

  return map;
}

export interface QwenGenerationMap {
  think: string;
  content: string;
}

const THINK_TAG_OPEN = '<think>';
const THINK_TAG_CLOSE = '</think>';

export function parseGenerationQwen(buffer: string): QwenGenerationMap {
  buffer = buffer.trim();

  const map: QwenGenerationMap = { think: '', content: '' };

  while (buffer.includes(THINK_TAG_OPEN)) {
    const thinkStart = buffer.indexOf(THINK_TAG_OPEN);
    const thinkEnd = buffer.indexOf(THINK_TAG_CLOSE);
    const think = buffer.substring(
      thinkStart + THINK_TAG_OPEN.length,
      thinkEnd > thinkStart ? thinkEnd : buffer.length,
    );
    buffer = buffer.replace(
      THINK_TAG_OPEN + think + (thinkEnd > thinkStart ? THINK_TAG_CLOSE : ''),
      '',
    );
    map.think += '\n\n' + think;
  }
  map.think = map.think.trim();
  map.content = buffer.trim();

  return map;
}
