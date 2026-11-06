import assert from "node:assert/strict";
import { test } from "node:test";
import { createSSE } from "../../src/frontend/src/services/chat-sse.ts";

const encoder = new TextEncoder();
const frame = (delta) =>
  `data: ${JSON.stringify({ object: "chat.completion.chunk", choices: [{ delta }] })}\n\n`;

const response = (chunks) =>
  new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );

const session = (t, responses) => {
  t.mock.method(globalThis, "fetch", async (url) => {
    assert.equal(url, "https://stream.invalid/v1/chat/completions");
    assert.ok(responses.length > 0);
    return responses.shift();
  });
  let resolve;
  let reject;
  let messages;
  const sse = createSSE({
    url: "https://stream.invalid/v1/chat/completions",
    onMessage: (message) => messages.push(message.data),
    onClose: () => resolve(messages),
    onError: (error) => reject(error),
  });
  const connect = () =>
    new Promise((yes, no) => {
      resolve = yes;
      reject = no;
      messages = [];
      sse.connect("test-model", []);
    });
  connect.disconnect = () => sse.disconnect();
  return connect;
};

test("preserves UTF-8 content across every possible read boundary", async (t) => {
  const text = "你好 🙂 café";
  const bytes = encoder.encode(frame({ content: text }));
  const responses = Array.from({ length: bytes.length - 1 }, (_, i) =>
    response([bytes.slice(0, i + 1), bytes.slice(i + 1)]),
  );
  const connect = session(t, responses);
  for (let split = 1; split < bytes.length; split++) {
    const messages = await connect();
    assert.equal(messages.length, 1, `split at byte ${split}`);
    assert.equal(
      messages[0].choices[0].delta.content,
      text,
      `split at byte ${split}`,
    );
  }
});

test("preserves reasoning and content delivered one byte at a time", async (t) => {
  const bytes = encoder.encode(
    frame({ reasoning: "想 🤔" }) + frame({ content: "答案 é" }),
  );
  const connect = session(t, [
    response(Array.from(bytes, (byte) => Uint8Array.of(byte))),
  ]);
  const messages = await connect();
  assert.deepEqual(
    messages.map((m) => m.choices[0].delta),
    [{ reasoning: "想 🤔" }, { content: "答案 é" }],
  );
});

test("keeps ASCII events and newline buffering unchanged", async (t) => {
  const bytes = encoder.encode(
    frame({ content: "hello" }) + frame({ content: " world" }),
  );
  const connect = session(t, [
    response([bytes.slice(0, 7), bytes.slice(7, 61), bytes.slice(61)]),
  ]);
  const messages = await connect();
  assert.deepEqual(
    messages.map((m) => m.choices[0].delta.content),
    ["hello", " world"],
  );
});

test("a truncated final character cannot contaminate a later connection", async (t) => {
  const prefix = encoder.encode('data: {"unfinished":"');
  const incomplete = new Uint8Array([...prefix, encoder.encode("你")[0]]);
  const connect = session(t, [
    response([incomplete]),
    response([encoder.encode(frame({ content: "ok" }))]),
  ]);
  assert.deepEqual(await connect(), []);
  const messages = await connect();
  assert.equal(messages.length, 1);
  assert.equal(messages[0].choices[0].delta.content, "ok");
});

test("cancelling a partial UTF-8 character keeps the next connection clean", async (t) => {
  let read;
  const firstRead = new Promise((resolve) => {
    read = resolve;
  });
  const originalRead = ReadableStreamDefaultReader.prototype.read;
  let reads = 0;
  t.mock.method(
    ReadableStreamDefaultReader.prototype,
    "read",
    function (...args) {
      const result = originalRead.apply(this, args);
      if (++reads === 2) read();
      return result;
    },
  );
  const partial = new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(
          new Uint8Array([...encoder.encode('data: {"unfinished":"'), 0xe4]),
        );
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
  const connect = session(t, [
    partial,
    response([encoder.encode(frame({ content: "next" }))]),
  ]);
  const pending = connect();
  await firstRead;
  connect.disconnect();
  assert.deepEqual(await pending, []);
  const messages = await connect();
  assert.equal(messages.length, 1);
  assert.equal(messages[0].choices[0].delta.content, "next");
});
