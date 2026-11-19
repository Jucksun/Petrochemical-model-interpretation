// Observatory AI Worker: API only. The terminal page lives on your own site.
// Deploy: npx wrangler deploy

const MODELS = {
  llama: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  mistral: "@cf/mistralai/mistral-small-3.1-24b-instruct",
  gemma: "@cf/google/gemma-3-12b-it",
};

// Only these sites may call the API from a browser.
const ALLOWED_ORIGINS = [
  "https://jacksonsobservatory.com",
  "https://www.jacksonsobservatory.com",
  "http://localhost:8000", // for testing on your own computer
];

const SYSTEM = `You are the terminal assistant on Jackson's Observatory (JacksonOS), a personal site about open-source intelligence and industrial monitoring.
Be concise: plain text, short paragraphs, no markdown.
If you are not sure of a fact, say so instead of guessing. Never invent names, numbers, filings or sources.
Decline requests that would help someone cause harm, and say briefly why.`;

const MAX_MSG_CHARS = 1000;
const MAX_TURNS = 8;

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const cors = {
      "Access-Control-Allow-Origin": ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "content-type",
      Vary: "Origin",
    };

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });

    if (request.method !== "POST") {
      return new Response("JacksonOS AI endpoint. POST only.", { status: 200, headers: cors });
    }
    if (!ALLOWED_ORIGINS.includes(origin)) {
      return new Response("Origin not allowed", { status: 403, headers: cors });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response("Bad JSON", { status: 400, headers: cors });
    }
    const model = MODELS[body.model] || MODELS.llama;
    const history = Array.isArray(body.messages) ? body.messages.slice(-MAX_TURNS * 2) : [];
    const messages = history
      .filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
      .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_MSG_CHARS) }));
    if (!messages.length || messages[messages.length - 1].role !== "user") {
      return new Response("Need a user message", { status: 400, headers: cors });
    }

    try {
      const stream = await env.AI.run(model, {
        messages: [{ role: "system", content: SYSTEM }, ...messages],
        stream: true,
        max_tokens: 500,
        temperature: 0.3,
      });
      return new Response(stream, {
        headers: { ...cors, "content-type": "text/event-stream", "cache-control": "no-store" },
      });
    } catch (err) {
      return new Response("Model error: " + (err.message || "unknown"), { status: 502, headers: cors });
    }
  },
};
