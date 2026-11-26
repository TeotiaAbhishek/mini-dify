// Every call the console makes to the backend goes through here.
// It adds the console token (if one is saved), turns errors into ApiError,
// and reads streamed answers (Server-Sent Events).

const TOKEN_KEY = "mini-dify-console-token";

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return ""; // storage can be blocked (private windows): carry on without a token
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // nothing to do: the token just won't be remembered
  }
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(`${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

function headers(extra = {}) {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}`, ...extra } : extra;
}

// FastAPI's 422 detail is a list of {loc, msg}. Turn it into one line per problem.
function describe(detail) {
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc || []).slice(1).join(".") || "body"}: ${d.msg}`).join("\n");
  }
  return typeof detail === "string" ? detail : JSON.stringify(detail);
}

async function fail(response) {
  let detail = response.statusText;
  try {
    detail = describe((await response.json()).detail ?? detail);
  } catch {
    // the body wasn't JSON: keep the status text
  }
  throw new ApiError(response.status, detail);
}

// api("/v1/apps")                              GET, returns the parsed JSON
// api("/v1/apps", { method: "POST", json })    sends JSON
// api(path, { method: "POST", form })          sends a FormData (file uploads)
export async function api(path, { method = "GET", json, form } = {}) {
  const options = { method, headers: headers() };
  if (json !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(json);
  } else if (form) {
    options.body = form; // the browser sets the multipart Content-Type itself
  }
  const response = await fetch(path, options);
  if (!response.ok) await fail(response);
  return response.status === 204 ? null : response.json();
}

// POST a JSON body and call onEvent(event) for each "data: {...}" line of the answer.
// Resolves when the server sends [DONE] or closes the stream.
export async function stream(path, json, onEvent) {
  const response = await fetch(path, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ ...json, stream: true }),
  });
  if (!response.ok) await fail(response);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    // Events are separated by a blank line. The last piece may be half an event: keep it.
    const events = buffer.split("\n\n");
    buffer = events.pop();
    for (const event of events) {
      const data = event.replace(/^data: /, "").trim();
      if (data === "[DONE]") return;
      if (data) onEvent(JSON.parse(data));
    }
  }
}
