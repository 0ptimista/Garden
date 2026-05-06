const KV_KEY = "popular.json";
const DEFAULT_RANGE_DAYS = 180;
const DEFAULT_LIMIT = 8;
const DEFAULT_METRICS_LIMIT = 200;
const DEFAULT_UMAMI_API_BASE = "https://api.umami.is/v1";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/popular.json") {
      return servePopularJson(env);
    }

    if (request.method === "POST" && url.pathname === "/popular.json") {
      return refreshFromRequest(request, env);
    }

    if (request.method === "GET" && url.pathname === "/health") {
      return jsonResponse({ ok: true });
    }

    return jsonResponse({ error: "not_found" }, { status: 404 });
  },

  async scheduled(_controller, env, ctx) {
    ctx.waitUntil(refreshPopularContent(env));
  },
};

async function servePopularJson(env) {
  const body = await env.POPULAR_CONTENT.get(KV_KEY);

  if (!body) {
    return jsonResponse({
      updatedAt: null,
      rangeDays: getInteger(env.POPULAR_RANGE_DAYS, DEFAULT_RANGE_DAYS),
      source: "umami",
      items: [],
    }, {
      headers: cacheHeaders(60),
    });
  }

  return new Response(body, {
    headers: {
      "content-type": "application/json; charset=utf-8",
      ...cacheHeaders(300),
    },
  });
}

async function refreshFromRequest(request, env) {
  if (!env.REFRESH_TOKEN) {
    return jsonResponse({ error: "manual_refresh_not_configured" }, { status: 404 });
  }

  const authorized = await hasValidBearerToken(request, env.REFRESH_TOKEN);
  if (!authorized) {
    return jsonResponse({ error: "unauthorized" }, { status: 401 });
  }

  const payload = await refreshPopularContent(env);
  return jsonResponse(payload, {
    headers: { "cache-control": "no-store" },
  });
}

async function refreshPopularContent(env) {
  const startedAt = new Date();
  const rangeDays = getInteger(env.POPULAR_RANGE_DAYS, DEFAULT_RANGE_DAYS);
  const limit = getInteger(env.POPULAR_LIMIT, DEFAULT_LIMIT);
  const metricsLimit = getInteger(env.UMAMI_METRICS_LIMIT, DEFAULT_METRICS_LIMIT);
  const endAt = startedAt.getTime();
  const startAt = endAt - rangeDays * 24 * 60 * 60 * 1000;

  assertRequired(env.UMAMI_API_KEY, "UMAMI_API_KEY");
  assertRequired(env.UMAMI_WEBSITE_ID, "UMAMI_WEBSITE_ID");
  assertRequired(env.SITE_ORIGIN, "SITE_ORIGIN");

  const [contentIndex, metrics] = await Promise.all([
    fetchContentIndex(env),
    fetchUmamiPathMetrics(env, { startAt, endAt, metricsLimit }),
  ]);

  const postsByPath = buildPostLookup(contentIndex.items || []);
  const totals = new Map();

  for (const metric of metrics) {
    const normalizedPath = normalizePath(metric.x);
    const post = postsByPath.get(normalizedPath);

    if (!post) {
      continue;
    }

    totals.set(post.url, {
      ...post,
      views: (totals.get(post.url)?.views || 0) + Number(metric.y || 0),
    });
  }

  const items = [...totals.values()]
    .filter((item) => item.views > 0)
    .sort((a, b) => b.views - a.views)
    .slice(0, limit)
    .map((item, index) => ({
      rank: index + 1,
      title: item.title,
      url: item.url,
      views: item.views,
    }));

  const payload = {
    updatedAt: startedAt.toISOString(),
    rangeDays,
    source: "umami",
    items,
  };

  await env.POPULAR_CONTENT.put(KV_KEY, JSON.stringify(payload, null, 2));

  console.log(JSON.stringify({
    event: "popular_content_refreshed",
    items: items.length,
    rangeDays,
  }));

  return payload;
}

async function fetchContentIndex(env) {
  const url = absoluteUrl(
    env.CONTENT_INDEX_URL || "/assets/data/content-index.json",
    env.SITE_ORIGIN,
  );
  const response = await fetch(url, {
    headers: { accept: "application/json" },
  });

  if (response.status === 404) {
    return fetchQuartoSearchIndex(env);
  }

  if (!response.ok) {
    throw new Error(`Failed to fetch content index: ${response.status}`);
  }

  return response.json();
}

async function fetchQuartoSearchIndex(env) {
  const response = await fetch(absoluteUrl("/search.json", env.SITE_ORIGIN), {
    headers: { accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch fallback search index: ${response.status}`);
  }

  const data = await response.json();
  const seen = new Set();
  const items = [];

  for (const entry of Array.isArray(data) ? data : []) {
    const href = String(entry.href || entry.objectID || "");

    if (!href || href.includes("#") || !href.startsWith("posts/")) {
      continue;
    }

    const url = href.startsWith("/") ? href : `/${href}`;
    if (seen.has(url)) {
      continue;
    }

    seen.add(url);
    items.push({
      title: entry.title || url,
      url,
      aliases: aliasesForUrl(url),
    });
  }

  return { items };
}

async function fetchUmamiPathMetrics(env, { startAt, endAt, metricsLimit }) {
  const apiBase = (env.UMAMI_API_BASE || DEFAULT_UMAMI_API_BASE).replace(/\/$/, "");
  const url = new URL(`${apiBase}/websites/${env.UMAMI_WEBSITE_ID}/metrics`);
  url.searchParams.set("startAt", String(startAt));
  url.searchParams.set("endAt", String(endAt));
  url.searchParams.set("type", "path");
  url.searchParams.set("limit", String(metricsLimit));

  const response = await fetch(url, {
    headers: {
      accept: "application/json",
      "x-umami-api-key": env.UMAMI_API_KEY,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch Umami metrics: ${response.status}`);
  }

  const data = await response.json();
  return Array.isArray(data) ? data : [];
}

function buildPostLookup(items) {
  const lookup = new Map();

  for (const item of items) {
    if (!item.url || !item.title) {
      continue;
    }

    for (const alias of item.aliases || [item.url]) {
      lookup.set(normalizePath(alias), item);
    }
  }

  return lookup;
}

function aliasesForUrl(url) {
  const aliases = new Set([url]);

  if (url.endsWith("/index.html")) {
    aliases.add(url.replace(/\/index\.html$/, "/"));
    aliases.add(url.replace(/\/index\.html$/, ""));
  } else if (url.endsWith(".html")) {
    aliases.add(url.replace(/\.html$/, ""));
  }

  return [...aliases];
}

function normalizePath(value) {
  if (!value) {
    return "/";
  }

  const url = new URL(value, "https://example.com");
  let pathname = decodeURIComponent(url.pathname);

  if (pathname.length > 1 && pathname.endsWith("/")) {
    pathname = pathname.slice(0, -1);
  }

  if (pathname.endsWith("/index.html")) {
    pathname = pathname.slice(0, -"/index.html".length);
  } else if (pathname.endsWith(".html")) {
    pathname = pathname.slice(0, -".html".length);
  }

  return pathname || "/";
}

function absoluteUrl(value, origin) {
  return new URL(value, origin).toString();
}

function getInteger(value, fallback) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : fallback;
}

function assertRequired(value, name) {
  if (!value) {
    throw new Error(`${name} is required`);
  }
}

async function hasValidBearerToken(request, expectedToken) {
  const authorization = request.headers.get("authorization") || "";
  const actualToken = authorization.startsWith("Bearer ") ? authorization.slice(7) : "";

  if (!actualToken || !expectedToken) {
    return false;
  }

  return timingSafeEqual(actualToken, expectedToken);
}

async function timingSafeEqual(left, right) {
  const encoder = new TextEncoder();
  const [leftHash, rightHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(left)),
    crypto.subtle.digest("SHA-256", encoder.encode(right)),
  ]);
  const leftBytes = new Uint8Array(leftHash);
  const rightBytes = new Uint8Array(rightHash);
  let difference = 0;

  for (let index = 0; index < leftBytes.length; index += 1) {
    difference |= leftBytes[index] ^ rightBytes[index];
  }

  return difference === 0;
}

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body, null, 2), {
    ...init,
    headers: {
      "content-type": "application/json; charset=utf-8",
      ...(init.headers || {}),
    },
  });
}

function cacheHeaders(maxAge) {
  return {
    "cache-control": `public, max-age=${maxAge}, s-maxage=${maxAge * 12}, stale-while-revalidate=86400`,
  };
}
