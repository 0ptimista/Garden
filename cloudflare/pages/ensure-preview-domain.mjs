const {
  CLOUDFLARE_API_TOKEN,
  CLOUDFLARE_ACCOUNT_ID,
  CF_PAGES_PROJECT = "luozx-garden-preview",
  CF_PAGES_DOMAIN = "preview.luozx.org",
  CF_PAGES_PRODUCTION_BRANCH = "preview",
} = process.env;

const apiBase = "https://api.cloudflare.com/client/v4";

if (!CLOUDFLARE_API_TOKEN || !CLOUDFLARE_ACCOUNT_ID) {
  throw new Error("Missing CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID.");
}

async function cloudflare(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${CLOUDFLARE_API_TOKEN}`,
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.success === false) {
    const messages = (body.errors || [])
      .map((error) => `${error.code}: ${error.message}`)
      .join("; ");
    const error = new Error(messages || `Cloudflare API request failed with ${response.status}.`);
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return body;
}

const projectPath = `/accounts/${CLOUDFLARE_ACCOUNT_ID}/pages/projects/${CF_PAGES_PROJECT}`;

await cloudflare(projectPath, {
  method: "PATCH",
  body: JSON.stringify({
    production_branch: CF_PAGES_PRODUCTION_BRANCH,
  }),
});

try {
  await cloudflare(`${projectPath}/domains/${CF_PAGES_DOMAIN}`, {
    method: "GET",
  });
  console.log(`${CF_PAGES_DOMAIN} is already attached to ${CF_PAGES_PROJECT}.`);
} catch (error) {
  if (error.status !== 404) {
    throw error;
  }

  await cloudflare(`${projectPath}/domains`, {
    method: "POST",
    body: JSON.stringify({
      name: CF_PAGES_DOMAIN,
    }),
  });
  console.log(`Attached ${CF_PAGES_DOMAIN} to ${CF_PAGES_PROJECT}.`);
}
