# Garden Popular Content Worker

This Worker refreshes `/popular.json` once per day from Umami path metrics and stores the public JSON payload in Workers KV.

## Setup

1. Create a KV namespace:

   ```sh
   npx wrangler@latest kv namespace create POPULAR_CONTENT
   ```

2. Copy the returned namespace id into `wrangler.jsonc`.

3. Store the Umami API key as a Worker secret:

   ```sh
   npx wrangler@latest secret put UMAMI_API_KEY
   ```

4. Optional: store a manual refresh token so you can populate KV immediately after deploy:

   ```sh
   npx wrangler@latest secret put REFRESH_TOKEN
   ```

5. Deploy:

   ```sh
   npx wrangler@latest deploy
   ```

The route in `wrangler.jsonc` serves `https://luozx.org/popular.json`. The scheduled trigger runs at `18:20 UTC`, which is `03:20` in Asia/Tokyo.

## Local Checks

Run the Worker locally and trigger the scheduled handler:

```sh
npx wrangler@latest dev --test-scheduled
```

Then open:

```text
http://localhost:8787/__scheduled
```

The Worker reads the public content index from `https://luozx.org/assets/data/content-index.json`, calls Umami's `metrics?type=path` endpoint, keeps only paths that match indexed posts, and writes the top items to KV under `popular.json`.

If `REFRESH_TOKEN` is configured, you can refresh production manually:

```sh
curl -X POST https://luozx.org/popular.json \
  -H "Authorization: Bearer $REFRESH_TOKEN"
```
