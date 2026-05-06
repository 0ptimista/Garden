import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const POSTS_DIR = "posts";
const OUTPUT_FILE = "assets/data/content-index.json";

const scalarKeys = new Set(["title", "description", "date", "draft"]);

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      files.push(...await walk(fullPath));
    } else if (entry.isFile() && entry.name.endsWith(".qmd")) {
      files.push(fullPath);
    }
  }

  return files;
}

function extractFrontMatter(source) {
  if (!source.startsWith("---\n")) {
    return "";
  }

  const end = source.indexOf("\n---", 4);
  return end === -1 ? "" : source.slice(4, end);
}

function unquote(value) {
  const trimmed = value.trim();

  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }

  return trimmed;
}

function parseFrontMatter(frontMatter) {
  const metadata = {};

  for (const line of frontMatter.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) {
      continue;
    }

    const [, key, rawValue] = match;
    if (!scalarKeys.has(key)) {
      continue;
    }

    metadata[key] = unquote(rawValue);
  }

  return metadata;
}

function toOutputUrl(sourcePath) {
  const withoutExt = sourcePath.replace(/\.qmd$/, "").replaceAll(path.sep, "/");
  return `/${withoutExt}.html`;
}

function aliasesFor(url) {
  const aliases = new Set([url]);

  if (url.endsWith("/index.html")) {
    aliases.add(url.replace(/\/index\.html$/, "/"));
    aliases.add(url.replace(/\/index\.html$/, ""));
  } else if (url.endsWith(".html")) {
    aliases.add(url.replace(/\.html$/, ""));
  }

  return [...aliases];
}

const files = await walk(POSTS_DIR);
const items = [];

for (const file of files) {
  const source = await readFile(file, "utf8");
  const metadata = parseFrontMatter(extractFrontMatter(source));

  if (metadata.draft === "true") {
    continue;
  }

  const url = toOutputUrl(file);
  items.push({
    title: metadata.title || path.basename(file, ".qmd"),
    description: metadata.description || "",
    date: metadata.date || "",
    url,
    aliases: aliasesFor(url),
    sourcePath: file.replaceAll(path.sep, "/"),
  });
}

items.sort((a, b) => String(b.date).localeCompare(String(a.date)));

await mkdir(path.dirname(OUTPUT_FILE), { recursive: true });
await writeFile(
  OUTPUT_FILE,
  `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    items,
  }, null, 2)}\n`,
);

console.log(`Wrote ${OUTPUT_FILE} with ${items.length} posts.`);
