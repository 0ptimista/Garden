import { copyFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

const siteBootstrapDir = path.join("_site", "site_libs", "bootstrap");
const sourceFont = path.join("assets", "fonts", "bootstrap-icons-subset.woff2");
const targetFont = path.join(siteBootstrapDir, "bootstrap-icons-subset.woff2");
const targetCss = path.join(siteBootstrapDir, "bootstrap-icons.css");

const icons = {
  "arrow-up": "f148",
  "circle-half": "f288",
  "envelope-fill": "f32c",
  facebook: "f344",
  github: "f3ed",
  "link-45deg": "f470",
  moon: "f497",
  rss: "f522",
  search: "f52a",
  "sort-down": "f575",
  sun: "f5a2",
  telegram: "f5b3",
  "twitter-x": "f8db",
};

if (!existsSync(targetCss) || !existsSync(sourceFont)) {
  process.exit(0);
}

const rules = Object.entries(icons)
  .map(([name, codepoint]) => `.bi-${name}::before { content: "\\${codepoint}"; }`)
  .join("\n");

const css = `@font-face {
  font-display: block;
  font-family: "bootstrap-icons";
  src: url("./bootstrap-icons-subset.woff2") format("woff2");
}

.bi::before,
[class^="bi-"]::before,
[class*=" bi-"]::before {
  display: inline-block;
  font-family: "bootstrap-icons" !important;
  font-style: normal;
  font-weight: normal !important;
  font-variant: normal;
  line-height: 1;
  text-transform: none;
  vertical-align: -.125em;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

${rules}

.quarto-color-scheme-toggle .bi::before { content: "\\${icons.moon}"; }
.quarto-color-scheme-toggle.alternate .bi::before { content: "\\${icons.sun}"; }
`;

await copyFile(sourceFont, targetFont);
await writeFile(targetCss, css);
await rm(path.join(siteBootstrapDir, "bootstrap-icons.woff"), { force: true });
