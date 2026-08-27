import assert from "node:assert/strict";
import { test } from "node:test";
import AdmZip from "adm-zip";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { loadIndex, resolve, RegistryError, DEFAULT_INDEX_URL } from "../src/registry.js";

async function tmp(): Promise<string> {
  return await mkdtemp(path.join(os.tmpdir(), "secryst-reg-"));
}

test("loadIndex reads a local file index", async () => {
  const dir = await tmp();
  const idx = path.join(dir, "index.yaml");
  await writeFile(idx, "version: 1\nmodels:\n  khm-latn-1.0:\n    filename: m.zip\n    url: \"\"\n    sha256: abc\n");
  const entries = await loadIndex(idx);
  assert.ok(entries["khm-latn-1.0"]);
  assert.equal(entries["khm-latn-1.0"].filename, "m.zip");
});

test("index must declare version: 1", async () => {
  const dir = await tmp();
  const idx = path.join(dir, "index.yaml");
  await writeFile(idx, "version: 2\nmodels: {}\n");
  await assert.rejects(() => loadIndex(idx), RegistryError);
});

test("resolve downloads via file:// channel, verifies, caches", async () => {
  const dir = await tmp();
  const enc = Buffer.from("enc"); const dec = Buffer.from("dec");
  const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");
  const metadata = `format: imf-v1\ntokenizer: bytes\nid: r-1.0\nsha256:\n  encoder.onnx: ${sha(enc)}\n  decoder.onnx: ${sha(dec)}\n`;
  const zip = new AdmZip();
  zip.addFile("metadata.yaml", Buffer.from(metadata));
  zip.addFile("encoder.onnx", enc);
  zip.addFile("decoder.onnx", dec);
  const zipPath = path.join(dir, "r-1.0.zip");
  zip.writeZip(zipPath);
  const whole = await readFile(zipPath);
  const index = path.join(dir, "index.yaml");
  await writeFile(index, `version: 1\nmodels:\n  r-1.0:\n    filename: r-1.0.zip\n    url: file://${zipPath}\n    sha256: ${sha(whole)}\n`);
  process.env.SECRYST_CACHE = path.join(dir, "cache");
  const resolved = await resolve("r-1.0", index);
  assert.ok(resolved.endsWith("r-1.0.zip"));
  assert.ok(resolved.includes(path.join("models", "r-1.0")));
});

test("sha256 mismatch fails loudly", async () => {
  const dir = await tmp();
  const index = path.join(dir, "index.yaml");
  await writeFile(index, "version: 1\nmodels:\n  bad-1.0:\n    filename: b.zip\n    url: file:///nonexistent\n    sha256: deadbeef\n");
  process.env.SECRYST_CACHE = path.join(dir, "cache2");
  await assert.rejects(() => resolve("bad-1.0", index), RegistryError);
});

test("DEFAULT_INDEX_URL pins a GitHub Release asset, never raw", () => {
  assert.match(
    DEFAULT_INDEX_URL,
    /^https:\/\/github\.com\/interscript\/interscript-ml\/releases\/download\/index-v\d+\/models-index\.yaml$/,
  );
  assert.doesNotMatch(DEFAULT_INDEX_URL, /raw\.githubusercontent/);
});

test("HTTP index fetch verifies the .sha256 sidecar before parsing", async () => {
  const { createServer } = await import("node:http");
  const body = "version: 1\nmodels: {}\n";
  const good = createHash("sha256").update(body).digest("hex");
  const bad = "0".repeat(64);

  async function withServer(sidecar: string | null, fn: (url: string) => Promise<void>) {
    const server = createServer((req, res) => {
      if (req.url === "/models-index.yaml") {
        res.writeHead(200); res.end(body); return;
      }
      if (req.url === "/models-index.yaml.sha256") {
        if (sidecar === null) { res.writeHead(404); res.end(); return; }
        res.writeHead(200); res.end(`${sidecar}  models-index.yaml\n`); return;
      }
      res.writeHead(404); res.end();
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    const { port } = server.address() as { port: number };
    try { await fn(`http://127.0.0.1:${port}/models-index.yaml`); }
    finally { await new Promise<void>((r) => server.close(() => r())); }
  }

  await withServer(good, async (url) => {
    const entries = await loadIndex(url);
    assert.deepEqual(entries, {});
  });
  await withServer(bad, async (url) => {
    await assert.rejects(() => loadIndex(url), /index sha256 mismatch/);
  });
  await withServer(null, async (url) => {
    await assert.rejects(() => loadIndex(url), /index sha256/);
  });
});
