import assert from "node:assert/strict";
import { test } from "node:test";
import AdmZip from "adm-zip";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { loadIndex, resolve, RegistryError } from "../src/registry.js";

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
