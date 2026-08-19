import assert from "node:assert/strict";
import { test } from "node:test";
import AdmZip from "adm-zip";
import { createHash } from "node:crypto";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { decode, encode, loadManifest, verifyAndRead, ModelFormatError } from "../src/imf.js";

test("encode/decode roundtrip (incl. multibyte)", () => {
  for (const text of ["hello", "ភាសា", "العربية", "עברית", "ไทย"]) {
    assert.equal(decode(encode(text)), text);
  }
});

test("ids follow the canonical table: byte+3, trailing EOS", () => {
  const ids = encode("A");
  assert.deepEqual(ids, [65 + 3, 1]);
});

async function makeZip(files: Record<string, Buffer>): Promise<string> {
  const dir = await mkdtemp(path.join(os.tmpdir(), "secryst-ts-"));
  const zip = new AdmZip();
  for (const [name, data] of Object.entries(files)) zip.addFile(name, data);
  const p = path.join(dir, "model.zip");
  zip.writeZip(p);
  return p;
}

test("manifest parse + sha256 verification", async () => {
  const enc = Buffer.from("fake-encoder");
  const dec = Buffer.from("fake-decoder");
  const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");
  const metadata = [
    "format: imf-v1", "tokenizer: bytes", "id: test-1.0", "task: g2p",
    "decoder: plain", `sha256:`, `  encoder.onnx: ${sha(enc)}`, `  decoder.onnx: ${sha(dec)}`,
  ].join("\n");
  const p = await makeZip({ "metadata.yaml": Buffer.from(metadata), "encoder.onnx": enc, "decoder.onnx": dec });
  const { manifest, graphs } = verifyAndRead(p);
  assert.equal(manifest.id, "test-1.0");
  assert.equal(graphs["encoder.onnx"].toString(), "fake-encoder");
});

test("tampered graph fails sha256 loudly", async () => {
  const enc = Buffer.from("fake-encoder");
  const dec = Buffer.from("fake-decoder");
  const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");
  const metadata = [
    "format: imf-v1", "tokenizer: bytes", "id: test-1.0", "task: g2p",
    `sha256:`, `  encoder.onnx: ${sha(enc)}`, `  decoder.onnx: ${sha(dec)}`,
  ].join("\n");
  const p = await makeZip({
    "metadata.yaml": Buffer.from(metadata), "encoder.onnx": Buffer.from("TAMPERED"), "decoder.onnx": dec,
  });
  assert.throws(() => verifyAndRead(p), ModelFormatError);
});

test("non-bytes tokenizer is rejected", async () => {
  const p = await makeZip({
    "metadata.yaml": Buffer.from("format: imf-v1\ntokenizer: chars\nid: x"),
    "encoder.onnx": Buffer.from("e"), "decoder.onnx": Buffer.from("d"),
  });
  assert.throws(() => loadManifest(p), /byte-level only/);
});
