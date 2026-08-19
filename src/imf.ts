/** Interscript Model Format v1 — loading + sha256 verification.
 * Token ids follow the canonical ByT5 table: byte b -> b+3 with a
 * trailing EOS; pad=0, unk=2. Ids are NOT raw byte values. */
import { createHash } from "node:crypto";
import AdmZip from "adm-zip";
import { parse as parseYaml } from "yaml";

export const BYTE_OFFSET = 3;
export const PAD_ID = 0;
export const EOS_ID = 1;
export const UNK_ID = 2;

export class ModelFormatError extends Error {}

export interface Manifest {
  id: string;
  task: string;
  decoder: string;
  precision: string;
  opset: number;
  sha256: Record<string, string>;
}

export function encode(text: string): number[] {
  const bytes = Buffer.from(text, "utf-8");
  const ids = new Array<number>(bytes.length + 1);
  for (let i = 0; i < bytes.length; i++) ids[i] = bytes[i]! + BYTE_OFFSET;
  ids[bytes.length] = EOS_ID;
  return ids;
}

export function decode(tokenIds: number[] | Int32Array): string {
  const out: number[] = [];
  for (const token of tokenIds) {
    if (token === EOS_ID) break;
    if (token === PAD_ID || token === UNK_ID) continue;
    out.push((token - BYTE_OFFSET) % 256);
  }
  return Buffer.from(out).toString("utf-8");
}

export function loadManifest(zipPath: string): Manifest {
  const zip = new AdmZip(zipPath);
  const names = new Set(zip.getEntries().map((e) => e.entryName));
  for (const required of ["metadata.yaml", "encoder.onnx", "decoder.onnx"]) {
    if (!names.has(required)) {
      throw new ModelFormatError(`missing required file: ${required}`);
    }
  }
  const raw = parseYaml(zip.readAsText("metadata.yaml")) as Record<string, unknown>;
  if (raw["format"] !== "imf-v1") {
    throw new ModelFormatError(`unsupported format: ${String(raw["format"])}`);
  }
  if (raw["tokenizer"] !== "bytes") {
    throw new ModelFormatError(`tokenizer ${String(raw["tokenizer"])}: this runtime is byte-level only`);
  }
  return {
    id: String(raw["id"]),
    task: String(raw["task"] ?? ""),
    decoder: String(raw["decoder"] ?? "plain"),
    precision: String(raw["precision"] ?? "fp32"),
    opset: Number(raw["opset"] ?? 14),
    sha256: (raw["sha256"] ?? {}) as Record<string, string>,
  };
}

export function verifyAndRead(zipPath: string): { manifest: Manifest; graphs: Record<string, Buffer> } {
  const manifest = loadManifest(zipPath);
  const zip = new AdmZip(zipPath);
  const graphs: Record<string, Buffer> = {};
  for (const entry of zip.getEntries()) {
    if (!entry.entryName.endsWith(".onnx")) continue;
    const data = entry.getData();
    const recorded = manifest.sha256[entry.entryName];
    if (!recorded) {
      throw new ModelFormatError(`${entry.entryName} is not covered by metadata sha256`);
    }
    const actual = createHash("sha256").update(data).digest("hex");
    if (actual !== recorded) {
      throw new ModelFormatError(
        `${entry.entryName} sha256 mismatch: zip has ${actual}, metadata says ${recorded}`,
      );
    }
    graphs[entry.entryName] = data;
  }
  return { manifest, graphs };
}
