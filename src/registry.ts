/** models.yaml index resolution + verified cache. Env overrides:
 * SECRYST_INDEX (URL or path), SECRYST_CACHE (default ~/.cache/secryst). */
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, open as fsOpen, readFile, rename, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

export class RegistryError extends Error {}

export const ENV_INDEX = "SECRYST_INDEX";
export const ENV_CACHE = "SECRYST_CACHE";
export const DEFAULT_INDEX_URL =
  "https://raw.githubusercontent.com/interscript/interscript-ml/main/models.yaml";

export interface Part { url: string; sha256: string; size: number; }
export interface IndexEntry {
  id: string; filename: string; url: string; sha256: string; size: number;
  precision: string; task: string; parts: Part[];
}

export function cacheDir(): string {
  return process.env[ENV_CACHE] ?? path.join(os.homedir(), ".cache", "secryst");
}

export async function loadIndex(indexUrl?: string): Promise<Record<string, IndexEntry>> {
  const source = indexUrl ?? process.env[ENV_INDEX] ?? DEFAULT_INDEX_URL;
  const text = source.startsWith("http://") || source.startsWith("https://")
    ? await (await fetch(source)).text()
    : await readFile(source, "utf-8");
  const raw = parseYaml(text) as Record<string, any>;
  if (!raw || typeof raw !== "object" || raw.version !== 1) {
    throw new RegistryError("index must be a mapping with version: 1");
  }
  const entries: Record<string, IndexEntry> = {};
  for (const [id, spec] of Object.entries(raw.models ?? {}) as [string, any][]) {
    entries[id] = {
      id,
      filename: spec.filename,
      url: spec.url ?? "",
      sha256: spec.sha256,
      size: Number(spec.size ?? 0),
      precision: spec.precision ?? "fp32",
      task: spec.task ?? "",
      parts: (spec.parts ?? []).map((p: any) => ({ url: p.url, sha256: p.sha256, size: Number(p.size ?? 0) })),
    };
  }
  return entries;
}

async function sha256File(p: string): Promise<string> {
  return await new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    createReadStream(p).on("data", (c) => hash.update(c))
      .on("error", reject)
      .on("end", () => resolve(hash.digest("hex")));
  });
}

async function openChannel(url: string): Promise<Readable> {
  if (url.startsWith("file://")) {
    return createReadStream(new URL(url).pathname);
  }
  if (url.startsWith("http://") || url.startsWith("https://")) {
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new RegistryError(`fetch failed: ${url} -> ${res.status}`);
    return Readable.fromWeb(res.body as any);
  }
  return createReadStream(url);
}

/** Resolve a verified local zip for `model_id`, downloading into the
 * cache when needed. Cache hits are re-verified against the index. */
export async function resolve(modelId: string, indexUrl?: string): Promise<string> {
  const entries = await loadIndex(indexUrl);
  const entry = entries[modelId];
  if (!entry) throw new RegistryError(`unknown model id '${modelId}' (known: ${Object.keys(entries).sort()})`);
  const target = path.join(cacheDir(), "models", modelId, entry.filename);
  if (await sha256File(target).then((h) => h === entry.sha256, () => false)) return target;

  await mkdir(path.dirname(target), { recursive: true });
  const tmp = path.join(path.dirname(target), `.${process.pid}.part`);
  const handle = await fsOpen(tmp, "w");
  await handle.close();
  try {
    const verifyPart = async (url: string, expected?: string, label?: string) => {
      const hash = createHash("sha256");
      try {
        await pipeline(
          await openChannel(url),
          async function* (source) {
            for await (const chunk of source) { hash.update(chunk); yield chunk; }
          },
          createWriteStream(tmp, { flags: "a" }),
        );
      } catch (err) {
        throw new RegistryError(`download failed (${url}): ${(err as Error).message}`);
      }
      if (expected) {
        const actual = hash.digest("hex");
        if (actual !== expected) {
          throw new RegistryError(`part ${label} of ${entry.filename} sha256 mismatch`);
        }
      }
    };
    if (entry.parts.length > 0) {
      for (const [i, part] of entry.parts.entries()) {
        await verifyPart(part.url, part.sha256, String(i));
      }
    } else {
      await verifyPart(entry.url);
    }
    const whole = await sha256File(tmp);
    if (whole !== entry.sha256) {
      throw new RegistryError(`${entry.filename} sha256 mismatch: got ${whole}, index says ${entry.sha256}`);
    }
    await rename(tmp, target);
    return target;
  } finally {
    await rm(tmp, { force: true });
  }
}
