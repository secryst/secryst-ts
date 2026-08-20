import { Byt5 } from "./byt5.js";
import { resolve } from "./registry.js";

export interface Model {
  readonly id: string;
  translate(text: string, maxSeqLength?: number): Promise<string>;
}

export async function load(modelId: string, indexUrl?: string): Promise<Model> {
  const zipPath = await resolve(modelId, indexUrl);
  return Byt5.load(zipPath);
}
