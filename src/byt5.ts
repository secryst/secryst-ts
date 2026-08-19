/** ByT5-family ONNX inference: greedy decode, KV-cache when the zip
 * ships decoder-kv.onnx, plain full-recompute otherwise. Ported from
 * the Ruby crystal (secryst/secryst lib/secryst/byt5_onnx.rb). */
import { EOS_ID, PAD_ID, Manifest, decode, encode, verifyAndRead } from "./imf.js";

export class Byt5 {
  private ort: any;
  private encoder: any;
  private decoder: any;
  private kv: boolean;
  private pastNames: string[] = [];
  readonly manifest: Manifest;

  constructor(zipPath: string) {
    const { manifest, graphs } = verifyAndRead(zipPath);
    this.manifest = manifest;
    // Lazy peer: importing onnxruntime-node only at model load keeps
    // registry/manifest consumers dependency-free.
    this.ort = require("onnxruntime-node");
    this.kv = manifest.decoder === "kv" && graphs["decoder-kv.onnx"] !== undefined;
    this.encoder = this.ort.InferenceSession.fromBuffer(graphs["encoder.onnx"]);
    this.decoder = this.ort.InferenceSession.fromBuffer(
      this.kv ? graphs["decoder-kv.onnx"] : graphs["decoder.onnx"],
    );
    this.pastNames = this.decoder.inputNames.filter((n: string) => n.startsWith("past_"));
  }

  get id(): string {
    return this.manifest.id;
  }

  async translate(text: string, maxSeqLength = 256): Promise<string> {
    const ids = encode(text);
    if (ids.length === 1) return "";
    const hidden = (
      await this.encoder.run({
        input_ids: new this.ort.Tensor("int32", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      })
    )["last_hidden_state"];
    const tokens = this.kv
      ? await this.greedyKv(hidden, maxSeqLength)
      : await this.greedyPlain(hidden, maxSeqLength);
    return decode(tokens);
  }

  private argmaxLastRow(logits: any): number {
    const dims: number[] = logits.dims;
    const seq = dims[dims.length - 2];
    const vocab = dims[dims.length - 1];
    const data = logits.data as Float32Array;
    const base = (seq - 1) * vocab;
    let best = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < vocab; i++) {
      const v = data[base + i];
      if (v > bestVal) { bestVal = v; best = i; }
    }
    return best;
  }

  private async greedyPlain(hidden: any, maxSeqLength: number): Promise<number[]> {
    const decoderIds: number[] = [PAD_ID];
    const generated: number[] = [];
    for (let step = 0; step < maxSeqLength; step++) {
      const out = await this.decoder.run({
        input_ids: new this.ort.Tensor(
          "int32", BigInt64Array.from(decoderIds.map(BigInt)), [1, decoderIds.length]),
        encoder_hidden_states: hidden,
      });
      const token = this.argmaxLastRow(out["logits"]);
      if (token === EOS_ID) break;
      generated.push(token);
      decoderIds.push(token);
    }
    return generated;
  }

  private async greedyKv(hidden: any, maxSeqLength: number): Promise<number[]> {
    const meta: any = (this.decoder as any).inputMetadata ?? {};
    let pasts: Record<string, any> = {};
    for (const name of this.pastNames) {
      const shape: (number | string)[] = meta[name]?.shape ?? [];
      const heads = typeof shape[1] === "number" ? shape[1] : 4;
      const dKv = typeof shape[3] === "number" ? shape[3] : 8;
      pasts[name] = new this.ort.Tensor("float32", new Float32Array(0), [1, heads, 0, dKv]);
    }
    let current = [PAD_ID];
    const generated: number[] = [];
    for (let step = 0; step < maxSeqLength; step++) {
      const results = await this.decoder.run({
        input_ids: new this.ort.Tensor("int32", BigInt64Array.from(current.map(BigInt)), [1, current.length]),
        encoder_hidden_states: hidden,
        ...pasts,
      });
      const token = this.argmaxLastRow(results["logits"]);
      if (token === EOS_ID) break;
      generated.push(token);
      for (const name of this.pastNames) {
        pasts[name] = results[name.replace("past_", "present_")];
      }
      current = [token];
    }
    return generated;
  }
}
