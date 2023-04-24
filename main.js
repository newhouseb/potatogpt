var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// math.ts
var Var = (d, label) => {
  return d;
};
function tensor(d, init) {
  return {
    data: new Float32Array(init || d.reduce((a, b) => a * b, 1)),
    shape: d
  };
}
function isTensor(a) {
  return a && a.data && a.shape;
}
function multiplyMatrix(a, b) {
  if (!isTensor(a) || !isTensor(b)) {
    throw new Error("Invalid tensor");
  }
  const output2 = tensor([a.shape[0], b.shape[1]]);
  for (let i = 0; i < a.shape[0]; i++) {
    for (let j = 0; j < b.shape[1]; j++) {
      let sum = 0;
      for (let k = 0; k < a.shape[1]; k++) {
        sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
      }
      output2.data[i * output2.shape[1] + j] = sum;
    }
  }
  return output2;
}
var transposeMatrix = (a) => {
  const output2 = tensor([a.shape[1], a.shape[0]]);
  for (let i = 0; i < a.shape[0]; i++) {
    for (let j = 0; j < a.shape[1]; j++) {
      output2.data[j * output2.shape[1] + i] = a.data[i * a.shape[1] + j];
    }
  }
  return output2;
};
function mapInPlace(a, fn) {
  a.data.set(a.data.map((i) => fn(i)));
  return a;
}
function split(a, chunkSize) {
  const stride = a.shape[a.shape.length - 1];
  if (stride % chunkSize !== 0) {
    throw new Error("Invalid chunk size, not evently divisible into last tensor dimension");
  }
  const out = [];
  const chunks = stride / chunkSize;
  for (let c = 0; c < chunks; c++) {
    out.push(tensor([...a.shape.slice(0, a.shape.length - 1), chunkSize]));
  }
  const outOffsets = out.map((_) => 0);
  let sourceOffset = 0;
  const macroChunks = a.data.length / stride;
  for (let i = 0; i < macroChunks; i++) {
    for (let j = 0; j < chunks; j++) {
      out[j].data.set(a.data.slice(sourceOffset, sourceOffset + chunkSize), outOffsets[j]);
      outOffsets[j] += chunkSize;
      sourceOffset += chunkSize;
    }
  }
  return out;
}
function merge(a, mergedSize) {
  const out = tensor([...a[0].shape.slice(0, a[0].shape.length - 1), mergedSize]);
  const chunk = a[0].shape[a[0].shape.length - 1];
  if (mergedSize % chunk !== 0 || mergedSize !== chunk * a.length) {
    throw new Error("Incalid merged size, not a multiple of the last tensor dimension");
  }
  const inOffsets = a.map((_) => 0);
  let outOffset = 0;
  const macroChunks = out.data.length / mergedSize;
  for (let i = 0; i < macroChunks; i++) {
    for (let j = 0; j < a.length; j++) {
      out.data.set(a[j].data.slice(inOffsets[j], inOffsets[j] + chunk), outOffset);
      inOffsets[j] += chunk;
      outOffset += chunk;
    }
  }
  return out;
}
function causalMask(n) {
  const empty = tensor([n, n]);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      empty.data[i * n + j] = j > i ? -1e10 : 0;
    }
  }
  return empty;
}
function addMatrix(a, b) {
  const data = a.data.map((x, i) => a.data[i] + b.data[i]);
  return {
    ...a,
    data
  };
}
function getSlice(a, idx) {
  const stride = a.shape.slice(1).reduce((a2, b) => a2 * b, 1);
  return {
    data: new Float32Array(a.data.buffer, 4 * idx * stride, stride),
    shape: a.shape.slice(1)
  };
}
function copy(params) {
  params.to.data.set(params.from.data);
}
function linear(activations, weights, bias) {
  if (!isTensor(activations) || !isTensor(weights) || !isTensor(bias)) {
    throw new Error("Invalid parameters");
  }
  const intermediate = multiplyMatrix(activations, weights);
  for (let i = 0; i < intermediate.shape[0]; i++) {
    const s = getSlice(intermediate, i);
    s.data.set(addMatrix(s, bias).data);
  }
  return intermediate;
}
function gelu(a) {
  return {
    shape: a.shape,
    data: a.data.map((x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))))
  };
}
function softmax(a) {
  const lastStride = a.shape[a.shape.length - 1];
  const elementCount = a.data.length / lastStride;
  for (let i = 0; i < elementCount; i++) {
    const layer = new Float32Array(a.data.buffer, 4 * i * lastStride, lastStride);
    const max = layer.reduce((a2, b) => Math.max(a2, b), -Infinity);
    const exp_x = layer.map((x) => Math.exp(x - max));
    const sum_exp_x = exp_x.reduce((a2, b) => a2 + b, 0);
    const data = exp_x.map((x) => x / sum_exp_x);
    layer.set(data);
  }
  return a;
}
function layerNorm(activations, gain, bias) {
  let out = {
    ...activations,
    data: Float32Array.from(activations.data)
  };
  const lastStride = activations.shape[activations.shape.length - 1];
  const elementCount = activations.data.length / lastStride;
  for (let i = 0; i < elementCount; i++) {
    const layer = new Float32Array(out.data.buffer, 4 * i * lastStride, lastStride);
    const eps = 1e-5;
    const mean = layer.reduce((a, b) => a + b, 0) / layer.length;
    const variance = layer.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / layer.length;
    layer.set(layer.map((x, i2) => gain.data[i2] * (x - mean) / Math.sqrt(variance + eps) + bias.data[i2]));
  }
  return out;
}

// main.ts
var fs = __toESM(require("fs"));
var import_readline = require("readline");
var import_process = require("process");
async function main() {
  const rl = (0, import_readline.createInterface)({ input: import_process.stdin, output: import_process.stdout });
  let prompt = await new Promise((resolve) => {
    rl.question("Please enter a prompt: ", (input2) => {
      resolve(input2);
      rl.close();
    });
  });
  let tokens = encodeString(prompt);
  console.time("Loading Model");
  const gpt = await loadSmallGPT();
  console.timeEnd("Loading Model");
  let toGenerate = 100;
  while (toGenerate > 0) {
    const start = (/* @__PURE__ */ new Date()).getTime();
    let x = tensor([Var(tokens.length, "Sequence Length"), gpt.EmbeddingDimensions]);
    tokens.map((token, i2) => {
      const slice = getSlice(x, i2);
      const embedding = getSlice(gpt.weights.wte, token);
      const position = getSlice(gpt.weights.wpe, i2);
      copy({
        from: addMatrix(embedding, position),
        to: slice
      });
    });
    let i = 0;
    for (const block of gpt.weights.blocks) {
      i += 1;
      process.stdout.write("\rBlock " + i + " of " + gpt.weights.blocks.length);
      const nx1 = layerNorm(x, block.ln_1.g, block.ln_1.b);
      const kqv = linear(nx1, block.attn.c_attn.w, block.attn.c_attn.b);
      const [q, k, v] = split(kqv, gpt.EmbeddingDimensions);
      const headWidth = Divide(gpt.EmbeddingDimensions, gpt.AttentionHeads);
      const kHeads = split(k, headWidth);
      const qHeads = split(q, headWidth);
      const vHeads = split(v, headWidth);
      const aHeads = [];
      const sqrtD = Math.sqrt(gpt.EmbeddingDimensions / gpt.AttentionHeads);
      const mask = causalMask(kHeads[0].shape[0]);
      for (let h = 0; h < gpt.AttentionHeads; h++) {
        aHeads.push(multiplyMatrix(
          softmax(addMatrix(
            mapInPlace(
              multiplyMatrix(qHeads[h], transposeMatrix(kHeads[h])),
              (n) => n / sqrtD
            ),
            mask
          )),
          vHeads[h]
        ));
      }
      const a = merge(aHeads, gpt.EmbeddingDimensions);
      x = addMatrix(x, linear(a, block.attn.c_proj.w, block.attn.c_proj.b));
      const nx2 = layerNorm(x, block.ln_2.g, block.ln_2.b);
      x = addMatrix(x, linear(gelu(linear(nx2, block.mlp.c_fc.w, block.mlp.c_fc.b)), block.mlp.c_proj.w, block.mlp.c_proj.b));
      process.stdout.write("\rBlock done                     ");
    }
    x = layerNorm(x, gpt.weights.ln_f.g, gpt.weights.ln_f.b);
    const transposed = transposeMatrix(gpt.weights.wte);
    const final = multiplyMatrix(x, transposed);
    let logits = getSlice(final, final.shape[0] - 1);
    let bestToken = 0;
    let bestScore = -Infinity;
    for (let j = 0; j < logits.data.length; j++) {
      if (logits.data[j] > bestScore) {
        bestScore = logits.data[j];
        bestToken = j;
      }
    }
    const duration = ((/* @__PURE__ */ new Date()).getTime() - start) / 1e3;
    console.log(`
Chose token in ${duration.toFixed(2)}s:`, [prompt, decodeTokens([bestToken])]);
    prompt += "" + decodeTokens([bestToken]);
    tokens.push(bestToken);
    toGenerate -= 1;
  }
}
main();
var loadSmallGPT = async () => {
  const model = GPT({
    VocabularySize: 50257,
    SequenceLength: 1024,
    EmbeddingDimensions: 768,
    AttentionHeads: 12,
    Layers: 12
  }, null);
  return model;
};
var Multiply = (a, b) => a * b;
var Divide = (a, b) => a / b;
function GPT(params, weights) {
  function load(path) {
    const buffer = fs.readFileSync(path);
    const newBuffer = new ArrayBuffer(buffer.length);
    const toReturn = new Float32Array(newBuffer);
    toReturn.set(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4));
    return toReturn;
  }
  const wte0 = load("weights/wte.0");
  const wte1 = load("weights/wte.1");
  const wte = new Float32Array(wte0.length + wte1.length);
  wte.set(wte0, 0);
  wte.set(wte1, wte0.length);
  return {
    ...params,
    weights: {
      // wpe stands for "word position embedding"
      wpe: tensor([params.SequenceLength, params.EmbeddingDimensions], load("weights/wpe")),
      // wte stands for "word token embedding"
      wte: tensor([params.VocabularySize, params.EmbeddingDimensions], wte),
      // ln_f stands for "layer normalization for the feedforward network"
      ln_f: {
        b: tensor([params.EmbeddingDimensions], load("weights/ln_f_b")),
        g: tensor([params.EmbeddingDimensions], load("weights/ln_f_g"))
      },
      blocks: [...Array(params.Layers).keys()].map(
        (i) => ({
          attn: {
            c_attn: {
              // b and w stands for "bias" and "weight"
              b: tensor([Multiply(3, params.EmbeddingDimensions)], load(`weights/blocks_${i}_attn_c_attn_b`)),
              w: tensor([params.EmbeddingDimensions, Multiply(3, params.EmbeddingDimensions)], load(`weights/blocks_${i}_attn_c_attn_w`))
            },
            c_proj: {
              b: tensor([params.EmbeddingDimensions], load(`weights/blocks_${i}_attn_c_proj_b`)),
              w: tensor([params.EmbeddingDimensions, params.EmbeddingDimensions], load(`weights/blocks_${i}_attn_c_proj_w`))
            }
          },
          // ln_1 stands for "layer normalization 1"
          ln_1: {
            // b and g stands for "bias" and "gain"
            b: tensor([params.EmbeddingDimensions], load(`weights/blocks_${i}_ln_1_b`)),
            g: tensor([params.EmbeddingDimensions], load(`weights/blocks_${i}_ln_1_g`))
          },
          // mlp stands for "multi-layer perceptron"
          mlp: {
            // c_fc stands for full connected
            c_fc: {
              b: tensor([Multiply(4, params.EmbeddingDimensions)], load(`weights/blocks_${i}_mlp_c_fc_b`)),
              w: tensor([params.EmbeddingDimensions, Multiply(4, params.EmbeddingDimensions)], load(`weights/blocks_${i}_mlp_c_fc_w`))
            },
            // c_proj stands for "projection"
            c_proj: {
              b: tensor([params.EmbeddingDimensions], load(`weights/blocks_${i}_mlp_c_proj_b`)),
              w: tensor([Multiply(4, params.EmbeddingDimensions), params.EmbeddingDimensions], load(`weights/blocks_${i}_mlp_c_proj_w`))
            }
          },
          // ln_2 stands for "layer normalization 2"
          ln_2: {
            // b and g stands for "bias" and "gain"
            b: tensor([params.EmbeddingDimensions], load(`weights/blocks_${i}_ln_2_b`)),
            g: tensor([params.EmbeddingDimensions], load(`weights/blocks_${i}_ln_2_g`))
          }
        })
      )
    }
  };
}
function bytesToUnicode() {
  const bs = [
    ...Array.from({ length: "~".charCodeAt(0) - "!".charCodeAt(0) + 1 }, (_, i) => "!".charCodeAt(0) + i),
    ...Array.from({ length: "\xAC".charCodeAt(0) - "\xA1".charCodeAt(0) + 1 }, (_, i) => "\xA1".charCodeAt(0) + i),
    ...Array.from({ length: "\xFF".charCodeAt(0) - "\xAE".charCodeAt(0) + 1 }, (_, i) => "\xAE".charCodeAt(0) + i)
  ];
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 2 ** 8; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(2 ** 8 + n);
      n += 1;
    }
  }
  const csStr = cs.map((n2) => String.fromCharCode(n2));
  const lookupTable = {};
  const unlookupTable = {};
  bs.forEach((key, index) => {
    lookupTable[key] = csStr[index];
    unlookupTable[csStr[index]] = key;
  });
  return [lookupTable, unlookupTable];
}
function encodeString(str) {
  const encoder = JSON.parse(fs.readFileSync("weights/encoder.json", "utf8"));
  const [byteMapping, _] = bytesToUnicode();
  str = str.split("").map((c) => byteMapping[c.charCodeAt(0)]).join("");
  const tokens = Object.keys(encoder);
  let out = [];
  while (str.length) {
    let bestToken = "";
    for (const token of tokens) {
      if (str.startsWith(token) && token.length > bestToken.length) {
        bestToken = token;
      }
    }
    out.push(encoder[bestToken]);
    str = str.slice(bestToken.length);
  }
  return out;
}
function decodeString(str) {
  const [_, byteUnmapping] = bytesToUnicode();
  return str.split("").map((c) => String.fromCharCode(byteUnmapping[c])).join("");
}
function decodeTokens(tokens) {
  const encoder = JSON.parse(fs.readFileSync("weights/encoder.json", "utf8"));
  const decoder = {};
  for (const key in encoder) {
    decoder[encoder[key]] = key;
  }
  return decodeString(tokens.map((token) => decoder[token]).join(""));
}
