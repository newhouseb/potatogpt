import { Var, addMatrix, causalMask, copy, gelu, getSlice, layerNorm, linear, mapInPlace, merge, multiplyMatrix, softmax, split, tensor, transposeMatrix } from "./math";
import * as fs from 'fs';
import type { Tensor } from "./math";
import { createInterface } from 'readline';
import { stdin as input, stdout as output } from 'process';

async function main() {
  // Read prompt from stdin 
  const rl = createInterface({ input, output });
  let prompt = await new Promise<string>((resolve) => {
    rl.question('Please enter a prompt: ', (input: string) => {
      resolve(input);
      rl.close();
    });
  });

  // Map the prompt to tokens
  let tokens = encodeString(prompt)

  console.time('Loading Model')
  const gpt = await loadSmallGPT();
  console.timeEnd('Loading Model')

  // Generate (up to) 100 tokens
  let toGenerate = 100;
  while (toGenerate > 0) {
    const start = (new Date()).getTime();

    // Map each token into an embedding + position vector
    let x = tensor([Var(tokens.length, 'Sequence Length'), gpt.EmbeddingDimensions])
    tokens.map((token, i) => {
      const slice = getSlice(x, i)
      const embedding = getSlice(gpt.weights.wte, token);
      const position = getSlice(gpt.weights.wpe, i);
      copy({ from: 
        addMatrix(embedding, position), 
        to: slice})
    })

    let i = 0;
    for (const block of gpt.weights.blocks) {
      i += 1
      process.stdout.write('\rBlock ' + i + ' of ' + gpt.weights.blocks.length);

      // The start of a block kicks off with a layer normalization
      const nx1 = layerNorm(x, block.ln_1.g, block.ln_1.b)

      // We weight the inputs to self attention 
      const kqv = linear(nx1, block.attn.c_attn.w, block.attn.c_attn.b)

      // Split out the k, q, and v tensors
      const [q, k, v] = split(kqv, gpt.EmbeddingDimensions);

      // Next split out each of the heads
      const headWidth = Divide(gpt.EmbeddingDimensions, gpt.AttentionHeads);
      const kHeads = split(k, headWidth);
      const qHeads = split(q, headWidth);
      const vHeads = split(v, headWidth);
      const aHeads = [] as Tensor<readonly [Var<'Sequence Length'>, typeof headWidth]>[];

      // Perform (masked) self-attention for each head
      const sqrtD = Math.sqrt(gpt.EmbeddingDimensions / gpt.AttentionHeads);
      const mask = causalMask(kHeads[0].shape[0]);
      for (let h = 0; h < gpt.AttentionHeads; h++) {
        aHeads.push(multiplyMatrix(
          softmax(addMatrix(
              mapInPlace(
                multiplyMatrix(qHeads[h], transposeMatrix(kHeads[h])), 
                (n) => n / sqrtD), 
              mask)), 
          vHeads[h]));
      }

      // Next merge the heads all back together
      const a = merge(aHeads, gpt.EmbeddingDimensions);

      // Project the attention back into the embedding space and add to our residuals
      x = addMatrix(x, linear(a, block.attn.c_proj.w, block.attn.c_proj.b))

      // Do our second layer norm
      const nx2 = layerNorm(x, block.ln_2.g, block.ln_2.b)

      // Project up to 4x wide, run gelu activate, project back down
      x = addMatrix(x, linear(gelu(linear(nx2, block.mlp.c_fc.w, block.mlp.c_fc.b)), block.mlp.c_proj.w, block.mlp.c_proj.b))

      process.stdout.write('\rBlock done                     ');
    }

    // Do the final layer norm
    x = layerNorm(x, gpt.weights.ln_f.g, gpt.weights.ln_f.b);
    const transposed = transposeMatrix(gpt.weights.wte)
    const final = multiplyMatrix(x, transposed);

    // Greedily choose the highest logit and use that as our next token
    let logits = getSlice(final, final.shape[0] - 1);
    let bestToken = 0;
    let bestScore = -Infinity;
    for (let j = 0; j < logits.data.length; j++) {
      if (logits.data[j] > bestScore) {
        bestScore = logits.data[j];
        bestToken = j;
      }
    }

    // Log the chosen token and prepare to loop again
    const duration = ((new Date()).getTime() - start) / 1000;
    console.log(`\nChose token in ${duration.toFixed(2)}s:`, [prompt, decodeTokens([bestToken])]);
    prompt += '' + decodeTokens([bestToken]);
    tokens.push(bestToken);

    toGenerate -= 1;
  }
}
main();

const loadSmallGPT = async () => {
  const model = GPT({
    VocabularySize: 50257, 
    SequenceLength: 1024, 
    EmbeddingDimensions: 768, 
    AttentionHeads: 12, 
    Layers: 12
  }, null)
  
  return model;
}

type Multiply<A extends number, B extends number> = number & { label: `${A} * ${B}` }
const Multiply = <A extends number, B extends number>(a: A, b: B) => a * b as Multiply<A, B>;
type Divide<A extends number, B extends number> = number & { label: `${A} / ${B}` }
const Divide = <A extends number, B extends number>(a: A, b: B) => a / b as Divide<A, B>;

function GPT<
    SequenceLength extends number, 
    VocabSize extends number,
    EmbeddingDimensions extends number,
    AttentionHeads extends number,
    Layers extends number
>(params: {
        SequenceLength: SequenceLength,
        VocabularySize: VocabSize,
        EmbeddingDimensions: EmbeddingDimensions,
        AttentionHeads: AttentionHeads,
        Layers: Layers,
    }, weights: any) {

  function load(path: string) {
    // It appears that readFileSync reuses buffers without realizing that
    // we've established views into them. So we need to copy the buffer.
    const buffer = fs.readFileSync(path);
    const newBuffer = new ArrayBuffer(buffer.length);
    const toReturn = new Float32Array(newBuffer);
    toReturn.set(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4));
    return toReturn;
  } 
  
  // Git LFS kicks in at 100MB, so we split this up into two
  const wte0 = load('weights/wte.0');
  const wte1 = load('weights/wte.1');
  const wte = new Float32Array(wte0.length + wte1.length);
  wte.set(wte0, 0);
  wte.set(wte1, wte0.length);

    return {
        ...params,
        weights: {
            // wpe stands for "word position embedding"
            wpe: tensor([params.SequenceLength, params.EmbeddingDimensions], load('weights/wpe') as any),
            // wte stands for "word token embedding"
            wte: tensor([params.VocabularySize, params.EmbeddingDimensions], wte as any),
            // ln_f stands for "layer normalization for the feedforward network"
            ln_f: {
                b: tensor([params.EmbeddingDimensions], load('weights/ln_f_b') as any),
                g: tensor([params.EmbeddingDimensions], load('weights/ln_f_g') as any)
            },
            blocks: [...Array(params.Layers).keys()].map((i: number) =>
                ({
                    attn: {
                        c_attn: {
                            // b and w stands for "bias" and "weight"
                            b: tensor([Multiply(3, params.EmbeddingDimensions)] as const, load(`weights/blocks_${i}_attn_c_attn_b`) as any),
                            w: tensor([params.EmbeddingDimensions, Multiply(3, params.EmbeddingDimensions)] as const, load(`weights/blocks_${i}_attn_c_attn_w`) as any)
                        },
                        c_proj: {
                            b: tensor([params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_attn_c_proj_b`) as any),
                            w: tensor([params.EmbeddingDimensions, params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_attn_c_proj_w`) as any)
                        }
                    },
                    // ln_1 stands for "layer normalization 1"
                    ln_1: { 
                        // b and g stands for "bias" and "gain"
                        b: tensor([params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_ln_1_b`) as any), 
                        g: tensor([params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_ln_1_g`) as any)
                    },
                    // mlp stands for "multi-layer perceptron"
                    mlp: {
                        // c_fc stands for full connected
                        c_fc: {
                            b: tensor([Multiply(4, params.EmbeddingDimensions)] as const, load(`weights/blocks_${i}_mlp_c_fc_b`) as any),
                            w: tensor([params.EmbeddingDimensions, Multiply(4, params.EmbeddingDimensions)] as const, load(`weights/blocks_${i}_mlp_c_fc_w`) as any)
                        },
                        // c_proj stands for "projection"
                        c_proj: {
                            b: tensor([params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_mlp_c_proj_b`) as any),
                            w: tensor([Multiply(4, params.EmbeddingDimensions), params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_mlp_c_proj_w`) as any)
                        }
                    },
                    // ln_2 stands for "layer normalization 2"
                    ln_2: { 
                        // b and g stands for "bias" and "gain"
                        b: tensor([params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_ln_2_b`) as any), 
                        g: tensor([params.EmbeddingDimensions] as const, load(`weights/blocks_${i}_ln_2_g`) as any) 
                    },
            })
        )}
    }
}

function bytesToUnicode(): [{ [key: number]: string }, { [key: string]: number }] {
  const bs: number[] = [
      ...Array.from({ length: '~'.charCodeAt(0) - '!'.charCodeAt(0) + 1 }, (_, i) => '!'.charCodeAt(0) + i),
      ...Array.from({ length: '¬'.charCodeAt(0) - '¡'.charCodeAt(0) + 1 }, (_, i) => '¡'.charCodeAt(0) + i),
      ...Array.from({ length: 'ÿ'.charCodeAt(0) - '®'.charCodeAt(0) + 1 }, (_, i) => '®'.charCodeAt(0) + i),
  ];
  const cs: number[] = [...bs];
  let n = 0;
  for (let b = 0; b < 2 ** 8; b++) {
      if (!bs.includes(b)) {
          bs.push(b);
          cs.push(2 ** 8 + n);
          n += 1;
      }
  }
  const csStr: string[] = cs.map((n) => String.fromCharCode(n));
  const lookupTable: { [key: number]: string } = {};
  const unlookupTable: { [key: string]: number } = {};
  bs.forEach((key, index) => {
      lookupTable[key] = csStr[index];
      unlookupTable[csStr[index]] = key;
  });
  return [lookupTable, unlookupTable];
}

function encodeString(str: string) {
  // This is a giant dict of strings that map to token IDs
  const encoder = JSON.parse(fs.readFileSync('weights/encoder.json', 'utf8'));

  // A weird quirk of GPT2's tokenization is that they map control and whitespace characters up by 255 to make them printable, not entirely
  // clear why this is but perhaps so that everything can confidently be printable while debugging without things (for example) clearing your terminal
  const [byteMapping, _] = bytesToUnicode();
  str = str.split('').map((c) => byteMapping[c.charCodeAt(0)]).join('');

  const tokens = Object.keys(encoder);
  let out = [] as number[];

  while (str.length) {
    let bestToken = '';
    for (const token of tokens) {
      if (str.startsWith(token) && token.length > bestToken.length) {
        bestToken = token;
      }
    }
    out.push(encoder[bestToken])
    str = str.slice(bestToken.length);
  }

  return out;
}

function decodeString(str: string) {
  const [_, byteUnmapping] = bytesToUnicode();
  return str.split('').map((c) => String.fromCharCode(byteUnmapping[c])).join('');
}

function decodeTokens(tokens: number[]) {
  const encoder = JSON.parse(fs.readFileSync('weights/encoder.json', 'utf8'));

  const decoder = {} as { [key: number]: string };
  for (const key in encoder) {
    decoder[encoder[key]] = key;
  }

  return decodeString(tokens.map((token) => decoder[token]).join(''));
}
