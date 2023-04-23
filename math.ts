// Used to define a dynamically sized (at runtime) dimension
export type Var<N extends string> = number & { label: N };
export const Var = <L extends string>(d: number, label: L) => { return d as Var<L> };

// Used to ensure that there's not ambiguity that worms its way through the type system via Union
type UnionToIntersection<U> = (U extends any ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;
type IsUnion<T> = [T] extends [UnionToIntersection<T>] ? false : true;

// Utility types
type And<A, B> = A extends true ? B extends true ? true : false : false;
type Or<A, B> = A extends true ? true : B extends true ? true : false;

// Check that there is a literal number or an intentionally defined variable
type IsLiteralOrVar<T extends number | (string & number)> = And<IsUnion<T> extends true ? false : true,
                        Or<number extends T ? false : T extends number ? true : false,
                           T extends { label: string } & number ? true : false>>;

// Returns the same type if every element satisfied IsLiteralOrVar
type AsLiteralArray<T extends ReadonlyArray<number | ({ label: string } & number)>> = T extends ReadonlyArray<unknown>
  ? { [K in keyof T]: IsLiteralOrVar<T[K]> } extends { [K in keyof T]: true }
    ? T
    : never
  : never;

type Dim = number | number & { label: string };
export type Tensor<D extends readonly Dim[]> = {
    data: Float32Array
    shape: D
}

export function tensor<const D extends readonly (number | number & { label: string })[]>(
    d: D,
    init?: number[]
    ): D extends AsLiteralArray<D> ? Tensor<D> : [never, "Unlabeled dimension amidst", D] {
    return {
        data: new Float32Array(init || d.reduce((a, b) => a * b, 1) as any),
        shape: d
    } as any
}

function isTensor(a: any): a is Tensor<any> {
    return a && a.data && a.shape;
}

export function multiplyMatrix<const X extends Dim, const Y extends Dim, const Z extends Dim>(
    a: IsUnion<Y> extends true ? [never, "Differing types", Y] : Tensor<readonly [X, Y]>, 
    b: Tensor<readonly [Y, Z]>): Tensor<readonly [X, Z]> {
    if (!isTensor(a) || !isTensor(b)) {
        throw new Error("Invalid tensor");
    }

    const output = tensor<[X, Z]>([(a as any).shape[0], b.shape[1]]) as Tensor<[X, Z]>;

    for (let i = 0; i < a.shape[0]; i++) {
        for (let j = 0; j < b.shape[1]; j++) {
            let sum = 0;
            for (let k = 0; k < a.shape[1]; k++) {
                sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
            }
            output.data[i * output.shape[1] + j] = sum;
        }
    }
    
    return output as any;
}

export const transposeMatrix = <X extends number, Y extends number>
        (a: Tensor<readonly [X, Y]>): Tensor<readonly [Y, X]> => {

    const output = tensor<[Y, X]>([a.shape[1], a.shape[0]] as any) as Tensor<[Y, X]>;

    for (let i = 0; i < a.shape[0]; i++) {
        for (let j = 0; j < a.shape[1]; j++) {
            output.data[j * output.shape[1] + i] = a.data[i * a.shape[1] + j];
        }
    }
    
    return output;
}; 

export const unsqueeze = <D extends readonly Dim[]>
        (a: Tensor<D>): Tensor<PushHead<D, 1>> => {
    return {
        data: a.data,
        shape: [1, ...a.shape] as any
    } as any;
}

export function mapInPlace<D extends readonly Dim[]>(a: Tensor<D>, fn: (n: number) => number): Tensor<D> {
    a.data.set(a.data.map(i => fn(i)))
    return a;
}

export function split<D extends readonly Dim[], C extends number>
    (a: Tensor<D>, chunkSize: C): Tensor<PushTail<PopTail<D>, C>>[] {

    // Splits the last dimension into N sized chunks.
    // The intuition for this method's implementation is easiest to understand if you visualize
    // the memory layout of the tensor, i.e. let's say you have
    // 1 2 3 4 5 6 1 2 3 4 5 6 in a [2, 6] tensor and you want to split it by 3. If
    // you set up the output [2, 2] tensors contiguously, then we cycle through each, 2 at a time
    // and copy the output
    // 1: 1 2 1 2
    // 2: 3 4 3 4
    // 3: 5 6 5 6

    const stride = a.shape[a.shape.length - 1];
    if (stride % chunkSize !== 0) {
        throw new Error('Invalid chunk size, not evently divisible into last tensor dimension')
    }

    // Setup the output chunks
    const out = [] as Tensor<PushTail<PopTail<D>, C>>[];
    const chunks = stride / chunkSize;
    for (let c = 0; c < chunks; c++) {
        out.push(tensor([...a.shape.slice(0, a.shape.length - 1), chunkSize]) as any);
    }
    const outOffsets = out.map(_ => 0);
    let sourceOffset = 0;

    // Split up the actual data
    const macroChunks = a.data.length / stride;
    for (let i = 0; i < macroChunks; i++) {
        for (let j = 0; j < chunks; j++) {
            out[j].data.set(a.data.slice(sourceOffset, sourceOffset + chunkSize), outOffsets[j])
            outOffsets[j] += chunkSize;
            sourceOffset += chunkSize
        }
    }

    return out;
}

export function merge<D extends readonly Dim[], C extends number>
    (a: Tensor<D>[], mergedSize: C): Tensor<PushTail<PopTail<D>, C>> {
    const out: Tensor<PushTail<PopTail<D>, C>> = tensor([...a[0].shape.slice(0, a[0].shape.length - 1), mergedSize]) as any;

    const chunk = a[0].shape[a[0].shape.length - 1];
    if (mergedSize % chunk !== 0 || mergedSize !== chunk * a.length) {
        throw new Error('Incalid merged size, not a multiple of the last tensor dimension')
    }

    const inOffsets = a.map(_ => 0);
    let outOffset = 0;

    // Split up the actual data
    const macroChunks = out.data.length / mergedSize;
    for (let i = 0; i < macroChunks; i++) {
        for (let j = 0; j < a.length; j++) {
            //console.log(outOffset)
            out.data.set(a[j].data.slice(inOffsets[j], inOffsets[j] + chunk), outOffset)
            inOffsets[j] += chunk;
            outOffset += chunk
        }
    }

    return out;
}

export function causalMask<N extends Dim>(n: N): Tensor<[N, N]> {
  const empty: Tensor<[N, N]> = tensor([n,n]) as any;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      empty.data[i*n + j] = j > i ? -1e10 : 0;
    }
  }
  return empty;
}

export function addMatrix<D extends readonly Dim[]>(a: Tensor<D>, b: Tensor<D>): Tensor<D> {
    const data = a.data.map((x, i) => a.data[i] + b.data[i]);
    return {
        ...a, data
    } as Tensor<D>
}

export type Last<T extends readonly number[]> = ConstTuple<T> extends readonly [...readonly number[], infer V] ? V extends  number ? V : never : never;
export type Tail<T extends readonly number[]> = ConstTuple<T> extends readonly [number, ...infer U] ? U extends readonly number[] ? U : never : never;

export type PopTail<T extends readonly number[]> = ConstTuple<T> extends readonly [...infer U, number] ? U extends readonly number[] ? U : never : never;
export type PushTail<T extends readonly number[], A extends number> = [...T, A]
export type PushHead<T extends readonly number[], A extends number> = [A, ...T]

type ConstTuple<T extends readonly number[]> = T;


export function getSlice<const D extends readonly Dim[]>(a: Tensor<D>, idx: number): 
    Tensor<Tail<D>> {
    const stride = a.shape.slice(1).reduce((a, b) => a * b, 1);
    return {
        data: new Float32Array(a.data.buffer, 4 * idx * stride, stride),
        shape: a.shape.slice(1) as Tail<D>,
    }
}


export function copy<D extends readonly Dim[]>(params: {from: Tensor<D>, to: Tensor<D>}) {
    params.to.data.set(params.from.data)
}

export function linear<X extends Dim, Y extends Dim, Z extends Dim>(
    activations: IsUnion<Y> extends true ? [never, "Ambiguous dimension", Y] : Tensor<readonly [X, Y]>, 
    weights: IsUnion<Z> extends true ? [never, "Ambiguous dimension", Z] : Tensor<readonly [Y, Z]>, 
    bias: Tensor<readonly [Z]>): Tensor<readonly [X, Z]> {
    if (!isTensor(activations) || !isTensor(weights) || !isTensor(bias)) {
        throw new Error('Invalid parameters');
    }

    const intermediate = multiplyMatrix(activations as any, weights) as Tensor<readonly [X, Z]>

    for (let i = 0; i < intermediate.shape[0]; i++) {
        const s = getSlice(intermediate, i);
        s.data.set(addMatrix(s, bias).data);
    }

    return intermediate;
}

export function gelu<D extends readonly Dim[]>(a: Tensor<D>): Tensor<D> {
    return {
        shape: a.shape,
        data: a.data.map(x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x,3)))))
    } as Tensor<D>
}

export function softmax<D extends readonly Dim[]>(a: Tensor<D>): Tensor<D> {
    const lastStride = a.shape[a.shape.length - 1];
    const elementCount = a.data.length / lastStride;
    for (let i = 0; i < elementCount; i++) {
        const layer = new Float32Array(a.data.buffer, 4 * i * lastStride, lastStride);
        const max = (layer as any).reduce((a: number, b: number) => Math.max(a, b), -Infinity);
        const exp_x = layer.map(x => Math.exp(x - max));
        const sum_exp_x = (exp_x as any).reduce((a: number, b: number) => a + b, 0);
        const data = exp_x.map(x => x / sum_exp_x);
        layer.set(data);
    }

    return a;
}

export function layerNorm<D extends readonly Dim[]>(
        activations: Tensor<D>, 
        gain: Tensor<readonly [Last<D>]>, 
        bias: Tensor<readonly [Last<D>]>): Tensor<D> {
    let out = {
        ...activations,
        data: Float32Array.from(activations.data)
    } as Tensor<D>

    const lastStride = activations.shape[activations.shape.length - 1];
    const elementCount = activations.data.length / lastStride;
    for (let i = 0; i < elementCount; i++) {
        const layer = new Float32Array(out.data.buffer, 4 * i * lastStride, lastStride);
        const eps = 1e-5;
        const mean = (layer as any).reduce((a: number, b: number) => a + b, 0) / layer.length;
        const variance = (layer.map(x => Math.pow(x - mean, 2)) as any).reduce((a: number, b: number) => a + b, 0) / layer.length;
        layer.set(layer.map((x, i) => gain.data[i] * (x - mean) / Math.sqrt(variance + eps) + bias.data[i]));
    }

    return out;
}
