export type Precision = 'int8' | 'fp32';

export type DimensionalArray<P extends Precision, D extends readonly number[]> = { 
    data: Int8Array ,
    precision: 'int8',
    shape: D 
} | {
    data: Float32Array,
    precision: 'fp32',
    shape: D
};

type IsLiteralNumber<T> = T extends number ? (number extends T ? false : true) : false;

type ArrayOfLiteralNumbersOrNever<T> = T extends ReadonlyArray<number>
  ? { [K in keyof T]: IsLiteralNumber<T[K]> } extends { [K in keyof T]: true }
    ? T
    : never
  : never;

export function emptyTensor<P extends Precision, const D extends readonly number[]>(
        p: P, 
        d: D
    ): D extends ArrayOfLiteralNumbersOrNever<D> ? DimensionalArray<P, D> : [never, "DimensionalArray shape must be a tuple of literal numbers"] {
    if (p === 'int8') {
        return {
            data: new Int8Array(d.reduce((a, b) => a * b, 1)),
            precision: p,
            shape: d
        } as any
    }
    if (p === 'fp32') {
        return {
            data: new Float32Array(d.reduce((a, b) => a * b, 1)),
            precision: p,
            shape: d
        } as any
    }
    throw new Error('Invalid precision');
}

export function tensor<P extends Precision, const D extends readonly number[]>(
        p: P, 
        d: D,
        init: number[]
    ): D extends ArrayOfLiteralNumbersOrNever<D> ? DimensionalArray<P, D> : [never, "DimensionalArray shape must be a tuple of literal numbers"] {
    if (p === 'int8') {
        return {
            data: new Int8Array(init),
            precision: p,
            shape: d
        } as any
    }
    if (p === 'fp32') {
        return {
            data: new Float32Array(init),
            precision: p,
            shape: d
        } as any
    }
    throw new Error('Invalid precision');
}

type UnionToIntersection<U> = (U extends any ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;
type IsUnion<T> = [T] extends [UnionToIntersection<T>] ? false : true;
type IsLiteral<T> = T extends number ? (number extends T ? false : true) : false;
type AnyUnion<T> = { [K in keyof T]: [T[K]] extends [UnionToIntersection<T[K]>] ? false : true } extends { [K in keyof T]: false } ? false : true;

function isDimensionArray(a: any): a is DimensionalArray<any, any> {
    return a && a.data && a.precision && a.shape;
}

type CheckDims<T, A> = AnyUnion<T> extends false ? A : [never, "Matrix dimensions did not typecheck"];

export const multiplyMatrix = <P extends Precision, 
    X extends number, Y extends number, Z extends number>(
    a: CheckDims<[X, Y, Z], DimensionalArray<P, readonly [X, Y]>>,
    b: DimensionalArray<P, readonly [Y, Z]>):
    CheckDims<[X, Y, Z], DimensionalArray<P, readonly [X, Z]>> => {
    if (!isDimensionArray(a)) {
        throw new Error('Invalid parameters');
    }
        
    const output = emptyTensor<P, [X, Z]>(a.precision as P, [a.shape[0], b.shape[1]] as const as any) as DimensionalArray<P, [X, Z]>;

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
}; 

export const transposeMatrix = <P extends Precision, X extends number, Y extends number>
        (a: DimensionalArray<P, readonly [X, Y]>): DimensionalArray<P, readonly [Y, X]> => {

    const output = emptyTensor<P, [Y, X]>(a.precision as P, [a.shape[1], a.shape[0]] as any) as DimensionalArray<P, [Y, X]>;

    for (let i = 0; i < a.shape[0]; i++) {
        for (let j = 0; j < a.shape[1]; j++) {
            output.data[j * output.shape[1] + i] = a.data[i * a.shape[1] + j];
        }
    }
    
    return output;
}; 

export const unsqueeze = <P extends Precision, D extends readonly number[]>
        (a: DimensionalArray<P, D>): DimensionalArray<P, PushHead<D, 1>> => {
    return {
        data: a.data,
        precision: a.precision,
        shape: [1, ...a.shape] as any
    } as any;
}

export function mapInPlace<P extends Precision, D extends readonly number[]>(a: DimensionalArray<P, D>, fn: (n: number) => number): DimensionalArray<P, D> {
    a.data.set(a.data.map(i => fn(i)))
    return a;
}

export function split<P extends Precision, D extends readonly number[], C extends number>
    (a: DimensionalArray<P, D>, chunkSize: C): DimensionalArray<P, PushTail<PopTail<D>, C>>[] {

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
    const out = [] as DimensionalArray<P, PushTail<PopTail<D>, C>>[];
    const chunks = stride / chunkSize;
    for (let c = 0; c < chunks; c++) {
        out.push(emptyTensor(a.precision, [...a.shape.slice(0, a.shape.length - 1), chunkSize]) as any);
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

export function merge<P extends Precision, D extends readonly number[], C extends number>
    (a: DimensionalArray<P, D>[], mergedSize: C): DimensionalArray<P, PushTail<PopTail<D>, C>> {
    const out: DimensionalArray<P, PushTail<PopTail<D>, C>> = emptyTensor(a[0].precision, [...a[0].shape.slice(0, a[0].shape.length - 1), mergedSize]) as any;

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

export function causalMask<N extends number>(n: N): DimensionalArray<'fp32', [N, N]> {
  const empty: DimensionalArray<'fp32', [N, N]> = emptyTensor('fp32', [n,n]) as any;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      empty.data[i*n + j] = j > i ? -1e10 : 0;
    }
  }
  return empty;
}

export function addMatrix<P extends Precision, D extends readonly number[]>(a: DimensionalArray<P, D>, b: DimensionalArray<P, D>): DimensionalArray<P, D> {
    const data = a.data.map((x, i) => a.data[i] + b.data[i]);
    return {
        ...a, data
    } as DimensionalArray<P, D>
}

export type Last<T extends readonly number[]> = ConstTuple<T> extends readonly [...readonly number[], infer V] ? V extends  number ? V : never : never;
export type Tail<T extends readonly number[]> = ConstTuple<T> extends readonly [number, ...infer U] ? U extends readonly number[] ? U : never : never;

export type PopTail<T extends readonly number[]> = ConstTuple<T> extends readonly [...infer U, number] ? U extends readonly number[] ? U : never : never;
export type PushTail<T extends readonly number[], A extends number> = [...T, A]
export type PushHead<T extends readonly number[], A extends number> = [A, ...T]

type ConstTuple<T extends readonly number[]> = T;


export function getSlice<P extends Precision, const D extends readonly number[]>(a: DimensionalArray<P, D>, idx: number): 
    DimensionalArray<P, Tail<D>> {

    const stride = a.shape.slice(1).reduce((a, b) => a * b, 1);
    
    if (a.precision === 'int8') {
        return {
            data: new Int8Array(a.data.buffer, idx * stride, stride),
            precision: a.precision,
            shape: a.shape.slice(1) as Tail<D>,
        }
    }
    if (a.precision === 'fp32') {
        //console.log(idx, stride, a.data.buffer.byteLength)
        return {
            data: new Float32Array(a.data.buffer, 4 * idx * stride, stride),
            precision: a.precision,
            shape: a.shape.slice(1) as Tail<D>,
        }
    }
    throw new Error('Invalid precision');
}


export function copy<P extends Precision, D extends readonly number[]>(params: {from: DimensionalArray<P, D>, to: DimensionalArray<P, D>}) {
    params.to.data.set(params.from.data)
}

export function linear<P extends Precision, X extends number, Y extends number, Z extends number>(
    activations: CheckDims<[X, Y, Z], DimensionalArray<P, readonly [X, Y]>>, 
    weights: DimensionalArray<P, readonly [Y, Z]>, 
    bias: DimensionalArray<P, readonly [Z]>): DimensionalArray<P, readonly [X, Z]> {
    if (!isDimensionArray(activations)) {
        throw new Error('Invalid parameters');
    }
    const intermediate = multiplyMatrix(activations, weights) as DimensionalArray<P, readonly [X, Z]>

    for (let i = 0; i < intermediate.shape[0]; i++) {
        const s = getSlice(intermediate, i);
        s.data.set(addMatrix(s, bias).data);
    }

    return intermediate;
}

export function gelu<P extends Precision, D extends readonly number[]>(a: DimensionalArray<P, D>): DimensionalArray<P, D> {
    return {
        precision: a.precision,
        shape: a.shape,
        data: a.data.map(x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x,3)))))
    } as DimensionalArray<P, D>
}

export function softmax<P extends Precision, D extends readonly number[]>(a: DimensionalArray<P, D>): DimensionalArray<P, D> {
    const lastStride = a.shape[a.shape.length - 1];
    const elementCount = a.data.length / lastStride;
    for (let i = 0; i < elementCount; i++) {
        const layer = a.precision === 'int8' ? 
            new Int8Array(a.data.buffer, i * lastStride, lastStride) : 
            new Float32Array(a.data.buffer, 4 * i * lastStride, lastStride);

        const max = (layer as any).reduce((a: number, b: number) => Math.max(a, b), -Infinity);
        const exp_x = layer.map(x => Math.exp(x - max));
        const sum_exp_x = (exp_x as any).reduce((a: number, b: number) => a + b, 0);
        const data = exp_x.map(x => x / sum_exp_x);
        layer.set(data);
    }

    return a;
    
    /*
    return {
        ...a, data
    } as DimensionalArray<P, D>
    */
}

export function layerNorm<P extends Precision, D extends readonly number[]>(
        activations: DimensionalArray<P, D>, 
        gain: DimensionalArray<P, readonly [Last<D>]>, 
        bias: DimensionalArray<P, readonly [Last<D>]>): DimensionalArray<P, D> {
    let out = {
        ...activations,
        data: activations.precision === 'int8' ? 
            Int8Array.from(activations.data) :
            Float32Array.from(activations.data)
    } as DimensionalArray<P, D>

    const lastStride = activations.shape[activations.shape.length - 1];
    const elementCount = activations.data.length / lastStride;
    for (let i = 0; i < elementCount; i++) {
        const layer = activations.precision === 'int8' ? 
            new Int8Array(out.data.buffer, i * lastStride, lastStride) : 
            new Float32Array(out.data.buffer, 4 * i * lastStride, lastStride);
    
        const eps = 1e-5;
        const mean = (layer as any).reduce((a: number, b: number) => a + b, 0) / layer.length;
        const variance = (layer.map(x => Math.pow(x - mean, 2)) as any).reduce((a: number, b: number) => a + b, 0) / layer.length;
        layer.set(layer.map((x, i) => gain.data[i] * (x - mean) / Math.sqrt(variance + eps) + bias.data[i]));
    }

    return out;
}



export type Multiply<A extends number, B extends number> = number;
