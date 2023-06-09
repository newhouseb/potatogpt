import { Var, causalMask, gelu, getSlice, layerNorm, linear, merge, multiplyMatrix, softmax, split, tensor, transposeMatrix } from "./math"

test('Multiplication', () => {
    const left = tensor([4, 3], [
        1, 2, 3,
        4, 0, 5,
        6, 7, 8,
        9, 10, 11,
    ]);

    expect(left.data[2]).toBe(3)

    const double = tensor([3, 3], [
        2, 0, 0,
        0, 2, 0,
        0, 0, 2
    ]);

    const out = multiplyMatrix(left, double)

    expect(out.data[0]).toStrictEqual(2)
    expect(out.data[3]).toStrictEqual(8)
})

test('Transpose', () => {
    const a = tensor([2, 3], [
        1, 2, 3,
        4, 5, 6
    ]);

    const b = transposeMatrix(a);

    expect([...b.data]).toStrictEqual([
        1, 4, 2, 5, 3, 6
    ]);
})

test('Split & Merge', () => {
    const a = tensor([2, 6], [
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6
    ])

    const out = split(a, 2);

    expect([...out[0].data]).toStrictEqual([
        1, 2, 1, 2
    ])

    expect([...out[1].data]).toStrictEqual([
        3, 4, 3, 4
    ])

    expect([...out[2].data]).toStrictEqual([
        5, 6, 5, 6
    ])

    const merged = merge(out, 6)

    expect([...merged.data]).toStrictEqual([
        ...a.data
    ])
})

test('Causual Maks', () => {
    const m = causalMask(4);
    expect([...m.data]).toStrictEqual([
        0, -1e10, -1e10, -1e10,
        0,     0, -1e10, -1e10,
        0,     0,     0, -1e10,
        0,     0,     0,     0
    ])
})

test('Linear', () => {
    const a = tensor([3, 3], [
        2, 0, 0,
        0, 2, 0,
        0, 0, 2
    ]);

    const x = tensor([3, 3], [
        1, 2, 3,
        4, 0, 5,
        6, 7, 8,
    ]);

    const b = tensor([3], [
       1, 1, 1,
    ]);

    const out = linear(a, x, b);

    expect([...out.data]).toStrictEqual([
        3, 5, 7, 9, 1, 11, 13, 15, 17
    ])
});

test('Softmax', () => {
    const a = tensor([2, 2], [
        2, 100,
        -5, 0
    ]);

    const out = softmax(a)

    expect([...out.data].reduce((a, b) => a + b, 0)).toBeCloseTo(2.0)
})

test('Layer Norm', () => {
    const activations = tensor([2, 3], [
        2, 2, 3, 
        -5, 0, 1
    ]);

    const gain = tensor([3], [
        1, 1, 1,
    ]);

    const bias = tensor([3], [
        0, 0, 0,
    ]);

    const out = layerNorm(activations, gain, bias);
    expect([...out.data].reduce((a, b) => a + b, 0)).toBeCloseTo(0.0)
})

test('GELU', () => {
    const activations = tensor([3, 1], [
        -1000, 0, 1000
    ]);

    const out = gelu(activations)

    expect(out.data[0]).toBeCloseTo(0.0)
    expect(out.data[1]).toBeCloseTo(0.0)
    expect(out.data[2]).toBeCloseTo(1000.0)
})

test("Slicing", () => {
    const a = tensor([2, 6, 1], [
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12
    ])

    const out = getSlice(a, 0)
    expect([...out.data]).toStrictEqual([
        1, 2, 3, 4, 5, 6
    ])

    const out2 = getSlice(a, 1)
    expect([...out2.data]).toStrictEqual([
        7, 8, 9, 10, 11, 12
    ])

})

const seq = [];

const a = Var(seq.length, 'Sequence Length');

const tensorA = tensor([a, 4]);
const tensorB = tensor([4, 5]);


const tensorC = multiplyMatrix(tensorA, tensorB);

const tensor1 = tensor([3, 2]);
const tensor2 = tensor([4, 5]);

// @ts-expect-error
const out = multiplyMatrix(tensor1, tensor2);

const tensor3 = tensor([3, 4]);
const tensor4 = tensor([4, 5]);

const out2 = multiplyMatrix(tensor3, tensor4);

const bias = tensor([5]);
const add = linear(tensor3, tensor4, bias);

const sliced = getSlice(bias, 0)
