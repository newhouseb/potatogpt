# 🥔 PotatoGPT

This is a reimplementation of GPT2 (small) inference from scratch with no runtime dependencies (seriously, check package.json). Implementation was designed to be an educational exercise and was heavily based off of the wonderful [picoGPT](https://github.com/jaymody/picoGPT) repository. 

Because there are no accelerated math libraries under the hood (read: there's no javascript equivalent of numpy), this implementation is potato-slow: on the order of a few tokens per minute on a Macbook Air M2 (depending on prompt length).

![Screen Shot 2023-04-23 at 10 38 55 PM](https://user-images.githubusercontent.com/77915/233888362-a1fb784e-ae6b-43a3-9c8f-0bbebc605cc8.png)

## 🏃 How to Run

1. Clone the repository (weights are included, about 500mb or so, so it may take a bit)
2. Run `node main.js` (no `yarn install` needed unless you want to edit the code)

## 🌈 Fully Typed Tensors

The _novel_ part of this codebase is that it demonstrates type-safe tensor operations. Said another way: you no longer have to run your code to track and verify the shape of your tensors!

<img width="379" alt="Screen Shot 2023-04-23 at 6 14 01 PM" src="https://user-images.githubusercontent.com/77915/233869473-98d2e38f-a2ac-47b2-9fb9-72fb4f611708.png">

In the screenshot above, I'm multiplying a 3x4 matrix with a 4x5 matrix. The typesystem concludes that the output matrix is (correctly) 3x5.

But what happens if I screw up and try to multiply two matrixes that don't share the inner dimension?

<img width="678" alt="Screen Shot 2023-04-23 at 6 14 41 PM" src="https://user-images.githubusercontent.com/77915/233869607-1c6257fa-e327-4826-9477-2319d13e966b.png">

As you can see, typescript can detect this and provide an error message that there are differing types, 4 and 5. This way you can catch your error the instant you make it.

Another challenge is that you might have a dynamically sized tensor (i.e. based on the length of the input sequence). Typing this dimension as `number` would destroy the aforementioned typechecking because one `number` is indistinguishable for another `number`. The solution to this is to use branded types like so:

<img width="509" alt="Screen Shot 2023-04-23 at 6 42 03 PM" src="https://user-images.githubusercontent.com/77915/233870004-bd95475a-82c9-44c6-b49b-d947055042b2.png">

In this example, the first dimension of tensorA is only known at runtime. But because we can tag it as `Var<'Sequence Length'>`, all future uses will typecheck just as they would above: if you tried to intermingle differently branded numbers, typescript would yell.

Note that I've only implemented the bare minimum of tensor math to get GPT2 to work, representing a tiny fraction of what you would get out of something like numpy. Notably, there's nothing like broadcasting implemented.

PotatoGPT uses fully type-safe tensors everywhere.

## 🙋 FAQ

### What about training?

Many years ago, I wrote an autodifferentiating compiler in clojure so I have a reasonable idea of how training works. The primary point of _this_ exercise was to prove to myself that I understand the architecture of GPTs.

### How were the weights generated?

The weights were generated using the python file found in this repository (`export_weights.py`). I spent a while trying to figure out if I could read the raw tensorflow checkpoints but doing so without third party libraries is tough. Tensorflow stores weights in a vaguely LevelDB-like SSTable format "for efficiency" but after reading a bunch of enterprise-grade Google C++, I decided to just write each tensor to disk in a packed floats. This ended up being smaller than the "optimized" tensorflow checkpoint format and way faster to load than anything else I tried ¯\_(ツ)_/¯

### Can we make Tensorflow or PyTorch typesafe?

I'm not sure! Python looks to have generic types, but without conditional types it is tricky to have any meaningful dynamic behavior. I would love to be proven wrong here.
