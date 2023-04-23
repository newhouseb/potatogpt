# PotatoGPT

This is a reimplementation of GPT2 (small) inference from scratch with no runtime dependencies (seriously, check package.json). Implementation was designed to be an educational exercise and was heavily based off of the wonderful [picoGPT](https://github.com/jaymody/picoGPT) repository. 

Because there are no accelerated math libraries under the hood (read: there's no javascript equivalent of numpy), this implementation is potato-slow: on the order of a couple tokens per minute on a Macbook Air M2.

```
(base) ben@Bens-MacBook-Air transformers % yarn start
yarn run v1.22.19
warning package.json: No license field
$ ts-node main.ts
loading gpt
done
Block done                     
Chosen token: [ 'peanut butter and', ' jelly' ]

...

Block done                     
Chosen token: [
  'peanut butter and jelly.\n' +
    '\n' +
    "I'm not sure if I'm going to be able to get this recipe to work for me, but I'm going to try it",
  '.'
]
```

## Fully Typed Tensors

The _novel_ part of this codebase is that it demonstrates type-safe tensor operations. Said another way: you no longer have to run your code to track and verify the shape of your tensors!

<img width="379" alt="Screen Shot 2023-04-23 at 6 14 01 PM" src="https://user-images.githubusercontent.com/77915/233869473-98d2e38f-a2ac-47b2-9fb9-72fb4f611708.png">

In the screenshot above, I'm multiplying a 3x4 matrix with a 4x5 matrix. The typesystem concludes that the output matrix is (correctly 3x5).

But what happens if I screw up and try to multiply two matrixes that don't share the inner dimension?

<img width="678" alt="Screen Shot 2023-04-23 at 6 14 41 PM" src="https://user-images.githubusercontent.com/77915/233869607-1c6257fa-e327-4826-9477-2319d13e966b.png">

As you can see, typescript can detect this and provide an error message that there are differing types, 4 and 5. This way you can catch your error the instant you make it.

Another challenge is that you might have a dynamically sized tensor (i.e. based on the length of the input sequence). Typing this dimension as `number` would destroy the aforementioned typechecking because one `number` is indistinguishable for another `number`. The solution to this is to use branded types like so:

<img width="509" alt="Screen Shot 2023-04-23 at 6 42 03 PM" src="https://user-images.githubusercontent.com/77915/233870004-bd95475a-82c9-44c6-b49b-d947055042b2.png">

In this example, the first dimension of tensorA is only known at runtime. But because we can tag it as `Var<'Sequence Length'>`, all future uses will typecheck just as they would above: if you tried to intermingle differently branded numbers, typescript would yell.

PotatoGPT uses fully type-safe tensors everywhere.

## FAQ

### What about training?

I've written an autodifferentiating compiler in clojure years ago so I have a pretty good idea of how that end of things works. The primary point of this exercise was to prove to myself that I understand the architecture of GPTs.

### How were the weights generated?

The weights were generated using the python notebook found in this repository. I spent a while trying to figure out if I could read the raw tensorflow checkpoints but doing so without third party libraries is tough. Tensorflow stores weights in a vaguely LevelDB-like SSTable format "for efficiency" but after reading a bunch of enterprise-grade Google C++, I decided to just write each tensor to disk in a structured format. This ended up being smaller than the "optimized" tensorflow checkpoint format and way faster to load than anything else I tried ¯\_(ツ)_/¯

### Can we make Tensorflow or PyTorch typesafe?

I'm not sure! Python looks to have generic types, but without conditional types it is tricky to have any meaningful dynamic behavior. I would love to be proven wrong here.
