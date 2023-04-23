# PotatoGPT(2)

This is a reimplementation of GPT2 (small) inference from scratch with no runtime dependencies (seriously, check package.json). Implementation was designed to be an educational exercise and was heavily based off of the wonderful [picoGPT](https://github.com/jaymody/picoGPT) repository. Because there are no accelerated math libraries under the hood (read: a javascript equivalent of numpy), this implementation is potato-slow: on the order of a couple tokens per minute on a Macbook Air M2.

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




## FAQ

### What about training?

I've written an autodifferentiating compiler in clojure years ago so I have a pretty good idea of how that end of things works. The primary point of this exercise was to prove to myself that I understand the architecture of GPTs.

### How were the weights generated?

The weights were generated using the python notebook found in this repository. I spent a while trying to figure out if I could read the raw tensorflow checkpoints but doing so without third party libraries is tough. Tensorflow stores weights in a vaguely LevelDB-like SSTable format "for efficiency" but after reading a bunch of enterprise-grade Google C++, I decided to just write each tensor to disk in a structured format. This ended up being smaller than the "optimized" tensorflow checkpoint format and way faster to load than anything else I tried ¯\_(ツ)_/¯