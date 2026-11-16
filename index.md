# I wrote a neural network framework in C++ and trained a tiny LM on it

My name is Eli Levasseur and I'm a first-year computer science and math student at uOttawa.
Since July I've been writing a neural network framework in C++ with no outside math or ML libraries,
because I wanted to understand how every piece works instead of making just another PyTorch data science project.
This month I used it to train a small language model on children's stories.
Here's what it writes when I give it the start of a story:

```text
Once upon a time, there was a little boy named Tim. He loved to go to the dog.
He saw a knowin felt a big smile. She was very excited. She cat not like came
to the door too. They are don
```

The first sentence makes sense but after that it becomes apparent the model doesn't understand context.
This is why it is version 1.0, this post is about my journey to this point and what comes next.

## Starting with the simplest version

My first commit was a small simple hardcoded multilayer perceptron written with way too many nested
`std::vector`: sigmoid layers, binary cross-entropy, and plain SGD one sample at
a time, with the backward pass written out by hand.

That model is frozen now and it is now used as the reference. In the tests whenever the
newer code produces a number, I compared it against what the old model gives for
the same weights and input.

## The framework

The real framework, nnet, is built around a `Tensor`: a flat array of doubles
with a shape, where the leading dimensions up to the final row and col dimensions count as batch dimensions. On top of
that it has:

- operations as plain functions (matrix multiply, bias add, ReLU, sigmoid,
  softmax cross-entropy, embedding lookup), each with its own gradient function
- autograd, where every operation leaves behind a small record of what it did,
  and `backward(loss)` walks those records from the loss back to every weight
- layers (`Embedding`, `Dense`, `ReLU`, `Sigmoid`) that chain together in a
  `Sequential`
- SGD, momentum and Adam optimizers
- checkpoints to save and load a trained model

Because of autograd, the system I built to track every operation and send the gradients back to the weights
automatically, layers don't have backward methods at all.
Here's the language model's architecture from the top of this post, built with the neural network framework:

```cpp
std::mt19937 weightGenerator(42);
std::vector<std::unique_ptr<nnet::Unary_Module>> components;
components.push_back(std::make_unique<nnet::Embedding>(256, 16, weightGenerator));
components.push_back(std::make_unique<nnet::Dense>(32 * 16, 128, weightGenerator));
components.push_back(std::make_unique<nnet::ReLU>());
components.push_back(std::make_unique<nnet::Dense>(128, 64, weightGenerator));
components.push_back(std::make_unique<nnet::ReLU>());
components.push_back(std::make_unique<nnet::Dense>(64, 256, weightGenerator));
nnet::Sequential model(std::move(components));
```

And one training step:

```cpp
// windowIds is [128, 32] byte ids, nextCharacters is [128, 256] one hot encoded rows
nnet::AdamOptimizer optimizer(model.parameters(), 0.002);

model.zeroGrad();
nnet::Value prediction = model.forward(nnet::makeLeaf(windowIds, false));
nnet::Value loss = nnet::softmaxCrossEntropy(prediction, nnet::makeLeaf(nextCharacters, false));
nnet::backward(loss);
optimizer.step();
```

There are 420 checks across four test suites. The parity tests work out every
gradient three ways: with the old reference model, with a backward pass built by
hand from tensor operations, and with autograd. All three have to agree to
within 1e-12 error rate. Everything builds without warnings under
`-Wall -Wextra -Wpedantic` and passes under AddressSanitizer and
UndefinedBehaviorSanitizer. 
I did use AI for some of the tests since thorough parity testing is new to me.

## Making it fast enough to train a model

Training a language model multiplies the same few matrix shapes millions of
times, so the internal matrix multiply system decides how long everything takes. The biggest one
in my model is a batch of 128 rows by 512 times a 512 by 128 weight.

My first matmul used the textbook loop order, (i-j-k), and ran at 3.33 GFLOP/s.
Swapping the two inner loops to i-k-j got it to 30.0 GFLOP/s. That's 9 times
faster for exactly the same math.

The reason why is memory. Each matrix is stored one row after another. With i-j-k,
the innermost loop walks down a column of the right-hand matrix, jumping 128
doubles (1 KB) on every step, so almost every read misses the cache. With
i-k-j, the innermost loop walks along a row of the right-hand matrix and a row
of the output which is what a CPU cache is built for.
Compiling with `-march=native` almost doubled training speed on top of that.

Then I tried threads to split the workload. Big matrix multiplies split their rows across 16 threads, but
starting 16 threads costs about 0.85 ms on my machine, so for small multiplies one thread is
faster. On my Ryzen 9 9950X the point where it became beneficial was about 16 million multiply-adds,
so I set the cutoff at 32 million. My language model's biggest multiply is 8.4
million, which means it still trains on one core. An epoch on it takes about 11
minutes.

30 GFLOP/s is still a long way from OpenBLAS, which does about 113 on one core
for the same shape. Closing that gap is what I'm working on next.

## GLM: a language model implementation on top

GLM reads text as raw bytes, so its vocabulary is the 256 possible byte values
and there's no word list. It simply looks at 32 characters and predicts the next one.
To write, it predicts a character, adds it to the end of the text,
slides the window forward by one and repeats.

I train it on a 22.5 MB subset of TinyStories, a dataset of short children's
stories with a small vocabulary. The first 90% is for training, which is about
20 million windows. Copying every window as doubles would take about 5 GB, so a
window is just a position in the stored text.

The model is

```text
Embedding(256, 16) -> Dense(512, 128) -> ReLU -> Dense(128, 64) -> ReLU -> Dense(64, 256)
```

which is 94,656 parameters, trained with Adam and softmax cross entropy.

An untrained model scores a loss of about 5.55. That's ln 256, the loss you'd get by guessing evenly between all 256 bytes.
My earlier model with one hidden layer got down to 1.20 on validation data, with training loss at 1.18. Since
e^1.20 is about 3.3, it's about as unsure as choosing between three or four
characters. Training and validation being that close means it isn't memorizing;
it's just full. Version 0.1, with the second hidden layer, reached 1.125 loss per character.

Before TinyStories I trained it on Alice's Adventures in Wonderland alone. It
stopped at 1.78 and started memorizing, because one book isn't enough text to
learn much past short common words.

To test version 1.0, I gave it "Once upon a time, there was a little" ten times.
All ten continued with a english looking sentence, almost all sentences did not have a made up word."
Two of the responses were:

```text
Once upon a time, there was a little girl named Amy. He saw it was toy cat and
back as the ball. He loved to play and her better and fun together.
```

```text
Once upon a time, there was a little boy named Tim. He saw the share it on the
ball. They played together to play with a little girl named Spot when the sky
played for the picked and her
```

## What version 1.0 can't do

As you can see, it writes sentences, not stories. The model only ever sees the last 32
characters, so anything earlier is gone, It can't use
context from earlier in the story, so it can't carry a character or a plot from
one sentence to the next.

I have done some research and I now know that a transformer can look back over the whole story and
learn which earlier characters matter for the next one. This also means that i will need to implement: layer normalization. self attention and position embeddings. A transformer also does far more math per character than this
model, which is why I'm making the framework faster as my main goal first.

## What's next

1. A faster matmul kernel, aiming for OpenBLAS's single-core speed.
2. A thread pool, so medium-sized multiplies can use more than one core.
3. float32 instead of double.
4. A transformer version of GLM.

The code is all on GitHub:

- nnet, the framework: https://github.com/EliLevasseur/neural-network
  (release v0.2.0)
- GLM, the language model: https://github.com/EliLevasseur/glm

If you've written fast kernels or trained small models from scratch, I'd like to
hear what you would do differently.
Any advice or possible contributions are welcome. You can reach me at eleva098@uottawa.ca
