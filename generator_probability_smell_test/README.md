# "Smell Tester" for Speculative Decoding

It turns out that implementing speculative decoding correctly can be difficult, especially when using a tree-based
method, instead of a single speculative generation.
This is a mini-library to ensure that the output of the overall speculative decoding process follows the correct
probability distribution; i.e. that of the main model (at least for short sequences).

New methods can be added by implementing a generator (see examples in the generators directory),
and then adding a reference in tester.py (which is also the entry point).

Essentially, this library works by creating a random discrete probability distribution over outputs
(in addition to a few hardcoded examples).
It then mocks a network that _should_ generate samples that match that distribution.
Because these are discrete distributions, we can just do a chi-square test to see if the distributions match.

Because of numerical/floating point errors, the distributions diverge surprisingly often- you'd expect 1/100 samples to
have a p-value of <0.01 in a correct implementation, but this tends to happen maybe 3/100 times (and often p<0.001).
However, an implementation error in the sampling method beyond small floating point issues will lead to much much more
significant divergences between the two distributions.

This library was later adapted to test how closely a method for preventing errors (hallucinations, etc) aligns with
the theoretical optimal of rejection sampling.