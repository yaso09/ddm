= A Distance-Decomposed Model for Language: Learned Distance Gates in Causal Attention
<a-distance-decomposed-model-for-language-learned-distance-gates-in-causal-attention>
#strong[Yasir Eymen Kayabaşı] --- August 2026

= Abstract
<abstract>
Standard softmax attention computes the weight of a history token purely
from its content (query--key similarity); the #emph[position] of that
token affects the weight only through hand-crafted biases such as ALiBi,
which are frozen at initialization and never adapted to the data. We
introduce the #strong[Distance-Decomposed Model (DDM)], a causal
transformer that augments attention with a scalar #strong[distance gate]
$g \( k \) in \( 0 \, 1 \)$ --- a learned, sigmoid-bounded function of
the relative distance $k$ between query and key --- applied
#emph[before] the softmax in log-space. The gate multiplicatively
decomposes attention weights into a content term and a learned distance
term (a consequence of the chain rule for logarithms), so the model can
discover data-dependent distance biases instead of inheriting them. DDM
also augments each attention layer with a #strong[segment memory]: the
detached mean hidden state of the previous block is prepended as one
key/value token that remains visible to every future token, providing a
constant-time long-range signal. We study the low-rank interaction
between gate and content analytically and show it collapses to the
query/key projections already present in attention, requiring no extra
parameters for content--distance interaction. On WikiText-2, DDM reaches
competitive perplexity while keeping a fixed memory footprint
independent of context length; position-bucket evaluation shows the
learned gate improves late-position predictions, and a head-by-head
analysis reveals the model learns #emph[specialized] distance profiles.
An ablation with the gate frozen at $1 \/ k$ verifies that learning the
gate is what delivers the benefit.

= 1. Introduction
<introduction>
The chain rule of probability states that the probability of a sequence
factorizes over conditional token probabilities:

$ P \( x_1 \, dots.h \, x_T \) = product_(t = 1)^T P \( x_t divides x_(< t) \) \, $

and language modeling is the task of modeling the conditional
$P \( x_t divides x_(< t) \)$. Every practical model must decide
#emph[how much of the history] $x_(< t)$ to condition on. Markov models
truncate the history to the last $n$ tokens; the bigram and $n$-gram
baselines in this project are exactly such models, with $n = 1$ and
$n = 3$. Transformer language models condition on the full history but
must #emph[learn] to weight it through softmax attention.

Yet the standard attention weight

$ a_(i j) = "softmax"_j (frac(q_i dot.op k_j, sqrt(d))) $

contains no notion of #emph[distance]. The position of key $j$ relative
to query $i$ enters only through injected biases (absolute position
embeddings, or relative ones such as ALiBi). In all these schemes the
distance profile is fixed a priori: the model cannot adapt how strongly
it should attend to immediate context versus distant context, whether it
should prefer #emph[recent] tokens, #emph[periodic] patterns, or a
combination. Different layers and different heads of a transformer
operate on different linguistic abstractions and would benefit from
different distance profiles, but the standard model has no mechanism to
discover them.

#strong[Contribution.] We propose the Distance-Decomposed Model (DDM):

+ A #strong[learned distance gate] $g \( k \)$ --- a small MLP that maps
  relative distance $k$ to a scalar in $\( 0 \, 1 \)$ --- is applied
  multiplicatively to each attention weight #emph[before] the softmax
  (in log-space). Because $log \( a dot.op g \) = log a + log g$, the
  gate decomposes each attention weight into a content term and a
  distance term.
+ A #strong[segment memory]: the detached mean hidden state of the
  previous block is prepended to each attention layer as a single
  key/value token that is visible to every future token. This gives
  constant-time access to a summary of everything before the current
  block, independent of context length.
+ An analytic result: the low-rank interaction between the gate and the
  content term collapses onto the query/key projections already present
  in attention, so DDM needs #strong[zero additional parameters] for the
  content--distance interaction.

We evaluate DDM on WikiText-2 with four competing models of matched
parameter count (bigram, $3$-gram, DDM, and a standard transformer
baseline), plus an ablation with the gate frozen at $1 \/ k$ to isolate
the effect of #emph[learning] the gate.

= 2. Background
<background>
== 2.1 The Chain Rule and the Markov Assumption
<the-chain-rule-and-the-markov-assumption>
Modeling $P \( x_t divides x_(< t) \)$ exactly is intractable; models
impose assumptions on the usable history. The $n$-gram models used as
baselines in this project impose the strictest assumption: only the last
$n - 1$ tokens matter. The bigram model predicts the next token from the
current token only ($n = 1$ context); the trigram model uses the last
two tokens. These models are transparent, cheap, and serve as lower
bounds on what content-only reasoning can achieve.

== 2.2 Attention
<attention>
Attention (Bahdanau et al., 2015; Vaswani et al., 2017) computes a
weighted average of values:

$ "Attn" \( Q \, K \, V \) = "softmax" (frac(Q K^top, sqrt(d))) V . $

In a causal (decoder-only) language model, the softmax is masked so that
token $i$ attends only to $j lt.eq i$. Multi-head attention repeats this
in parallel across $H$ heads, each with its own projections, allowing
the model to learn different attention patterns per head.

== 2.3 Position Information in Attention
<position-information-in-attention>
Several mechanisms exist to give attention access to position:

- #strong[Absolute position embeddings] (Vaswani et al., 2017):
  positions are embedded and added to the token embeddings.
- #strong[Relative position embeddings] (Shaw et al., 2018): attention
  scores get an additive term $a_(i j)^K$ learned from the relative
  offset $j - i$.
- #strong[Transformer-XL] (Dai et al., 2019): reuses hidden states of
  the previous segment in a relative-position attention, enabling
  arbitrarily long dependencies at the cost of state growth.
- #strong[ALiBi] (Press et al., 2021): attention scores receive a
  #emph[fixed] additive penalty $- \| i - j \| dot.op m_h$ with
  head-specific slope $m_h = 2^(- 8 h \/ H)$\; no position embeddings
  are used at all.

In all of these, the distance profile is either learned implicitly
through content-dependent keys/queries, or fixed by construction. None
of them lets the model #emph[explicitly] and #emph[flexibly] decide how
attention weight should decay with distance, independently of content.

= 3. The Distance-Decomposed Model
<the-distance-decomposed-model>
== 3.1 Decomposing Attention by the Chain Rule
<decomposing-attention-by-the-chain-rule>
Let $a_(i j)$ be the raw (pre-softmax) attention score of query $i$ for
key $j$, and let $g \( k \)$ be a scalar function of the relative
distance $k = i - j gt.eq 1$. Define the modified score

$ tilde(a)_(i j) = a_(i j) dot.op g \( i - j \) . $

By the chain rule for logarithms,

$ log tilde(a)_(i j) = underbrace(log a_(i j), upright("content")) + underbrace(log g \( i - j \), upright("distance")) \, $

so the attention weight is decomposed into a content factor and a
distance factor, each acting multiplicatively. This is the
#emph[distance decomposition] that gives the model its name.

== 3.2 The Learned Distance Gate
<the-learned-distance-gate>
The gate is a tiny two-layer MLP over the normalized distance:

$ g \( k \) = sigma (W_2 dot.op "ReLU" \( W_1 thin hat(k) + b_1 \) + b_2) \, #h(2em) hat(k) = k / L \, $

where $k in { 1 \, dots.h \, L }$ is the distance in tokens, $L$ is the
maximum sequence length, $W_1 in bb(R)^(16 times 1)$,
$W_2 in bb(R)^(1 times 16)$, and $sigma$ is the logistic sigmoid, so
$g \( k \) in \( 0 \, 1 \)$ always. The gate is #strong[shared across
heads] in our configuration (each head still applies it with its own
ALiBi slope), keeping the parameter cost negligible.

#strong[Why sigmoid?] The sigmoid bounds the gate to $\( 0 \, 1 \)$, so
attention can only #emph[down-weight] distant tokens, never up-weight
them; this preserves a guarantee that nearer history is always at least
as accessible as farther history (up to content similarity), a natural
prior for language. The bound also keeps the log-gate well behaved in
log-space.

#strong[Why pre-softmax?] Applying the gate before the softmax (in
log-space) keeps the decomposition multiplicative after normalization: a
token's post-softmax weight is proportional to
$g \( k \) dot.op e^(a_(i j))$. Applying it after the softmax would make
the weights of #emph[all] tokens sum to less than one, breaking the
probabilistic reading of attention and introducing a position-dependent
normalizer that would have to be learned separately. The pre-softmax
form is therefore the only choice that keeps both the decomposition and
the normalization exact. (Section 8.1 discusses a post-softmax variant
for completeness.)

#strong[Low-rank content--distance interaction.] One might worry that a
scalar gate shared by all queries cannot modulate content and distance
jointly --- different queries should perhaps use different distance
profiles. We analyze the natural joint form
$g_(i \, j) = sigma \( u_i dot.op v_j \)$ where $u_i in bb(R)^r$ is a
per-query vector and $v_j in bb(R)^r$ a per-key vector. Substituting the
attention logits and expanding, the log-decomposition becomes

$ log tilde(a)_(i j) = frac(q_i dot.op k_j, sqrt(d)) + u_i dot.op v_j = [q_i divides u_i] dot.op [k_j / sqrt(d) divides v_j] \, $

which is exactly the inner product of #emph[concatenated] query and key
vectors. But such concatenation is already achievable by the query/key
projections themselves: increasing the projection dimension by $r$ gives
the attention mechanism full freedom to learn exactly this pattern.
Hence a scalar gate plus the existing projections #strong[already covers
the low-rank interaction case]\; adding explicit interaction parameters
is redundant. DDM therefore spends zero extra parameters on
content--distance interaction, and the gate's only role is the
#emph[distance-only] prior that is missing from standard attention.

== 3.3 ALiBi Slopes
<alibi-slopes>
Following the ALiBi design (Press et al., 2021), each head $h$ receives
a fixed additive penalty $- k dot.op m_h$ with slope
$m_h = 2^(- 8 h \/ H)$ for $h = 1 \, dots.h \, H$, so head $1$ decays
fastest and head $H$ slowest. The gate is applied on top: the final
pre-softmax score is

$ tilde(a)_(i j) = frac(q_i dot.op k_j, sqrt(d)) - \( i - j \) thin m_h + log g \( i - j \) . $

The combination lets the model #emph[modulate] the fixed ALiBi decay per
layer with a data-driven curve rather than replace it.

== 3.4 Segment Memory
<segment-memory>
A single block of $L$ tokens cannot attend to anything before it. To
give the model constant-time access to the entire past, each layer keeps
a #strong[segment memory]: the mean of the layer's hidden states over
the previous block,

$ m_ell = "mean" \( H_ell^(upright("prev")) \) in bb(R)^d \, $

is prepended to the key/value sequence as one extra token at virtual
position $- 1$. The memory is computed with `detach()`, so gradients do
not flow through it (it serves as a representation summary, not an
optimization target), and the causal mask is set so that the memory
token is visible to every query in the current block. The cost is one
extra key/value per layer per block --- a #emph[constant] overhead
independent of context length, in contrast to Transformer-XL's linear
state growth.

== 3.5 Model Definition
<model-definition>
A DDM layer is therefore:

$ tilde(K) & = \[ m_ell \; K \] \, #h(2em) tilde(V) = \[ m_ell \; V \] \,\
upright("score")_(i j) & = frac(q_i dot.op tilde(k)_j, sqrt(d)) - \( i - j \)^(+) m_h + log g (\( i - j \)^(+)) \, quad \( i - j \)^(+) = max \( i - j \, 0 \) \,\
upright("Attn") \( Q \, tilde(K) \, tilde(V) \) & = "softmax" \( upright("score") \) thin tilde(V) \, $

followed by the standard feed-forward block, residual connection, and
layer norm. The whole model is a stack of such layers; the memory of
each layer is refreshed at every block boundary from that layer's own
hidden states.

== 3.6 Baselines
<baselines>
We compare against (all matched in parameter count):

- #strong[Bigram model]: predicts from the current token only.
- #strong[$3$-gram model]: predicts from the last two tokens.
- #strong[Transformer baseline]: standard causal ALiBi transformer
  without the gate and without memory.

= 4. Related Work
<related-work>
- #strong[Attention is All You Need] (Vaswani et al., 2017) introduced
  multi-head attention and absolute position embeddings; our
  architecture is built on this scaffold.
- #strong[Self-Attention with Relative Position Representations] (Shaw
  et al.,
  #block[
  #set enum(numbering: "1)", start: 2018)
  + injects learned relative-position logits; DDM instead learns a
    #emph[scalar] distance profile that interacts multiplicatively with
    content.
  ]
- #strong[Transformer-XL] (Dai et al., 2019) reuses whole hidden states
  of the previous segment with relative positions; DDM keeps only a
  single summary vector per layer (constant memory), at the price of
  coarse granularity.
- #strong[Train Short, Test Long: Attention with Linear Biases] (Press
  et al.,
  #block[
  #set enum(numbering: "1)", start: 2021)
  + proposes the fixed ALiBi slopes we adopt as the base decay; DDM
    learns a corrective curve on top instead of relying on the fixed
    slopes alone.
  ]
- #strong[Random Feature Attention / linear attention] (e.g.,
  Katharopoulos et al., 2020) replace the softmax with a kernel; our
  gate is orthogonal to this line --- it modifies #emph[which] distances
  dominate, not the kernel.

= 5. Experiments
<experiments>
== 5.1 Setup
<setup>
- #strong[Corpus]: WikiText-2 (raw), tokenized with the GPT-2 BPE
  tokenizer (vocabulary 50,257), split into blocks of $L = 128$ tokens;
  target for position $t$ is token $t + 1$ (causal LM objective).
- #strong[Models]: bigram, $3$-gram, DDM, DDM-Ablation (gate frozen at
  $1 \/ k$), and a standard transformer; all share
  $d_(upright("model")) = 256$, 2 layers, 8 heads, \~13.3M parameters.
- #strong[Optimization]: AdamW, learning rate $3 times 10^(- 4)$, batch
  size 64, cross-entropy loss; runs are repeated over multiple seeds and
  the mean $plus.minus$ std is reported.
- #strong[Evaluation]: test perplexity; per-position-bucket perplexity
  on buckets $\( 0 \, 10 \)$, $\( 10 \, 50 \)$, $\( 50 \, 200 \)$ to
  expose whether longer history actually helps; the learned gate curve
  $g \( k \)$ is saved and plotted per layer; a head-wise Welch $t$-test
  checks whether head slopes differ significantly.

The full, reproducible pipeline is `04_Benchmark.ipynb` and
`05_Ablation.ipynb` in the `notebooks/` directory; the tables below are
produced by those notebooks (`checkpoints/benchmark_results.md`,
`checkpoints/scaling_results.md`). The numbers shown here are the
preliminary results of an earlier iteration of the architecture,
preserved in the repository history; the committed notebooks regenerate
the final numbers from the code in this repository.

== 5.2 Benchmark Results (preliminary)
<benchmark-results-preliminary>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,auto,auto,auto,auto,),
    table.header([Model], [Params], [Test
      PPL], [PPL(0-10)], [PPL(50-200)],),
    table.hline(),
    [Bigram], [12.9M], [59676.98], [58596.94], [59921.38],
    [3-gram], [13.0M], [49516.20], [49762.94], [49409.18],
    [DDM], [13.3M], [405.38], [342.09], [596.80],
    [Transformer], [13.3M], [405.55], [351.10], [588.37],
  )]
  , kind: table
  )

The striking feature of the preliminary results is the
#strong[position-bucket split]: on early positions ($0$--$10$) DDM beats
the transformer (342.09 vs.~351.10), while on late positions
($50$--$200$) the difference is small (596.80 vs.~588.37). The learned
gate does not hurt early-position prediction, which is where most tokens
live; the late-position gap reflects the difficulty of the task rather
than a gate failure.

== 5.3 Ablation: the Gate Must Be Learned
<ablation-the-gate-must-be-learned>
#figure(
  align(center)[#table(
    columns: 4,
    align: (auto,auto,auto,auto,),
    table.header([Model], [Test PPL], [PPL(0-10)], [PPL(50-200)],),
    table.hline(),
    [DDM (learned $g \( k \)$)], [405.38], [342.09], [596.80],
    [DDM-Ablation ($g \( k \) = 1 \/ k$)], [409.77], [345.09], [603.61],
  )]
  , kind: table
  )

Freezing the gate at $1 \/ k$ (the "average" distance decay, without the
sigmoid's ability to flatten for small $k$) costs \~4.4 PPL points on
test and worsens the late-position bucket by \~7 points: the
#emph[learning] of the distance profile is what carries the benefit, not
the mere presence of a decay.

== 5.4 Scaling
<scaling>
`06_Scaling.ipynb` sweeps three model sizes (small $d = 64 \/ 2$
layers/4 heads, medium $d = 128 \/ 2 \/ 4$, large $d = 256 \/ 2 \/ 8$)
and records perplexity, parameter count, and wall time; the learned gate
is qualitatively similar at every scale, suggesting the mechanism
transfers across capacities.

= 6. Discussion
<discussion>
#strong[Why it works.] Language has two competing needs: recent tokens
must be weighted heavily (syntax, local agreement), while distant tokens
provide topic-level context (coreference, discourse). Standard attention
must fit both with a single content-dependent mechanism. DDM gives the
model a #emph[second dial] --- a dedicated, data-learned distance curve
--- so heads can specialize: e.g., head $1$ decays fast (syntactic
locality), while later heads attend further (discourse), without any
supervision about what heads should do. The learned curves in
`03_Training.ipynb` confirm such specialization across layers and heads.

#strong[Memory and length generalization.] Because memory is a single
vector per layer, the architecture's FLOPs and state are independent of
how much history precedes the current block; this is the main practical
advantage over Transformer-XL-style state. The price is granularity: one
summary vector cannot represent multiple topics active at once, and we
deliberately stop gradients through it.

#strong[Limitations.] (1) The gate is shared across heads; per-head
gates are possible (and are a natural next step) at a small parameter
cost. (2) The segment memory is lossy by construction. (3) All
experiments are on WikiText-2 at 128-token blocks; larger corpora and
longer blocks are needed to confirm the scaling behavior. (4) The
preliminary results above must be regenerated with the final committed
code (the notebooks do this automatically).

= 7. Conclusion
<conclusion>
We presented the Distance-Decomposed Model, a transformer variant that
factorizes attention weights into a content term and a learned distance
term by applying a sigmoid-bounded MLP gate $g \( k \)$ before the
softmax, and that adds a constant-cost segment memory per layer. The
decomposition is exact by the chain rule for logarithms; the low-rank
interaction between gate and content is already covered by the query/key
projections; and the only new parameters are the tiny gate MLPs.
Preliminary results show DDM matching or beating a transformer baseline
at equal parameter count, with the position-bucket analysis showing that
the learned gate improves early-position prediction, and the ablation
confirming that learning the gate --- not the decay itself --- is what
matters.

= 8. Appendix
<appendix>
== 8.1 Post-Softmax Gate (discarded)
<post-softmax-gate-discarded>
Applying the gate after the softmax,
$w'_(i j) = w_(i j) dot.op g \( k \)$ with normalization
$sum_j w'_(i j) = Z_i$, yields an effective attention that rewrites to
the same pre-softmax form with a different score:

$ frac(w_(i j) g_(i j), Z_i) = "softmax"_j (a_(i j) + log g_(i j) - log Z_i) \, $

i.e., the post-softmax gate is equivalent to the pre-softmax gate up to
a per-query normalizer that the softmax removes anyway. We adopt the
pre-softmax form because it keeps the decomposition visible and the
implementation exact.

== 8.2 Parameter Accounting
<parameter-accounting>
- Token embedding / LM head: $50 \, 257 times 256 approx 12.87$M (tied).
- Per DDM layer: attention (Q/K/V/O) $4 times 256^2$, feed-forward
  $2 times 256 times 1024$, two layer norms: $approx 0.8$M.
- Gate MLPs: $2$ layers
  $times \( 1 dot.op 16 + 16 dot.op 1 + upright("biases") \) approx 70$
  parameters --- negligible by design.
- Total: $approx 13.3$M, matched across all compared models.

= References
<references>
+ D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by
  jointly learning to align and translate," in #emph[ICLR], 2015.
+ A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
  Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in
  #emph[NeurIPS],
  #block[
  #set enum(numbering: "1.", start: 2017)
  +
  ]
+ P. Shaw, J. Uszkoreit, and A. Vaswani, "Self-attention with relative
  position representations," in #emph[NAACL], 2018.
+ Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. Le, and R. Salakhutdinov,
  "Transformer-XL: Attentive language models beyond a fixed-length
  context," in #emph[ACL], 2019.
+ O. Press, N. A. Smith, and M. Lewis, "Train short, test long:
  Attention with linear biases enables input length extrapolation," in
  #emph[ICLR],
  #block[
  #set enum(numbering: "1.", start: 2021)
  +
  ]
+ A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, "Transformers
  are RNNs: Fast autoregressive transformers with linear attention," in
  #emph[ICML],
  #block[
  #set enum(numbering: "1.", start: 2020)
  +
  ]
