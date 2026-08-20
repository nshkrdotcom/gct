# What the categorical scaffold is doing

The reviewer question: does the site/topology apparatus predict anything the four defect
measures (transport, cycle, base-change, descent) wouldn't already predict on their own,
if you'd derived them from "test compositionality and order-independence of a learned
vector transport map" without the topos vocabulary? If yes, name the prediction. If no,
say so.

## Answer: one genuine prediction, one structural constraint, and otherwise organizing vocabulary

### 1. The commuting-square defect is a categorical prediction

A flat "learn a transport operator, test on held-out data" framing generates the transport
defect and, with a little imagination, the cycle defect (apply the inverse and check you
return). It does not, on its own, generate the prediction that *two independent substantive
operators composed in either order should arrive at the same point*.

That prediction comes directly from the Beck-Chevalley / commuting-square structure: if
pressure and concentration are independent base coordinates, and the model's residual
geometry respects their independence, then the square

```text
    (S,P,M) ──pressure──▶ (S,P',M)
       │                       │
  concentration           concentration
       ▼                       ▼
    (S,P,M') ──pressure──▶ (S,P',M')
```

should commute — not at the oracle level (ToyThermo's interaction term `q·M·ln(P)` means
the oracle values at the corners differ depending on path) but at the *operator* level:
applying `T_pressure` then `T_concentration` should land on approximately the same
activation vector as `T_concentration` then `T_pressure`, even though the oracle targets
at the intermediate corners are different.

This is implemented in [`_evaluate_squares`](file:///home/home/p/g/n/gct/src/gct/metrics/evaluate.py#L148-L191):
both routes are computed and compared against each other and against the observed target.
The commuting-square defect is a direct test of base-change commutativity in the
categorical sense, restricted to the two generators that have independent base coordinates.

A "learn a good map and test it" framing would test route-to-target accuracy. The
categorical framing additionally demands route-to-route agreement — a prediction about the
*internal consistency of the operator algebra*, not just its predictive accuracy. That is
the non-obvious test. If transport operators fit well but the square doesn't commute, the
categorical picture is wrong even though the linear algebra is fine.

### 2. The restriction/forward asymmetry is a structural constraint

The categorical vocabulary forces `g^*` (restriction: given a context change, what happens
to a representation pulled back from the new context?) apart from `g_!` (forward transport:
given a source representation, predict where it goes under a context change). These are
genuinely different maps. The code implements forward transport only
([`LowRankResidualTransport.predict`](file:///home/home/p/g/n/gct/src/gct/operators/low_rank.py#L44-L50),
[`ContinuousGeneratorTransport.predict_delta`](file:///home/home/p/g/n/gct/src/gct/operators/generator.py#L44-L60)),
and the cycle defect tests whether `g^* ∘ g_!` (forward then inverse) returns to the start
([`_evaluate_cycles`](file:///home/home/p/g/n/gct/src/gct/metrics/evaluate.py#L114-L145)).

Without the categorical discipline, it is natural to assume forward transport is
invertible and treat its inverse as the restriction map. The fibration framing says: *don't
assume that*. The cycle defect is exactly the test of whether the assumption holds. This
didn't generate a new *measurement* — you'd check round-trip error anyway if you were
careful — but the categorical framing is the reason it's a *primary preregistered
endpoint* rather than an afterthought diagnostic. The distinction between "this is an
interesting thing to check" and "this is a structural prediction of the theory that must be
tested" matters for what you preregister.

### 3. Everything else is organizing vocabulary

The matching/descent proxy, the transport defect itself, the identity normalization, the
generator composition test (`T_{a+b} ≈ T_b T_a`) — these are all derivable from "fit a
linear map, test compositionality on held-out data" without categorical language. The
descent proxy in particular (comparing representations that should be identified under a
nuisance rewrite) sounds like it needs a sheaf gluing condition, but operationally it's
just a distance between two vectors that the oracle says should represent the same
underlying state. You'd write the same test without ever hearing the word "descent."

THEORY.md §1 already says this: no literal sheaves, no cohomology, categorical language is
scaffold not evidence. This document makes explicit *which* pieces are scaffold and which
are load-bearing.

## Summary

| Defect family | Would exist without categorical framing? | Categorical contribution |
|---|---|---|
| Transport | Yes | None beyond vocabulary |
| Cycle | Yes, but likely diagnostic not primary | Elevated to primary by the restriction ≠ forward distinction |
| Commuting square | **No** | The prediction that independent-coordinate operators commute is a Beck-Chevalley test |
| Matching / descent | Yes | Vocabulary only |
| Generator composition | Yes | Vocabulary only |

One genuine prediction (commuting square). One structural constraint that shaped what got
preregistered (cycle as primary). The rest is organizing vocabulary — useful for not
making silent invertibility assumptions, but not generating additional empirical content
beyond what "test compositionality of learned vector maps" already provides.

PRIOR_ART_DIFF.md should state this as plainly as THEORY.md does.
