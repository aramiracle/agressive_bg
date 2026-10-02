# Guide to Aggressive Backgammon

This project trains a neural network to play backgammon. The network does not play by memorizing openings or by searching the whole game. It learns two things from games it plays against itself: how a position is likely to end, and when to turn the doubling cube. A short search then uses those predictions to pick a move.

The stage comment in `src/config.py` is the split the code implements. Stage 1 is 1-point games, no cube, and the training point values `R_WIN = 1`, `R_GAMMON = 3`, `R_BACKGAMMON = 5`. The comment calls those aggressive scores and says this stage learns checker play. Stage 2 is 7-point matches with the cube and the real multipliers 1, 2, and 3. `TRAIN_MODE` is what selects 1/3/5 versus 1/2/3. It defaults to on only in stage 1.

This document explains the game the program implements, the theory behind each piece, and the way the pieces fit together. No prior knowledge of the code is assumed. A little familiarity with backgammon and with the idea of a neural network is enough.

## 1. What has to be learned

A game of backgammon is a race of fifteen checkers, interrupted by hitting and blocking. Two facts make it a different learning problem from chess or Go.

The dice are random. After a player moves, the opponent rolls. There are 21 distinct rolls, and 36 equally likely outcomes, because a non-double can come up in either order. The default search does not average those rolls. It scores the position the current roll leaves behind. Averaging the opponent's best reply to every roll is a separate setting, `BG_SEARCH_PLY=2`, described in section 6. Training leaves the ply at 1.

A turn is several checker steps that together produce one new position. The rules require the player to use as many dice as possible. Different orders of the same steps often land on the same position. The decision that matters is which position to leave, not which intermediate step looked locally attractive.

The program is built around those two facts. The network looks at one position and predicts how the game will end. The search lists every legal way to play the dice that have already been rolled and asks the network what each resulting position is worth.

## 2. The board the engine keeps

White is player `+1`. Black is player `-1`. White's checkers are stored as positive counts, black's as negative counts.

There are 24 points, indexed `0` through `23`. Index `0` is white's 1-point (the ace point of white's home board). Index `23` is white's 24-point. White moves toward smaller indexes and bears off past `0`. Black moves toward larger indexes and bears off past `23`. Black's home board is indexes `18`–`23`.

The opening position is the standard one:

| Index | Point, from white's side | Checkers |
|------:|--------------------------|----------|
| 23 | white's 24-point | 2 white |
| 18 | white's 19-point | 5 black |
| 16 | white's 17-point | 3 black |
| 12 | white's 13-point | 5 white |
| 11 | white's 12-point | 5 black |
| 7 | white's 8-point | 3 white |
| 5 | white's 6-point | 5 white |
| 0 | white's 1-point | 2 black |

Each side also has a bar and a count of checkers borne off. A checker on the bar has not yet entered. A player who has borne off all 15 checkers has won the game.

The pip count is the total distance a side's checkers still have to travel. A white checker on index `i` has `i + 1` pips left. A black checker on index `i` has `24 - i` pips left. A checker on the bar counts as 25. The race formula in section 7 uses this count. If a game is cut off at `MAX_GAME_MOVES` (1000 turn-loop iterations), the side with fewer pips is given the game, including when the position still has contact.

### Contact and the race

`contact` reports whether the two sides still overlap. White's rearmost index is 24 if white has a checker on the bar, otherwise the highest index that holds a white checker (or −1 if white has none on the board). Black's rearmost index is −1 if black has a checker on the bar, otherwise the lowest index that holds a black checker (or 24 if black has none on the board). The function returns true when white's rearmost index is strictly greater than black's. When it returns false, the armies have passed and neither side can hit the other.

A race in which both sides have already borne off at least one checker can only end as a single game. A gammon is impossible, because the loser has already taken a checker off. That special case has an analytic answer, described in section 7, and both the search and self-play use it.

## 3. The rules the engine enforces

`BackgammonGame` in `src/engine.py` is a strict rules engine. Search and training never invent moves. They ask the engine what is legal, apply what it returns, and read the result back.

### Dice

An ordinary roll is two dice. If they differ, the player must try to play both numbers. If they match, the player plays that number four times.

The opening roll is separate. Each side rolls one die. The higher die moves first and plays both numbers. Ties are rolled again. After the opening play, the turn passes and the new side rolls both dice in the usual way.

A double may be offered only before the dice are rolled. The opening roll is played as it fell. The side that has already rolled cannot double.

### One step

If the player has any checker on the bar, the only legal step is to enter. White enters on index `24 - die` (black's home board). Black enters on index `die - 1`. The destination must be open: empty, occupied by the mover, or occupied by exactly one opposing checker. Two or more opposing checkers close the point, and a closed entry point means that die cannot be used to come in.

From the board, a die moves one of the player's checkers that many points in that player's direction. Landing on a single opposing checker hits it: that checker is sent to the bar and the mover occupies the point. Landing on two or more opposing checkers is illegal. Stacking on your own checkers is legal.

Bearing off is legal only when every checker of that side is in its home board (and none are on the bar). A die that lands exactly on the edge bears a checker off. A die larger than the distance to the edge bears a checker off only when no checker of that side is further from home than the one being moved. Any checker still farther from that side's edge blocks the overshoot. The engine does not ask whether that checker has a legal step with the same die.

### A whole turn

The player must use as many dice as possible. If two different plays use the same number of dice, and that number is less than the number of dice rolled, the play that spends more pips is required. This is the usual maximality rule: if only one die can be played, and both dice could be played alone, the larger die must be played.

The engine finds legal turns in two ways.

`get_legal_moves` returns the first legal step of every maximal play, together with the die that step consumes. The die is part of the action. Bearing off can be legal for more than one unused die, and which die is consumed changes what remains. The human interface uses these steps one at a time.

`get_legal_turns` returns every complete legal play and the position it produces. Plays that reach the same position are kept once. Doubles would otherwise explode into every ordering of the same steps; collapsing by position keeps that search small. The AI decides among these complete plays. A turn is usually a handful of distinct positions, sometimes a few dozen, which is small enough to score all of them.

### How a game is won

Bearing off the fifteenth checker wins. The win has a type:

- **Single.** The loser has borne off at least one checker. Worth 1 in real play.
- **Gammon.** The loser has borne off nothing, and has no checker on the bar or in the winner's home board. Worth 2 in real play.
- **Backgammon.** The loser has borne off nothing, and still has a checker on the bar or in the winner's home board. Worth 3 in real play.

The points written onto the match score are `cube × type`, except while training mode is on (section 9). A refused double is always a single at the current cube value: the player who offered it wins `cube` points, and the cube does not double.

If the turn loop reaches `MAX_GAME_MOVES` (1000 iterations) and neither side has borne off all fifteen checkers — possible only while a fresh network plays almost at random — the side with fewer pips is awarded a single. An exact pip tie goes to the side about to roll. Each iteration is one side's turn, cube decision included.

### The cube and the match

The cube starts at 1, owned by the center. A player may double when all of the following hold:

- The cube is enabled for this stage of training.
- This game is not a Crawford game.
- The dice have not yet been rolled.
- The player owns the cube, or it is still centered.
- Neither side has already won.
- The cube is still smaller than the number of points the player who is further behind still needs.

The last condition is the dead-cube rule used here. If the cube is already large enough that a single game wins the match for both players, another double cannot change the match result of a single game, so it is refused by the engine.

Offering a double hands the cube to the opponent at twice the value, but only if they take. A take doubles the stake and makes the taker the owner, so only the taker may double later. A drop ends the game at the old stake.

Matches are first to `MATCH_TARGET` points. With the stage defaults, stage 1 plays to 1 and stage 2 plays to 7. Each new game starts the cube at 1, centered. Match score is what carries over. The cube does not.

Stage 1 also sets `CUBE_ENABLED` to false, and `can_double` returns false immediately in that case. A 1-point match blocks doubles a second way even if the flag is turned on: `limit = match_target - min(scores)`, and a double is illegal once `cube >= limit`. At 0–0 to 1, that limit is 1 and the cube already is 1.

### Crawford

`can_double` returns false while `crawford_active` is true.

The engine sets that flag in `_update_crawford_status`. If the match is longer than 1, Crawford has not been used yet, and either score equals `match_target - 1`, the next game is Crawford. When that game ends by a borne-off win, a refused double, or the 1000-iteration cutoff, `crawford_used` becomes true and later games may double again. The engine does not set the flag when `match_target` is 1. The comment in the engine says Crawford has no meaning there, because every game would be match point.

Self-play does not leave the flag to that function alone. `play_self_play_match` and `play_vs_baseline_match` choose the Crawford game themselves and then assign `game.crawford_active`. The leader is white when white's score is strictly higher, and black otherwise, including when the scores are equal. The game is Crawford when Crawford has not yet occurred in the match, the match length is greater than 1, and `MATCH_TARGET - leader's score == 1`. The flag `crawford_occurred` is set before that game is played, so the following game is not Crawford.

## 4. What the network sees

The same physical position has to look the same to the network no matter which side is about to move. Encoding is always from the side to move, called the canonical view.

If white is to move, points stay in order and white's checkers stay positive. If black is to move, the 24 points are reversed and every sign is flipped, so black's checkers become the positive ones and black also appears to move toward smaller indexes. The bar and the off counts are swapped the same way. After this, "me" is always positive and always racing toward index 0. The network never has to learn the game twice.

Each point is one integer token. A point holds between −15 and +15 checkers, and the token is `count + 15`, so the vocabulary is the 31 values `0` … `30`. Token `0` means "fifteen opposing checkers". It is a real position, not an empty padding slot, so every token has a learned embedding.

The sequence is 28 tokens: the 24 points, my bar, the opponent's bar, my borne-off count, and the opponent's borne-off count. The opponent's bar and the opponent's borne-off count are negated before the offset is added, so my checkers and the opponent's stay opposite in sign. A learned positional embedding tells the transformer which point is which. Point 1 and point 24 are not interchangeable, and the bar is not a point.

Five numbers ride along as match context:

1. Who owns the cube, from my point of view: `+1` mine, `−1` theirs, `0` centered. The special value `2` means a double has been offered to me and I am answering it. I will not roll.
2. The cube value divided by 64, so it sits in a small range.
3. My match score divided by the match length.
4. The opponent's match score divided by the match length.
5. `1` if this is the Crawford game, otherwise `0`.

The `2` on the first feature is the whole difference between "I am on roll and might double" and "I am looking at a double and must take or drop". The same cube head handles both decisions, and this feature is what tells them apart.

## 5. What the network predicts

The default model is a transformer, `BackgammonTransformer` in `src/model.py`. A convolutional network with the same inputs and the same two heads can be selected with `MODEL_TYPE`. Training uses the transformer.

The context vector is projected to one token and placed in front of the 28 point tokens. A stack of pre-norm transformer layers (10 layers, width 128, 16 heads, feed-forward width 256, GELU) lets that leading token attend to every point. The leading token is the only thing read out. It is a summary of the position that already contains the score, the cube, and Crawford.

Two small heads sit on that summary.

### The outcome head

The outcome head produces six logits, one for each way the game can end, from the side the position was encoded for:

| Index | Meaning |
|------:|---------|
| 0 | I win a single |
| 1 | I win a gammon |
| 2 | I win a backgammon |
| 3 | I lose a single |
| 4 | I lose a gammon |
| 5 | I lose a backgammon |

A softmax turns the logits into a distribution. Everything else in the program — ranking moves, pricing a double, the training target — is computed from this distribution. The network is never asked to emit a single win-probability that already mixes the type of game with the match score.

That split is the central modeling choice. How the checkers will finish is a property of the position and the dice. How much a gammon matters is a property of the score and the cube. A scalar output would have to relearn the same contact position separately at 0–0, at 4–6, and at every cube value. A six-way distribution lets one checker evaluation be reused: change the score or the cube, and a table lookup converts the same six probabilities into a new match equity. The context features are still present, because cube ownership and Crawford do change how both sides will play, and because the cube head needs them. They are not forced to carry the entire equity calculation.

There is no policy head. The network does not predict which checker to move. Move choice is the search's job, using the outcome distribution as its evaluation.

### The cube head

The cube head produces two logits. After a softmax, index 0 is "no double" or "drop", and index 1 is "double" or "take". Which reading applies depends on the context feature described above.

`get_learned_cube_decision` returns the argmax of the cube logits when `stochastic` is false. That is what evaluation's `get_cube_action` does as well, and what the web server passes. Self-play passes `stochastic=True`, so the action is sampled from the softmax, except when a draw against `cube_epsilon` replaces it with `random.randint(0, 1)`. The training target for this head is not the action that was sampled. It is a soft label computed from match equity, derived in section 8. The network is pulled toward the priced decision even on turns where exploration did something else.

### The legacy adapter and the shipped baseline

An older architecture predates the six-way head, and `LegacyValueTransformer` in `src/model.py` can still reload it. `src/utils/checkpoint.py` rebuilds from that architecture only when the saved `config.py` sets `HEAD_KIND` to `value_policy`. Those nets predict a scalar equity in `[−1, 1]`, a from-point and a to-point policy (26 action slots: the 24 points, the bar, and the off), and a cube decision. The policy heads are part of the checkpoint, so they are loaded, and play never reads them. The scalar becomes six logits: a single win, a single loss, and the four gammon and backgammon logits set to −20.

The five-number context is rewritten into the four numbers that network expects: `[+1, raw cube, my score / match length, opponent's score / match length]`. The turn entry is fixed at `+1` because the position is already encoded for the mover. The normalized cube is multiplied by 64. The two score features are copied. Cube ownership and Crawford are dropped. Search can then treat an old network as an opponent without a second code path.

The checkpoint shipped in `checkpoints/baseline/` is no longer one of those old nets. Its `config.py` sets `HEAD_KIND` to `outcome` and repeats the current architecture — 10 layers, width 128, feed-forward width 256, five-number context — so the loader rebuilds an ordinary `BackgammonTransformer` from that file. It is a frozen stage-1 champion, and those frozen weights are the baseline of section 11. The adapter stays because a `value_policy` artifact from an older run still loads.

## 6. How a move is chosen

`MCTS` in `src/mcts.py` (also exported as `Searcher`) scores the current roll. It is a search over complete turns, not a deep tree of single checker steps, and not a Monte Carlo rollout to the end of the game.

Building a tree out of single steps would be the wrong shape for this game. Every step inside a turn belongs to the same player, so the values along that tree must not be flipped at each step the way they are in chess. Many different step orders are the same decision. And the real branch in backgammon is the dice, which a deterministic tree does not sample correctly unless it is expanded into all 21 rolls on purpose. The search here does that expansion explicitly, one ply at a time.

### One ply

1. Ask the engine for every legal complete play and the position it leaves.
2. Score each of those positions as a position where the opponent is about to roll. If the game is already over, the score is a one-hot outcome. If it is a gammon-free race, the score is the race formula of section 7. Otherwise the network evaluates it.
3. Swap the two halves of the six probabilities. A win for the opponent is a loss for the mover, and the other way around. The result is the mover's outcome distribution for that play.
4. Turn the distribution into a single equity in `[−1, 1]` (section 7). That number ranks the play.

The network always speaks for the player about to roll. A position I have just left is a position they are about to roll from, so it is encoded for them and then flipped back. Encoding it for me would silently assume that I roll again.

### Pruning

Most legal plays are obviously worse than the best one. The search keeps at most `SEARCH_PRUNE_TOP_K` plays (default 3) whose equity is within `SEARCH_PRUNE_MARGIN` (default 0.10) of the best. The discarded plays stay in the list with zero visits, so a later exploratory sample can still see them, but the bandit below runs only on the survivors. Looking ahead two plies is expensive, and it is spent on plays that might actually be chosen.

### Two plies

With `SEARCH_PLY=1`, which is the training default, the one-ply equity is the final judgment. `BG_SEARCH_PLY=2` re-scores the survivors, and only when more than one play survived pruning:

- For each of the 21 rolls, weighted by its true probability (`1/36` for a double, `2/36` otherwise), list the opponent's legal complete replies.
- Score each reply from the original mover's point of view, now on roll again.
- The opponent is assumed to take the reply that minimizes the mover's equity.
- The play's new distribution is the probability-weighted mixture of those best replies.

This is expectiminimax, not a random rollout. Every roll is included at its real frequency, and the opponent replies with the move the evaluation says is best for them. The cube is held fixed across those replies: two-ply search is a checker-play search. It does not imagine the opponent doubling before they roll.

A survivor on which the mover has already borne off all fifteen checkers is left at its one-ply value. The code skips it before the roll loop. A roll for which the opponent has no legal turn still contributes: the same position is evaluated again with the original mover to roll, at that roll's probability. That is a different call from the one-ply value, which was encoded for the opponent.

### The bandit on the survivors

The surviving plays are then visited with the PUCT rule in `_puct`. Each survivor starts with one visit and with `value_sum` equal to its equity. Further visits run until the survivors together have `NUM_SIMULATIONS` visits (default 32). If the survivors already account for that many visits, the loop does not run.

At each step the parent visit count `N` is the sum of the survivors' visits, and the score of a survivor with `n` visits is

```
score = Q + c · P · sqrt(N + 1e-5) / (1 + n)
```

`Q` is `value_sum / n`. `P` is that play's prior. `c` is `C_PUCT` (1.5). The `1e-5` is `MIN_PRIOR`. Ties in the score keep the survivor that appears earlier in the pruned list.

The prior is not a learned policy. The comment in `_puct` calls it the network's equity judgment. The logits are `(equity - max equity) / EXPLORE_TEMPERATURE`, with that temperature floored at `1e-6`, and the prior is their softmax. `EXPLORE_TEMPERATURE` is 0.05. The same floor is used when self-play samples a play from equity. When `stochastic` is true and `DIRICHLET_EPS` is positive, a Dirichlet sample with concentration `DIRICHLET_ALPHA` (0.3) on every survivor is mixed in:

```
prior = (1 - 0.25) · softmax + 0.25 · noise
```

`0.25` is `DIRICHLET_EPS`. The mixed prior is clamped to at least `MIN_PRIOR` and divided by its sum. Self-play calls `search` with `stochastic=True`. Evaluation calls it with `stochastic=False`, so the Dirichlet term is absent. `scripts/play_web.py` calls `search` without that argument, and the default is false.

Each backup adds the same equity again. `Q` therefore never changes with more visits: the leaf was already evaluated exactly once, and there is no deeper expansion inside the bandit. Extra simulations only redistribute visits toward high-prior, under-visited plays. Greedy selection takes the most visits, breaking ties by equity. Because `Q` is constant, that ranking agrees with equity except where Dirichlet noise has tilted the prior. The noise is how self-play tries a near-best play often enough for the network to learn that it was worse.

### Which play is actually made

In evaluation, and in human play, the play with the most visits is made.

In self-play the first `EXPLORE_TURNS` turns of a game (default 8) are sampled from a softmax over equity, temperature 0.05, across every legal play including those pruned from the bandit. Both sides' turns count toward those eight. The sample ignores visit counts, so the Dirichlet noise inside the bandit does not choose the play during this opening. A temperature that small stays close to the best equity and only leaks probability onto plays that are nearly as good. A play 0.10 worse than the best is down by a factor of about `e^-2`. After those opening turns, self-play becomes greedy on visits, so the Dirichlet noise is what still creates variety.

The stored training target for a checker decision is the six probabilities of the chosen play, not a visit distribution. Those probabilities are the bootstrap for temporal-difference learning in section 10.

## 7. From six probabilities to one number

Search needs a single number to rank plays. Two conversions exist. Which one is used depends on the match length.

### Money equity

Money equity ignores the match score and prices the game itself. Each outcome has a weight, the weights are divided by the largest one, and the result lies in `[−1, 1]`.

In ordinary scoring the weights are `+1, +2, +3` for a single, a gammon, and a backgammon won, and the negatives of those for losses. Dividing by 3, a certain single win is equity `1/3`, a certain gammon is `2/3`, and a certain backgammon is `1`.

In training mode the weights are `+1, +3, +5` and the negatives, divided by 5. A certain single win is then only `0.2`, a certain gammon is `0.6`, and a certain backgammon is `1`. The gap between "I win a single" and "I win a gammon" is wider than in real backgammon, and a bare single is worth less beside a gammon (the ratio is 3 instead of 2). Search that ranks by this number will give up a little safety to hit, to keep contact, and to trap the opponent, because a gammon moves the evaluation much more than it does in a cash game. That is the aggressive objective. It is applied in stage 1, where the match is to one point. At match length 1 the search does not use the match table at all, so this money equity is the entire ranking signal. Every win ends the match either way. The weights are how the search tells a gammon from a single. The network is trained on the six-way distribution itself, so those two endings are different targets even though both win the match.

### Match equity

At match length greater than 1, and when the searcher was given a match-equity table, ranking uses the table instead. A table built from the logistic guess counts. The file on disk is optional.

The table stores, for every score pair `(my points, opponent's points)`, an estimate of the probability that I go on to win the match. A finished match is `1` for the player who reached the target and `0` for the other. The value the search uses is that probability mapped to `[−1, 1]` by `2p − 1`, so it has the same scale as money equity.

Given a position's six probabilities, a cube, and the current score, match equity is the average of the table over the scores each outcome would produce. A single at cube `c` adds `c` points (or `c` times the training multiplier, if training mode is on). A gammon adds twice that in real play, three times in training mode, and so on, capped at the match target. The cube therefore enters as a stake, not as something the outcome head has to absorb.

`MatchEquityTable` starts from a formula and is revised from the matches the program plays. Nothing in the repository loads an external published table.

The guess, at a score where both sides still need points, is a logistic of the difference in points needed:

```
advantage = (points they need − points I need) / (points I need + points they need)
P(I win the match) = 1 / (1 + exp(−4 · advantage))
```

At 0–0 this is one half. At one point away against an opponent who needs seven, it is about 0.95. While both scores are still short of the target, the table is exactly zero-sum at initialization: my equity at `(a, b)` plus my equity at `(b, a)` is 1. A player who has reached the target is stored as 1, and the opponent at that score is stored as 0.

After each match, every score from which a game in that match was started is nudged toward `1` if that player eventually won the match and toward `0` otherwise, with learning rate `0.01`. The score after the last game is not one of those updates. Rows in which a player has already reached the target stay at their initial 1 or 0. The update is applied once from each player's point of view. Each worker applies it to its own copy before playing its next match, so later matches in that worker are priced from a table the parent has not updated yet. The parent never receives that copy. It replays the recorded observations, in worker order, onto the table it sent out. That replayed table is what gets saved as `match_equity.pt`. Later updates are small Monte Carlo corrections. They are not re-projected onto the zero-sum plane, and the log printed when the table is saved shows the largest violation so that drift is visible.

A one-point match makes the table trivial (the game and the match are the same event), which is another reason stage 1 ranks by money equity.

### The race formula

When the armies have passed and both sides have borne a checker off, the outcome distribution has only two nonzero entries: single win and single loss. The win probability of the side about to roll is a normal approximation, not a network call.

A backgammon roll moves `49/6 ≈ 8.17` pips on average, with a standard deviation near `4.3`. Being on roll is worth about half a roll, taken here as 4 pips. The lead is

```
lead = opponent's pips − my pips + 4
```

Over a race whose length is the sum of both pip counts, the uncertainty in that lead grows like the square root of the length. The z-score used in the code is

```
z = 0.665 · lead / sqrt(my pips + opponent's pips)
```

`0.665` is the constant written in `race_win_probability`. The function does not derive it. The win probability is `0.5 * (1 + erf(z / sqrt(2)))`, then clipped to `[0.001, 0.999]`. The comment above the function says bear-off wastage is ignored. Search uses this whenever `RACE_EARLY_TERMINATION` is true, which it is by default. Turning that flag off sends these leaves to the network instead.

Search uses the formula as the leaf value whenever it meets such a position. Self-play goes further, and only self-play: it stops the game, samples a winner from the formula, and scores a single. The formula's distribution is the terminal the backward pass in section 10 starts from. Earlier positions still receive the λ-return, so the race teaches the contact play that led into it. The long bearing-off sequence is not played out. The gate and any other evaluation play the race to the end. Their search leaves still use the formula.

## 8. Pricing a double

Checker play and cube play are separated on purpose. The outcome head answers "what happens if we play this out at the current stake?" The cube decision is then arithmetic on that answer, and the cube head is trained to imitate the arithmetic.

Take the six probabilities from the side whose match equity we are computing. (When the responder is being priced, those probabilities are the doubler's distribution with the two halves swapped. The responder is not on roll, so the responder's own network output, which assumes they are about to roll, is the wrong distribution.)

Let `E(stake)` be the table's win probability for playing the position out at that stake: the average from section 7, on the table's `[0, 1]` scale, before search maps it with `2p − 1`. A drop at the current cube `c` is not an average over outcomes. It is the table entry for "opponent wins `c` points as a single".

For the player on roll:

- If the opponent's equity of taking at `2c` is at least their equity of dropping, assume they take. The value of doubling is `E(2c)`.
- Otherwise assume they drop. The value of doubling is the table entry for winning `c` points immediately.
- The gain is that value minus `E(c)`. A positive gain means the double is correct.

For the responder:

- The gain is `E(2c)` minus the equity of dropping.
- A positive gain means the take is correct.

Gammons and backgammons are inside `E`, because each outcome moves a different number of points. A position that wins a lot of gammons is a stronger double than the same cubeless win rate in a pure race, and it is also a stronger drop for the trailer. The formula prices that automatically.

The gain is then turned into a soft two-class target for the cube head. Divide the gain by the match-equity gap between winning and losing a single at the current stake (how much this game is worth as a match). Floor that gap at `0.05` so a flat early table cannot explode the ratio. Clamp the ratio to `[−1.5, 1.5]`. Multiply by `CUBE_ME_TEMPERATURE` (2) and pass the product through a sigmoid. A clearly correct double or take produces a target near `0.95`, a clearly wrong one near `0.05`, and a toss-up stays near one half. The temperature is what keeps near-even cube decisions soft: a hard 0/1 label on a break-even cube would throw away the information that the decision was close, and the gradient would vanish whenever the network already agreed.

Label smoothing then mixes that target with the uniform distribution, `(1 − 0.02) · target + 0.02 · uniform`. The loss is the Jensen–Shannon divergence between the network's cube distribution and the smoothed target, which is a symmetric version of KL divergence. It is computed only on positions where a cube decision was actually made. The schedule below is read once, at the step where the iteration starts, and held fixed for the optimizer steps that follow.

An epsilon schedule mixes random cube actions into self-play so the label is not only computed on the decisions the network already likes:

| Training step | Random-action rate | Weight on the cube loss |
|--------------:|-------------------:|------------------------:|
| 0 | 0.20 | 1.5 |
| 25,000 | 0.10 | 1.3 |
| 50,000 | 0.05 | 1.2 |
| 75,000 | 0.02 | 1.1 |
| 100,000 | 0.01 | 1.0 |
| 150,000 | 0.005 | 1.0 |
| 200,000 and after | 0.002 | 1.0 |

Early on, the cube loss is up-weighted and a fifth of cube decisions are random. Later, the network is trusted to explore by sampling its own softmax, and the cube loss sits at the same scale as the outcome loss. Evaluation uses the argmax and never explores.

## 9. Two stages

`Config.STAGE` selects the curriculum. The environment variable `BG_STAGE` overrides it. `python -m src.trainer_vs_baseline --stage N` sets that variable before configuration is read (section 13 explains why the module form is required).

**Stage 1** plays matches to 1 point, with the cube disabled and training mode on. Training mode is what swaps the point values from `(1, 2, 3)` to `(1, 3, 5)`. Because the match ends on any win, those larger gammon and backgammon values do not change who wins the match, and they do not relabel the six-way target. A gammon is still the gammon bin. They change the money equity the search maximizes, and they change the replay priority in section 10, which is that same money equity. The games that get played therefore change, and so do the outcomes the network sees. The learner spends this stage on checker play alone: entering, hitting, making points, and judging when a gammon is worth staying back for. There is no cube head signal worth much, because `can_double` is false whenever the cube is disabled, so cube decisions are not recorded.

**Stage 2** plays matches to 7 with the cube enabled and with real `(1, 2, 3)` scoring. A fresh run, with no `latest_model.pt` yet, warm-starts from `checkpoints/stage1/best_model.pt`, or from `BG_INIT_FROM` when that is set. Warm start loads weights only. The optimizer, the step, and the Elo start over. A run that already has `latest_model.pt` resumes from that file. Stage 2 then learns match play: when a gammon's extra points actually change the chance of winning a match, and when to double, take, and drop. Search ranking switches from money equity to the match-equity table, because the match length is now greater than 1.

The split exists so that cube strategy is not learned at the same time as the meaning of a blot. A network that does not yet know whether a position wins cannot be taught a meaningful take point. Stage 1 builds that judgment under an objective that over-values gammons. Stage 2 inherits the judgment and retargets it at real match equity.

`BG_MATCH_TARGET`, `BG_CUBE_ENABLED`, and `BG_TRAIN_MODE` can override the stage defaults one at a time.

## 10. How a played game becomes a training target

Self-play is organized in matches. `reset` starts every game from the opening position with the cube at 1 and centered. The match score is restored afterward with `set_match_scores`. Crawford is then overwritten by the match loop described in section 3. For each decision the program stores the canonical position and context as they stood when the decision was made. A cube row is stored whenever `can_double` is true, including when the sampled action is "no double". An offer stores a second cube row for the take or the drop.

Checker decisions also store the six probabilities the search assigned to the chosen play. Cube decisions store the soft match-equity target from section 8. When the game ends, those records are turned into labels by a backward pass, separately for each player.

### Temporal difference on distributions

The return is the λ-return, with `TD_LAMBDA = 0.7`, applied to the six probabilities rather than to a scalar.

Walk the player's own decisions from the end of the game back to the start. The end of the chain is the true outcome: a one-hot vector if the game was played to a win, or the race formula's distribution if the game was cut off. For a checker decision, the label is

```
target = (1 − λ) · (search distribution at this player's next decision) + λ · (target of that next decision)
```

The last checker decision of the game has no later search distribution, so its label is the true outcome exactly. Each earlier decision is a blend of the search result at the player's following turn and everything that target already summarized. With `λ = 0.7`, seven tenths of the label comes from the continuation toward the real result, and three tenths comes from the next search. `λ = 1` would be Monte Carlo: every position labeled with the final outcome only, which is unbiased and very noisy. `λ = 0` would trust only the next search estimate, which is biased toward whatever the current network already believes. `0.7` gives a hit that pays off many turns later a path back to the position that played it, without ignoring the search.

The search distribution attached to a decision is the label's bootstrap for the previous decision of the same player. It is not the label of the decision itself. Training a position to reproduce the search that was just run on that same position would copy the network's current opinion around in a circle. Training it toward the later search and the eventual outcome is what makes the next generation of weights an improvement.

Cube decisions do not publish a new search distribution. When that player has a later checker decision, the cube row receives that decision's outcome label. When the cube row is the player's last decision, a drop included, the label is the terminal outcome. The cube head's own label is the soft target stored when the decision was made, and it does not pass through this backup.

A drop is recorded as a single, which is what a refusal is. The points added to the match are the current cube, not a gammon multiple.

After the labels are built, a game against the frozen baseline or against the champion keeps only the decisions made by the network being trained. The walk still covers the full game. A player's target mixes that player's own later search. The opponent's search distribution is the bootstrap for the opponent's earlier rows, and those rows are dropped after the walk. The learner's own later search was recorded on the position the opponent actually left, which is what keeps the bootstrap on the path that was played.

### The loss

Each optimizer step draws a batch from a prioritized replay buffer.

Every row trains the outcome head with soft-target cross-entropy: the label is already a distribution, and the loss is the cross-entropy between that distribution and the network's softmax. There is no label smoothing on this term.

Rows that were cube decisions also add the Jensen–Shannon loss of section 8, multiplied by the current cube-loss weight.

Gradients are clipped to global norm 1. On CUDA the step runs under automatic mixed precision. The optimizer is AdamW with learning rate `1e-5` and weight decay `1e-4`. A non-finite gradient skips the parameter update and leaves the sampled priorities as they were.

The replay buffer holds 50,000 transitions in stage 1 and 300,000 in stage 2. It is prioritized. New rows are inserted at priority `max_priority ** α`, with `α = 0.6`. `max_priority` starts at 1 and then tracks the largest priority stored in the tree. After a finite step, a sampled row is stored at `(|money-equity error| + 1e-5) ** α`. The error is the absolute gap between the money equity of the prediction and the money equity of the label, and `1e-5` is `MIN_PRIOR`. Positions the network is still mis-pricing are sampled more often. The importance weight is `(N · P) ** −β` with `β = 0.4`, divided by the largest weight in the batch.

### One training iteration

`src/trainer.py` and `src/trainer_vs_baseline.py` use the same five steps. They do not choose the opponent the same way. Section 11 is the exact split.

1. Read the cube schedule at the current step.
2. Play `MATCHES_PER_ITERATION` matches (default 40), split across worker processes. `src/trainer.py` runs those workers on CPU (`SELF_PLAY_DEVICE`). `src/trainer_vs_baseline.py` runs them on the training device, which is CUDA when it is available. Each worker has a copy of the weights and of the match-equity table. Games are sequential inside a worker; workers share nothing else.
3. Append the labeled transitions to the replay buffer and fold the workers' match results into the equity table.
4. If the buffer holds at least `BATCH_SIZE` rows (512 on CUDA, 256 otherwise), take `TRAIN_UPDATES_PER_ITER` optimizer steps (default 200) on the training device. A shorter buffer goes back to step 2. The step counter, the gate, and the checkpoint stay where they are.
5. When the step count is a multiple of `ELO_EVAL_INTERVAL` (default 1000), play a gate of matches and maybe promote the champion. Save `latest_model.pt` and the equity table. Save `best_model.pt` only on promotion.

Who the learner plays is described in the next section. The fraction `BASELINE_SELF_PLAY_RATIO` (one half) is self-play — the current weights against themselves — whenever an opponent is also being used.

## 11. The champion, the baseline, and the rating

Three sets of weights matter during training.

**The live network** is the one being updated. It can get worse for a while. Self-play against a worsening network teaches the wrong lessons, so the live network is not allowed to replace the published weights just because training loss moved.

**The champion** (`best_model.pt`) is the last live network that earned promotion. It is the opponent that defines "better".

**The baseline** (`checkpoints/baseline/`) is a frozen network — at the time of writing a stage-1 champion checked in with `elo: 500` at step 1000. The loader reads `best_model.pt`, `config.py`, and `match_equity.pt` from that directory; the stored Elo, not `Config.INITIAL_ELO`, is the number every "below the baseline" test compares against. The config file is what tells the loader which architecture to rebuild: the shipped one declares the current six-way head, and the legacy layout of section 5 is selected only by an older `value_policy` file. A missing or unreadable equity file leaves that opponent with a fresh table (under stage 1 the shipped 7-point table fails the target check and becomes a fresh one, which is harmless because stage-1 ranking never consults the table). A missing or unreadable model file leaves no baseline at all. The baseline exists so that early in a run the learner is measured against a fixed external player, not only against a copy of itself.

### When the baseline is the opponent

The live rating lags the champion, on purpose, given the small Elo step size below. The decision to face the baseline follows the champion's published rating only. Training games include the baseline when one is loaded and the champion's Elo is strictly below the baseline's. When the champion reaches the baseline, those games stop. In `src/trainer.py` the iteration then becomes pure self-play. In `src/trainer_vs_baseline.py` the non-self-play half switches from the baseline to the champion, so half the matches remain a test against the best published weights. With no baseline loaded, `src/trainer.py` is pure self-play from the start, and `src/trainer_vs_baseline.py` still plays that other half against the champion.

Against either opponent the learner sits white or black at random for the whole match. Both players' decisions are generated, and only the learner's decisions are stored.

### The gate

Every evaluation interval the live network plays `GATE_GAMES` matches (default 100) on CPU, greedy, no exploration. While the champion is still rated below the baseline, half of those matches are against the baseline and half against the champion. An odd count gives the extra match to the champion. Once the champion's Elo is no longer below the baseline's, every match is against the champion. Even-numbered matches start with the candidate as white and odd-numbered matches as black, and the two sides swap after every game. A game that reaches 1000 turns with no borne-off winner uses the same pip rule as self-play. That cap is the default argument of `play_single_game`, separate from `MAX_GAME_MOVES`.

Elo uses scale 400 and `K = 1`. The per-game difference is first clamped to ±`ELO_SCALE` (never binding at these win rates), and the update is multiplied by the number of games:

```
change = K · (actual win rate − expected win rate) · games
```

Against an equal opponent, a 60 percent score over 100 games is `+10` Elo. The step is deliberately small.

Promotion happens only when the win rate against the champion is strictly above `GATE_WIN_RATE` (default 54 percent). The live rating is always updated from the whole evaluation mix (total wins vs the weighted opponent Elo). That mixed update is the only rating path. On a gate pass, `best_model.pt` takes that updated live rating when it is higher, and otherwise keeps its own — the champion's Elo never moves backwards. A failed gate leaves the champion's file and Elo untouched, and the live rating still keeps the mix update.

`passes_gate` promotes only when the win rate against `best_model` is strictly greater than `GATE_WIN_RATE`. The default is 0.54; `BG_GATE_WIN_RATE` overrides it. If an evaluation somehow contained no champion games, the gate falls back to the total win rate, but the split always reserves at least half the matches for the champion, so that path is unreachable in practice. A failed comparison leaves the champion weights on disk.

## 12. Playing a human

`python -m scripts.play_web` serves `ui/html_ui.html` on port 8080 and a WebSocket on port 8765. The game object is constructed with `train_mode=False`, so a borne-off win adds `cube` times 1, 2, or 3 to the match even when the process was started under stage 1. The initial match length is `Config.MATCH_TARGET`. `new_match` can replace the game's own length with an integer clamped to 1–21. That length ends the match, drives Crawford on this server, and is what `can_offer_double` compares with the cube. Search ranking and the network's score features keep using `Config`. `Config.MATCH_TARGET` of 1 ranks plays by money equity; a greater value ranks them with the match table from section 7. While `TRAIN_MODE` is on, the money weights are 1, 3, and 5, including on a stage-1 server whose scoreboard uses 1, 2, and 3. Changing the length in the browser leaves both of those on `Config`. Score features are divided by `Config.MATCH_TARGET`. Search depth is `Config.SEARCH_PLY`, because the server constructs `Searcher` with no ply argument.

Cube offers on this server do not consult `CUBE_ENABLED`. They use `can_offer_double`, described below. At the stage-1 default, match length 1, that function still refuses a double: the cube starts at 1 and `1 >= 1 − min(scores)`.

The server loads the first checkpoint that exists, in the order stage 2, stage 1, baseline, and inside each directory `best_model.pt` before `latest_model.pt`. A directory with `config.py` is rebuilt from that file. The baseline's file sets `HEAD_KIND` to `outcome`, so that checkpoint rebuilds as an ordinary current-architecture transformer; only an older `value_policy` file would activate the legacy adapter of section 5. A stage checkpoint with no `config.py` loads into the current architecture. Along the same list, the first `match_equity.pt` is loaded before a model is chosen, and the file beside the model that actually loads replaces it when that file is present. `load` raises unless the table was saved for `Config.MATCH_TARGET`, and the two loads treat that differently. The early scan is not inside a try block: a table saved for a different match length aborts construction of the game server when the first client connects, so with the shipped 7-point `checkpoints/stage2/match_equity.pt` in place, a stage-1 default (`BG_STAGE` unset) fails every connection until the process is started under `BG_STAGE=2` or that file is moved. The load beside a model is inside the per-checkpoint try, so there a mismatch is caught and the next checkpoint is tried.

Human moves are checked with `get_legal_moves`, one step at a time. An illegal click is rejected and the board stays as it was. The AI asks the search for one complete play and then plays those steps in order so the board can animate them. AI decisions are greedy. In "human vs AI" the human is white and the network is black. "AI vs human" reverses that. "AI vs AI" lets the single loaded network play both sides.

The process started by `python -m scripts.play_web` is the server in that file. It does not import `ui/ws_server.py`. That other file is a separate, simpler server: WebSocket on port 8765 only, no HTTP file server, so `ui/html_ui.html` must be opened from disk. It loads only `checkpoints/baseline/best_model.pt` and `latest_model.pt`, always into the current architecture (no `config.py` rebuild), and builds its searcher with no equity table, so its search ranks every play by money equity however long the match is. Its `ai_move` rolls and plays checkers, and never calls the cube head. A human double there goes through `BackgammonGame.can_double`, so `CUBE_ENABLED` applies and dice already on the board cannot be doubled — including the opening roll, which `scripts/play_web.py` would allow.

In `scripts/play_web.py` the AI, before rolling, calls `get_learned_cube_decision` with `stochastic=False` when `can_offer_double` is true. That helper is not `BackgammonGame.can_double`. It does not read `CUBE_ENABLED`. It returns false once the side to move has rolled, except while `opening_pending` is set, so an opening roll that is already on the board can be offered. It also returns false in a Crawford game, when the player does not own the cube, and when `cube >= match_target − min(scores)`. A taken double calls `apply_double`, and `apply_double` calls `can_double`. Dice still showing, or `CUBE_ENABLED` left false by stage 1, make that call return false, so the take leaves the cube unchanged. A drop does end the game: `handle_cube_refusal` awards the current cube to the player who offered. The take or drop action is the argmax of the cube head. The call uses `is_take=True` and `stochastic=False`, and it does not pass the doubler's flipped outcome distribution. That distribution only prices the soft target, which this server throws away, so the action is the cube head either way. Self-play passes the flipped distribution because there the soft target is the cube loss. With no model loaded, the take or drop is a coin flip.

Checker play calls `search` with the default `stochastic=False` and plays `result.best()`. The chosen steps are applied with `step_atomic`, which checks each step against `get_legal_moves`. If the search returns no play while dice remain and some atomic move is legal, the server samples one of those atomic moves at random.

## 13. Where the code lives

```
src/config.py                 stages, scores, model size, search, schedule
src/engine.py                 rules, encoding, cube, Crawford
src/model.py                  transformer, CNN, legacy baseline adapter
src/mcts.py                   one-ply and two-ply search
src/search.py                 re-exports the search
src/utils/outcome.py          six-way equity, race formula, perspective flip
src/utils/match_equity.py     learned match-equity table
src/utils/cube.py             double and take pricing, soft targets
src/utils/game.py             self-play, backups, baseline games
src/utils/train.py            one optimizer step
src/utils/distribution.py     label smoothing, Jensen–Shannon loss
src/utils/elo.py              rating, gate, mixed evaluation
src/utils/checkpoint.py       save, load, warm start, legacy loader
src/utils/history.py          legacy flat-reward helper; no importer in the pipeline
src/utils/move.py             legacy index/format helper; no importer in the pipeline
src/replay_buffer.py          prioritized replay
src/trainer.py                self-play training loop
src/trainer_vs_baseline.py    same loop with an explicit stage and opponent
scripts/play_web.py           browser game
ui/html_ui.html               the board
ui/ws_server.py               separate browser server; its AI does not cube
checkpoints/baseline/         frozen opponent and its saved architecture
checkpoints/stage1/           live stage-1 run
checkpoints/stage2/           live stage-2 run
tests/test.py                 regression suite for this document
tests/test_checkpoint.py      checkpoint probe; doubles as a manual Elo editor
tests/write_table.py          print a saved match-equity table
```

Everything imports from the repository root (`from src.…`), so the entry points must run in module form with the repo root as the working directory: `python -m src.trainer`, `python -m src.trainer_vs_baseline --stage 1`, `python -m scripts.play_web`, `python -m tests.test`. Running a file path directly (`python src/trainer.py`) puts the wrong directory on `sys.path` and dies with `ModuleNotFoundError: No module named 'src'`. The `bg-train` console script in `pyproject.toml` is stale in the same way: it names `backgammon.trainer:train`, a module path the repository no longer contains.

`python -m tests.test` is the executable version of this document's claims. It checks canonical perspective symmetry, the flip/equity helpers and the race-formula band, complete-turn enumeration and `apply_turn` consistency, that a search restores the game object untouched and visits at most `SEARCH_PRUNE_TOP_K` plays, the same-player backup (`Q` equals the leaf equity), high-temperature exploration, the TD(λ) recursion including the cube row sharing the next move decision's target, a self-play game through a short move cap and through one vectorized training step with a decreasing loss, the prune margin, the stage cube gates, a 52.5-percent promotion arithmetic, the lagged-rating rule that keeps training on self-play, the eval split and weighted opponent Elo, the legacy adapter's output shapes, the opening roll, the cube-offered encoding, the price of a certain backgammon loss, the worker match split, and the evaluation move cap naming a one-point winner. One guard is about repo hygiene rather than theory: it demands a saved AdamW checkpoint stay under 15 MB, the number behind the depth comment in `src/config.py`. At the shipped 10-layer width 128 model the file lands at roughly 15.6 MB, so this check currently reports a failure while every behavioral check passes.

A checkpoint file holds the weights, the optimizer, the step, the Elo, the recent loss, and a small record of model type, width, and depth. The equity table is a sibling file, `match_equity.pt`, because it is data about match scores rather than part of the network.

## 14. The path through one decision

Put together, a single checker decision in self-play does the following.

The engine rolls, or keeps the opening roll. The opening dice are already out, so self-play cannot double before that roll is played. On a later turn, if the cube is enabled, it is not Crawford, and the player may double, the cube head samples an action and the match-equity price is stored as the cube label. A declined double stores that one row. An offered double stores the response as well. A drop ends the game as a single. A take doubles the stake and play continues.

If the position is a gammon-free race, self-play stops. The race formula picks a winner, scores a single, and is the terminal distribution the backward pass starts from. Otherwise the search lists every legal play, scores the afterstates from the opponent's seat, flips them, converts them to equity, prunes, optionally expectimaxes one reply, and runs the bandit. The chosen play is applied. The position from before the play, and the six probabilities of the play, are stored.

At the end of the game the backward pass writes an outcome distribution on every stored position. Those rows wait in the replay buffer until a training step samples them, moves the outcome head toward the backed-up distribution, and, for cube rows, moves the cube head toward the priced soft label.

The champion changes only when the live network wins strictly more than `GATE_WIN_RATE` of a greedy gate, played on CPU, against the current champion.

## 15. Knobs that change the theory

Most fields in `src/config.py` are sizes and rates. These are the ones that change what is being optimized or how a decision is reached.

| Knob | Role |
|------|------|
| `BG_STAGE` | `1` aggressive one-point checker play. `2` seven-point matches with the cube. A fresh stage-2 run warm-starts from stage 1. |
| `R_WIN`, `R_GAMMON`, `R_BACKGAMMON` | Training-mode point values, default `1, 3, 5`. Wider than `1, 2, 3`, which is why stage 1 is aggressive. |
| `BG_SEARCH_PLY` | `1` scores the position you leave. `2` averages the opponent's best reply to every roll. Training defaults to 1. |
| `SEARCH_PRUNE_TOP_K`, `SEARCH_PRUNE_MARGIN` | How many near-best plays receive the bandit and, at ply 2, the expectimax. |
| `NUM_SIMULATIONS`, `C_PUCT` | Visit budget and exploration constant inside the bandit. |
| `TD_LAMBDA` | How far outcome labels lean on the real game result versus the next search. |
| `CUBE_CURRICULUM_STAGES` | How often self-play doubles at random, and how hard the cube loss is weighted. |
| `CUBE_ME_TEMPERATURE` | How soft a near-even cube label is. |
| `GATE_WIN_RATE`, `GATE_GAMES` | How often, and by how much, the live network must beat the champion before it becomes the champion. Defaults 0.54 and 100. |
| `BASELINE_SELF_PLAY_RATIO` | Share of training matches that are self-play while an external opponent is also in use. |

Change the stage, the point values, or the ply, and you are changing the game the network is trying to solve. Change the learning rate or the batch size, and you are only changing how fast it walks there.

Twelve fields are read from the environment once, when `src/config.py` is first imported: `BG_STAGE`, `BG_MATCH_TARGET`, `BG_CUBE_ENABLED`, `BG_TRAIN_MODE`, `BG_NUM_SIMULATIONS`, `BG_SEARCH_PLY`, `BG_BUFFER_SIZE` (defaults 50,000 / 300,000 by stage), `BG_GATE_WIN_RATE`, `BG_GATE_GAMES`, `BG_CHECKPOINT_DIR`, `BG_INIT_FROM`, and `BG_BASELINE_DIR`. A long-running process keeps its first answer; the rest of `Config` — model sizes, pruning constants, temperatures, the cube schedule, the optimizer settings — changes only by editing the file.
