"""Python side of the chess AlphaZero agent: net, training, self-play, eval.

The C++ module (`chess_engine`, built at the repo root) owns the board, the
encoders and the batched MCTS runner. Everything here is torch: the network
(`az.net`, docs/03), and later the bootstrap, self-play and eval drivers.
"""
