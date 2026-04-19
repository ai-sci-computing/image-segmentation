# src/models/ — Model factories for all 5 attempts
#
# Each module exposes a factory function that returns a ready-to-train
# nn.Module with the correct input/output signature:
#   input:  (B, 3, 1024, 1024) float32
#   output: (B, 1, 1024, 1024) float32 logits (pre-sigmoid)
