import sys

# Exit 0 (good) = feature doesn't exist yet
# Exit 1 (bad) = feature exists
if hasattr(sys, "_jit"):
    sys.exit(1)  # Feature exists - mark as "bad"
sys.exit(0)  # Feature doesn't exist - mark as "good"
