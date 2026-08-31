"""Deprecated historical smoke-check entry point.

The former train-then-predict workflow used legacy predictor paths that are not
compatible with the current isolated artifact bundles. It is intentionally
disabled rather than presented as a runnable sanity check.

For the current Rank-NDCG workflow, use the trainer and application commands in
README.md.
"""

DEPRECATION_MESSAGE = (
    "Deprecated historical script: the legacy train-then-predict artifact paths "
    "are not compatible with current Rank-NDCG bundles. See README.md."
)


def main() -> None:
    """Explain why the obsolete workflow is disabled."""
    print(DEPRECATION_MESSAGE)


if __name__ == "__main__":
    main()
