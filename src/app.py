"""Terminal interface for Rank-NDCG ranked-watchlist inference."""

from src.inference.ranked_watchlist import (
    RankedWatchlistError,
    rank_candidate_universe,
)


DEFAULT_RESEARCH_UNIVERSE = "large_mega_cap_stocks"
WELCOME_MESSAGE = "\nWelcome to the Rank-NDCG ranked watchlist.\n"
GOODBYE = "Exiting the program. Thank you and goodbye!"


def _prompt_horizon() -> int | None:
    """Require an explicit supported ranking horizon, or let the user quit."""
    while True:
        print("\nChoose a prediction horizon:")
        print("  1. 10 trading days")
        print("  2. 20 trading days")
        choice = input("Choice (1 or 2, q to quit): ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice == "1":
            return 10
        if choice == "2":
            return 20
        print("Invalid horizon choice. Please select 1 or 2.")


def _prompt_candidate_source() -> dict | None:
    """Prompt for one backend candidate source without normalizing ticker input."""
    while True:
        print("\nChoose a candidate source:")
        print(
            "  1. Configured default research universe "
            f"({DEFAULT_RESEARCH_UNIVERSE})"
        )
        print("  2. Custom comma-separated ticker list")
        choice = input("Choice (1 or 2, q to quit): ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice == "1":
            return {"universe": DEFAULT_RESEARCH_UNIVERSE}
        if choice == "2":
            ticker_text = input("Enter tickers separated by commas: ")
            return {"tickers": ticker_text.split(",")}
        print("Invalid candidate-source choice. Please select 1 or 2.")


def display_ranked_watchlist(result: dict) -> None:
    """Print one backend ranking result in a compact, terminal-friendly layout."""
    source = result["candidate_source"]
    if source["type"] == "configured_universe":
        source_label = f"Configured universe: {source['universe_name']}"
    else:
        source_label = "Custom ticker list"

    print("\nRank-NDCG Ranked Watchlist")
    print(f"As-of date: {result['as_of_date']}")
    print(f"Horizon: {result['prediction_days']} trading days")
    print(f"Candidate source: {source_label}")
    print(f"Candidates ranked: {result['ranked_count']}")
    print(f"Candidates skipped: {result['skipped_count']}")
    print("\nRank | Ticker | Rank-NDCG score")
    print("-----+--------+----------------")
    for row in result["ranked_rows"]:
        print(
            f"{row['rank']:>4} | {row['ticker']:<6} | "
            f"{row['ranker_score']:.6f}"
        )
    print("\nRank-NDCG score is a ranking score, not a predicted return or probability.")

    if result["skipped_tickers"]:
        print("\nSkipped tickers:")
        for skipped in result["skipped_tickers"]:
            print(f"  - {skipped['ticker']}: {skipped['reason']}")

    print(f"\nBundle ID: {result['bundle_id']}")


def _should_run_again() -> bool:
    """Return whether to start another explicit ranking request."""
    return input("\nRun another ranking? [y/N]: ").strip().lower() in {"y", "yes"}


def main_menu() -> None:
    """Run the interactive Rank-NDCG ranked-watchlist console flow."""
    print(WELCOME_MESSAGE)
    while True:
        prediction_days = _prompt_horizon()
        if prediction_days is None:
            print(GOODBYE)
            return

        candidate_source = _prompt_candidate_source()
        if candidate_source is None:
            print(GOODBYE)
            return

        try:
            result = rank_candidate_universe(prediction_days, **candidate_source)
        except RankedWatchlistError as error:
            print(f"\nUnable to generate ranked watchlist: {error}")
        else:
            display_ranked_watchlist(result)

        if not _should_run_again():
            print(GOODBYE)
            return


if __name__ == "__main__":
    main_menu()
