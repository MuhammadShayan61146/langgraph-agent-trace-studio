"""Offline fixture corpora for the experiments.

These are **synthetic fixtures**, not real studies: URLs live under
``example.org/fixtures`` and every title is prefixed ``Fixture``. They exist
so scenarios are repeatable offline. Each snippet starts with a stance tag
(``[supports]``, ``[disputes]``, ``[context]``) and carries signed
percentages the Analyst can feed to the calculator tool.
"""

from __future__ import annotations

import re

from agentic_research.tools.search import SearchResultItem

_BASE = "https://example.org/fixtures"


def _i(slug: str, title: str, snippet: str) -> SearchResultItem:
    return SearchResultItem(title=f"Fixture: {title}", snippet=snippet, url=f"{_BASE}/{slug}")


#: topic -> (keywords, items)
CORPORA: dict[str, tuple[set[str], list[SearchResultItem]]] = {
    "remote-work": (
        {"remote", "work", "productivity", "hybrid", "distributed", "office", "teams"},
        [
            _i("remote-a", "Output gains in distributed teams",
               "[supports] Self-reported task output rose +13% for fully remote teams after a 12-month rollout."),
            _i("remote-b", "Hybrid schedules and delivery speed",
               "[supports] Hybrid teams shipped features +9% faster, driven by fewer interruptions."),
            _i("remote-c", "Collaboration decay in remote engineering",
               "[disputes] Cross-team collaboration fell -18% where no office days were kept, slowing complex projects."),
            _i("remote-d", "Onboarding outcomes without an office",
               "[disputes] New hires reached full productivity -11% slower when onboarded entirely remotely."),
            _i("remote-e", "Measurement caveats in productivity research",
               "[context] Most remote-work studies rely on self-reported output; role mix differs by up to 40% between samples."),
        ],
    ),
    "four-day-week": (
        {"four", "day", "week", "workweek", "shorter", "hours", "wellbeing"},
        [
            _i("4dw-a", "Pilot results from a 61-company trial",
               "[supports] Revenue held roughly flat (+1%) while reported burnout dropped -39%."),
            _i("4dw-b", "Public-sector scheduling trial",
               "[supports] Service-desk resolution rates improved +6% with compressed schedules."),
            _i("4dw-c", "Coverage gaps in customer-facing roles",
               "[disputes] Response-time targets slipped -7% in teams without staggered coverage."),
            _i("4dw-d", "Long-run retention data",
               "[context] Voluntary attrition fell -22% in year one; year-two data is limited to 14 firms."),
        ],
    ),
    "urban-cycling": (
        {"urban", "cycling", "bike", "lanes", "infrastructure", "city", "cities"},
        [
            _i("cycle-a", "Protected lane adoption study",
               "[supports] Cities with protected lanes saw +45% more daily cycling trips within two years."),
            _i("cycle-b", "Injury rates near new lanes",
               "[supports] Serious injuries per million trips fell -28% on corridors with physical separation."),
            _i("cycle-c", "Retail impact review",
               "[context] Neighbourhood retail sales changed +4% on average; sample limited to 9 corridors."),
        ],
    ),
    "cycling-costs": (
        {"cost", "costs", "funding", "budget", "maintenance", "price", "expensive", "paying"},
        [
            _i("cost-a", "Capital cost per kilometre of protected lane",
               "[context] Median build cost was 1.2 million per km; painted lanes were 85% cheaper but offered less protection."),
            _i("cost-b", "Maintenance budgets after year five",
               "[disputes] Maintenance overruns of +15% appeared where winter clearing was not budgeted."),
            _i("cost-c", "Return on infrastructure spend",
               "[supports] Modelled health and congestion benefits exceeded costs by +30% over 20 years."),
        ],
    ),
    "home-battery": (
        {"home", "battery", "solar", "storage", "household", "energy", "grid"},
        [
            _i("batt-a", "Self-consumption uplift with storage",
               "[supports] Households raised solar self-consumption by +38% after adding a battery."),
            _i("batt-b", "Payback period analysis",
               "[disputes] Payback exceeded 10 years in markets with flat tariffs; savings were only +6% annually."),
            _i("batt-c", "Grid resilience during outages",
               "[context] 71% of surveyed owners cited backup power as the primary motivation, not savings."),
            _i("batt-d", "Degradation over cycles",
               "[disputes] Usable capacity fell -12% after five years of daily cycling."),
        ],
    ),
}

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class CorpusSearchBackend:
    """Keyword-overlap `SearchBackend` over `CORPORA` (satisfies the Protocol).

    Picks the topic whose keywords overlap the query most, then ranks that
    topic's items by overlap with the query. Returns ``[]`` if nothing
    overlaps -- so unrelated queries behave like a search with no hits.
    """

    def search(self, query: str, max_results: int) -> list[SearchResultItem]:
        q = _tokens(query)
        best_topic, best_score = None, 0
        for topic, (keywords, _) in CORPORA.items():
            score = len(q & keywords)
            if score > best_score:
                best_topic, best_score = topic, score
        if best_topic is None:
            return []
        items = CORPORA[best_topic][1]
        ranked = sorted(
            enumerate(items),
            key=lambda p: (-len(q & _tokens(p[1].title + " " + p[1].snippet)), p[0]),
        )
        return [item for _, item in ranked][:max_results]
