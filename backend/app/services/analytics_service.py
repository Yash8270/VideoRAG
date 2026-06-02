"""
Analytics service — computes engagement metrics for YouTube videos
and Instagram Reels, and produces a side-by-side comparison summary.

Core formula:
    engagement_rate = ((likes + comments) / views) * 100

Edge cases handled:
    • views is None or 0            → engagement_rate = None, flagged
    • likes is None                 → treated as 0, flagged
    • comments is None              → treated as 0, flagged
    • likes and comments both None  → engagement_rate = None, flagged

Public interface:
    from_youtube(data)     → VideoAnalyticsSummary
    from_instagram(data)   → VideoAnalyticsSummary
    compute_analytics(inp) → VideoAnalyticsSummary
    compare(yt, ig)        → ComparisonSummary
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional, Any

from pydantic import BaseModel, Field, field_validator

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.services.instagram_service import InstagramReelData
    from app.services.youtube_service import YouTubeVideoData

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Platform engagement benchmarks
# ─────────────────────────────────────────────────────────────────────────────
# Sources: Social media industry averages (Hootsuite / Sprout Social 2024-25)
# Engagement rate = (likes + comments) / views × 100

_BENCHMARKS: dict[str, dict[str, float]] = {
    "youtube": {
        "low":     0.5,
        "average": 2.0,
        "good":    5.0,
        "high":    10.0,
        # > 10 → viral
    },
    "instagram": {
        "low":     1.0,
        "average": 4.0,
        "good":    8.0,
        "high":    15.0,
        # > 15 → viral
    },
    # Generic fallback for unknown platforms
    "generic": {
        "low":     1.0,
        "average": 3.0,
        "good":    6.0,
        "high":    10.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class Platform(str, Enum):
    YOUTUBE   = "youtube"
    INSTAGRAM = "instagram"
    UNKNOWN   = "unknown"


class EngagementLevel(str, Enum):
    """
    Qualitative classification of the engagement rate relative to
    platform-specific industry benchmarks.
    """
    LOW     = "low"      # Below platform average — needs improvement
    AVERAGE = "average"  # Meets typical platform baseline
    GOOD    = "good"     # Above average — healthy audience interaction
    HIGH    = "high"     # Significantly above average — strong performance
    VIRAL   = "viral"    # Top tier — exceptional reach or highly shareable
    UNKNOWN = "unknown"  # Cannot be determined (zero views / missing data)


# ─────────────────────────────────────────────────────────────────────────────
# Input model
# ─────────────────────────────────────────────────────────────────────────────


class VideoAnalyticsInput(BaseModel):
    """
    Normalised input accepted by compute_analytics().
    Use from_youtube() / from_instagram() helpers to construct this
    directly from the platform-specific data classes.
    """
    platform:         Platform      = Field(..., description="Content platform")
    video_id:         str           = Field(..., description="Platform-specific unique ID")
    title:            str           = Field(..., description="Video / Reel title")
    creator:          str           = Field(..., description="Channel name or @username")
    url:              str           = Field(..., description="Canonical content URL")
    upload_date:      Optional[str] = Field(None, description="ISO 8601 upload date")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")

    # Engagement counters — all optional because platforms may hide them
    views:    Optional[int] = Field(None, ge=0, description="Total views / plays")
    likes:    Optional[int] = Field(None, ge=0, description="Total likes / hearts")
    comments: Optional[int] = Field(None, ge=0, description="Total comments")

    @field_validator("views", "likes", "comments", mode="before")
    @classmethod
    def coerce_negative_to_none(cls, v: Any) -> Any:
        """
        yt-dlp and other scrapers often use -1 to indicate that a metric
        is missing or hidden by the creator. We convert negatives to None
        so they are handled gracefully as 'missing' data.
        """
        if isinstance(v, (int, float)) and v < 0:
            return None
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Metrics model
# ─────────────────────────────────────────────────────────────────────────────


class EngagementMetrics(BaseModel):
    """
    All computed engagement metrics for a single video.
    None values mean the metric could not be calculated (see flags).
    """

    # ── Primary metric ───────────────────────────────────────────────────────
    engagement_rate:       Optional[float] = Field(
        None,
        description="((likes + comments) / views) × 100. "
                    "None when views = 0 or all interaction data is missing.",
    )

    # ── Secondary rates ──────────────────────────────────────────────────────
    like_rate:             Optional[float] = Field(
        None, description="(likes / views) × 100 — percentage of viewers who liked"
    )
    comment_rate:          Optional[float] = Field(
        None, description="(comments / views) × 100 — percentage of viewers who commented"
    )
    like_to_comment_ratio: Optional[float] = Field(
        None, description="likes ÷ comments — how many likes per comment"
    )

    # ── Normalised raw counters (None → 0, documented in flags) ─────────────
    effective_views:    int = Field(..., description="views or 0 if None/missing")
    effective_likes:    int = Field(..., description="likes or 0 if None/missing")
    effective_comments: int = Field(..., description="comments or 0 if None/missing")
    total_interactions: int = Field(..., description="effective_likes + effective_comments")

    # ── Formatted display strings ─────────────────────────────────────────────
    views_formatted:        str           = Field(..., description="e.g. '1.2M', '450K', '8,342'")
    likes_formatted:        str           = Field(..., description="Formatted like count or 'Hidden'")
    comments_formatted:     str           = Field(..., description="Formatted comment count or 'Hidden'")
    engagement_rate_label:  str           = Field(..., description="e.g. '3.45%' or 'N/A'")

    # ── Classification ───────────────────────────────────────────────────────
    engagement_level:        EngagementLevel = Field(..., description="Qualitative performance tier")
    benchmark_context:       str             = Field(
        ..., description="How this rate compares to the platform average"
    )

    # ── Data quality flags ────────────────────────────────────────────────────
    has_zero_views:          bool = Field(..., description="True when views = 0")
    has_missing_views:       bool = Field(..., description="True when views was not reported")
    has_missing_likes:       bool = Field(..., description="True when likes was not reported")
    has_missing_comments:    bool = Field(..., description="True when comments was not reported")
    is_engagement_reliable:  bool = Field(
        ...,
        description="False when any key counter is missing — rate may underestimate true engagement",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output models
# ─────────────────────────────────────────────────────────────────────────────


class VideoAnalyticsSummary(BaseModel):
    """Complete analytics output for a single video."""

    input:          VideoAnalyticsInput = Field(..., description="Normalised input data")
    metrics:        EngagementMetrics   = Field(..., description="All computed metrics")
    insights:       list[str]           = Field(..., description="Human-readable insight sentences")
    computed_at:    str                 = Field(..., description="UTC timestamp of computation")


class ComparisonResult(BaseModel):
    """Outcome of head-to-head YouTube vs Instagram comparison."""

    metric:         str           = Field(..., description="Metric being compared")
    youtube_value:  Optional[str] = Field(None, description="YouTube value (formatted)")
    instagram_value:Optional[str] = Field(None, description="Instagram value (formatted)")
    winner:         Optional[str] = Field(None, description="'youtube', 'instagram', or 'tied'")
    delta:          Optional[str] = Field(None, description="Difference between values")


class ComparisonSummary(BaseModel):
    """Side-by-side analytics comparison between a YouTube video and an Instagram Reel."""

    youtube:           VideoAnalyticsSummary = Field(..., description="YouTube analytics")
    instagram:         VideoAnalyticsSummary = Field(..., description="Instagram analytics")
    overall_winner:    Optional[str]         = Field(
        None, description="Platform with higher engagement rate ('youtube'/'instagram'/'tied')"
    )
    head_to_head:      list[ComparisonResult]= Field(..., description="Metric-by-metric comparison")
    comparison_notes:  list[str]             = Field(..., description="Narrative comparison insights")
    computed_at:       str                   = Field(..., description="UTC timestamp")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_rate(numerator: int, denominator: int, scale: float = 100.0) -> Optional[float]:
    """
    Safely compute (numerator / denominator) × scale.

    Returns:
        Rounded float or None when denominator is 0.
    """
    if denominator == 0:
        return None
    return round((numerator / denominator) * scale, 4)


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    """
    Safely compute numerator / denominator.

    Returns:
        Rounded float or None when denominator is 0.
    """
    if denominator == 0:
        return None
    return round(numerator / denominator, 2)


def _format_count(n: Optional[int], missing_label: str = "Hidden") -> str:
    """
    Convert a raw integer count to a human-readable abbreviated string.

    Examples:
        1_234_567 → '1.23M'
        450_000   → '450.0K'
        8_342     → '8,342'
        None      → missing_label
    """
    if n is None:
        return missing_label
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _classify_engagement(
    rate: Optional[float],
    platform: Platform,
) -> tuple[EngagementLevel, str]:
    """
    Map an engagement rate to a qualitative level using platform benchmarks.

    Args:
        rate:     Computed engagement rate (%) or None.
        platform: Content platform for benchmark selection.

    Returns:
        (EngagementLevel, benchmark_context_string)
    """
    if rate is None:
        return EngagementLevel.UNKNOWN, "Cannot classify — insufficient data."

    key       = platform.value if platform.value in _BENCHMARKS else "generic"
    thresholds = _BENCHMARKS[key]

    low_t     = thresholds["low"]
    avg_t     = thresholds["average"]
    good_t    = thresholds["good"]
    high_t    = thresholds["high"]

    if rate < low_t:
        level   = EngagementLevel.LOW
        context = (
            f"Below the {platform.value.capitalize()} average "
            f"({low_t}% threshold). Audience interaction is limited."
        )
    elif rate < avg_t:
        level   = EngagementLevel.AVERAGE
        context = (
            f"Meets the typical {platform.value.capitalize()} baseline "
            f"({low_t}–{avg_t}%). Healthy but room to grow."
        )
    elif rate < good_t:
        level   = EngagementLevel.GOOD
        context = (
            f"Above the {platform.value.capitalize()} average "
            f"({avg_t}–{good_t}%). Strong audience interaction."
        )
    elif rate < high_t:
        level   = EngagementLevel.HIGH
        context = (
            f"Significantly above the {platform.value.capitalize()} average "
            f"({good_t}–{high_t}%). Excellent content performance."
        )
    else:
        level   = EngagementLevel.VIRAL
        context = (
            f"Top-tier {platform.value.capitalize()} engagement "
            f"(>{high_t}%). Viral-level audience resonance."
        )

    return level, context


def _generate_insights(
    inp: VideoAnalyticsInput,
    metrics: EngagementMetrics,
) -> list[str]:
    """
    Produce a list of human-readable, actionable insight sentences about
    the video's performance.

    Args:
        inp:     Normalised video input.
        metrics: Computed engagement metrics.

    Returns:
        List of insight strings (3–8 items typically).
    """
    insights: list[str] = []
    platform = inp.platform.value.capitalize()

    # ── Engagement rate ───────────────────────────────────────────────────────
    if metrics.engagement_rate is not None:
        insights.append(
            f"Engagement rate is {metrics.engagement_rate_label} — "
            f"{metrics.engagement_level.value.upper()} tier for {platform}."
        )
    else:
        insights.append(
            "Engagement rate could not be calculated — "
            "views, likes or comments data is unavailable."
        )

    # ── View scale ────────────────────────────────────────────────────────────
    if metrics.effective_views > 0:
        insights.append(
            f"Video has reached {metrics.views_formatted} views, "
            + (
                "indicating broad distribution."
                if metrics.effective_views >= 1_000_000
                else "indicating moderate reach."
                if metrics.effective_views >= 100_000
                else "still building an audience."
            )
        )

    # ── Like rate ─────────────────────────────────────────────────────────────
    if metrics.like_rate is not None:
        insights.append(
            f"{metrics.like_rate:.2f}% of viewers liked the content "
            + (
                "— very strong positive sentiment."
                if metrics.like_rate >= 5
                else "— average positive sentiment."
                if metrics.like_rate >= 1
                else "— below-average like rate; content may not be resonating."
            )
        )

    # ── Comment rate ──────────────────────────────────────────────────────────
    if metrics.comment_rate is not None:
        if metrics.comment_rate >= 0.5:
            insights.append(
                f"Comment rate of {metrics.comment_rate:.2f}% suggests viewers are "
                "actively discussing and sharing opinions."
            )
        else:
            insights.append(
                f"Comment rate is low ({metrics.comment_rate:.2f}%). "
                "Adding a call-to-action may drive more discussion."
            )

    # ── Like-to-comment ratio ─────────────────────────────────────────────────
    if metrics.like_to_comment_ratio is not None:
        ratio = metrics.like_to_comment_ratio
        if ratio >= 20:
            insights.append(
                f"Like-to-comment ratio is {ratio:.1f}:1 — "
                "viewers prefer passive engagement (liking over commenting)."
            )
        elif ratio >= 5:
            insights.append(
                f"Like-to-comment ratio is {ratio:.1f}:1 — "
                "healthy balance of passive and active engagement."
            )
        else:
            insights.append(
                f"Like-to-comment ratio is {ratio:.1f}:1 — "
                "unusually high comment activity relative to likes."
            )

    # ── Missing data warnings ─────────────────────────────────────────────────
    missing: list[str] = []
    if metrics.has_missing_likes:
        missing.append("likes")
    if metrics.has_missing_comments:
        missing.append("comments")
    if metrics.has_missing_views:
        missing.append("views")

    if missing:
        insights.append(
            f"⚠ {platform} did not expose: {', '.join(missing)}. "
            "Reported metrics may underestimate true engagement."
        )

    # ── Duration context ──────────────────────────────────────────────────────
    if inp.duration_seconds:
        mins = inp.duration_seconds / 60
        if mins < 1:
            insights.append(
                f"Short-form content ({inp.duration_seconds}s) — "
                "typically benefits from higher replay rates."
            )
        elif mins > 20:
            insights.append(
                f"Long-form content ({mins:.0f} min) — "
                "sustained engagement over duration indicates strong retention."
            )

    return insights


def _utc_now() -> str:
    """Return the current UTC datetime as an ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_metrics(inp: VideoAnalyticsInput) -> EngagementMetrics:
    """
    Calculate all engagement metrics from a VideoAnalyticsInput.

    Applies the formula:
        engagement_rate = ((likes + comments) / views) * 100

    with full edge-case handling for missing and zero values.

    Args:
        inp: Normalised VideoAnalyticsInput.

    Returns:
        Fully populated EngagementMetrics instance.
    """
    # ── Normalise raw counts ──────────────────────────────────────────────────
    has_missing_views    = inp.views    is None
    has_missing_likes    = inp.likes    is None
    has_missing_comments = inp.comments is None

    eff_views    = inp.views    if inp.views    is not None else 0
    eff_likes    = inp.likes    if inp.likes    is not None else 0
    eff_comments = inp.comments if inp.comments is not None else 0
    has_zero_views = eff_views == 0

    total_interactions = eff_likes + eff_comments

    # ── Engagement rate ───────────────────────────────────────────────────────
    # Formula: ((likes + comments) / views) * 100
    # Guard:   views == 0 → None  (division by zero)
    engagement_rate: Optional[float] = _safe_rate(total_interactions, eff_views)

    # ── Secondary rates ───────────────────────────────────────────────────────
    like_rate:             Optional[float] = _safe_rate(eff_likes,    eff_views)
    comment_rate:          Optional[float] = _safe_rate(eff_comments, eff_views)
    like_to_comment_ratio: Optional[float] = _safe_ratio(eff_likes,   eff_comments)

    # ── Classification ────────────────────────────────────────────────────────
    engagement_level, benchmark_context = _classify_engagement(
        engagement_rate, inp.platform
    )

    # ── Reliability flag ──────────────────────────────────────────────────────
    # Rate is unreliable when ANY counter is missing (platform hid it)
    is_reliable = not (has_missing_views or has_missing_likes or has_missing_comments)

    # ── Formatted display ─────────────────────────────────────────────────────
    engagement_rate_label = (
        f"{engagement_rate:.2f}%" if engagement_rate is not None else "N/A"
    )

    return EngagementMetrics(
        # Core formula result
        engagement_rate=engagement_rate,

        # Secondary metrics
        like_rate=like_rate,
        comment_rate=comment_rate,
        like_to_comment_ratio=like_to_comment_ratio,

        # Normalised counters
        effective_views=eff_views,
        effective_likes=eff_likes,
        effective_comments=eff_comments,
        total_interactions=total_interactions,

        # Formatted labels
        views_formatted=_format_count(inp.views, missing_label="0"),
        likes_formatted=_format_count(inp.likes),
        comments_formatted=_format_count(inp.comments),
        engagement_rate_label=engagement_rate_label,

        # Classification
        engagement_level=engagement_level,
        benchmark_context=benchmark_context,

        # Data quality flags
        has_zero_views=has_zero_views,
        has_missing_views=has_missing_views,
        has_missing_likes=has_missing_likes,
        has_missing_comments=has_missing_comments,
        is_engagement_reliable=is_reliable,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_analytics(inp: VideoAnalyticsInput) -> VideoAnalyticsSummary:
    """
    Compute a full analytics summary for any video given normalised input.

    Args:
        inp: VideoAnalyticsInput — use from_youtube() or from_instagram()
             to build this from the platform-specific data classes.

    Returns:
        VideoAnalyticsSummary with metrics + insights.

    Example:
        inp = VideoAnalyticsInput(
            platform=Platform.YOUTUBE,
            video_id="dQw4w9WgXcQ",
            title="Never Gonna Give You Up",
            creator="Rick Astley",
            url="https://youtu.be/dQw4w9WgXcQ",
            views=1_600_000_000,
            likes=16_000_000,
            comments=2_000_000,
        )
        summary = compute_analytics(inp)
        print(summary.metrics.engagement_rate)  # → 1.125
        print(summary.metrics.engagement_level) # → EngagementLevel.AVERAGE
    """
    logger.info(
        "compute_analytics: platform=%s | video_id=%s | views=%s | likes=%s | comments=%s",
        inp.platform.value, inp.video_id,
        inp.views, inp.likes, inp.comments,
    )

    metrics  = _compute_metrics(inp)
    insights = _generate_insights(inp, metrics)

    logger.info(
        "Analytics result: engagement_rate=%s | level=%s | reliable=%s",
        metrics.engagement_rate_label,
        metrics.engagement_level.value,
        metrics.is_engagement_reliable,
    )

    return VideoAnalyticsSummary(
        input=inp,
        metrics=metrics,
        insights=insights,
        computed_at=_utc_now(),
    )


def from_youtube(data: "YouTubeVideoData") -> VideoAnalyticsSummary:
    """
    Convenience wrapper — build analytics directly from a YouTubeVideoData.

    Args:
        data: YouTubeVideoData returned by youtube_service.extract_youtube().

    Returns:
        VideoAnalyticsSummary.
    """
    inp = VideoAnalyticsInput(
        platform=Platform.YOUTUBE,
        video_id=data.video_id,
        title=data.title,
        creator=data.creator,
        url=data.url,
        upload_date=data.upload_date,
        duration_seconds=data.duration_seconds,
        views=data.views,
        likes=data.likes,
        comments=data.comments,
    )
    return compute_analytics(inp)


def from_instagram(data: "InstagramReelData") -> VideoAnalyticsSummary:
    """
    Convenience wrapper — build analytics directly from an InstagramReelData.

    Args:
        data: InstagramReelData returned by instagram_service.extract_instagram().

    Returns:
        VideoAnalyticsSummary.
    """
    inp = VideoAnalyticsInput(
        platform=Platform.INSTAGRAM,
        video_id=data.shortcode,
        title=data.title,
        creator=data.creator,
        url=data.url,
        upload_date=data.upload_date,
        duration_seconds=data.duration_seconds,
        views=data.views,
        likes=data.likes,
        comments=data.comments,
    )
    return compute_analytics(inp)


def compare(
    youtube_summary: VideoAnalyticsSummary,
    instagram_summary: VideoAnalyticsSummary,
) -> ComparisonSummary:
    """
    Produce a head-to-head comparison between a YouTube video and an
    Instagram Reel analytics summary.

    Compares: engagement rate, views, likes, comments, total interactions.

    Args:
        youtube_summary:   VideoAnalyticsSummary from from_youtube().
        instagram_summary: VideoAnalyticsSummary from from_instagram().

    Returns:
        ComparisonSummary with metric-by-metric results and narrative notes.
    """
    yt  = youtube_summary.metrics
    ig  = instagram_summary.metrics

    def _winner(yt_val: Optional[float], ig_val: Optional[float]) -> Optional[str]:
        """Determine which platform wins on a given numeric metric."""
        if yt_val is None and ig_val is None:
            return None
        if yt_val is None:
            return "instagram"
        if ig_val is None:
            return "youtube"
        if math.isclose(yt_val, ig_val, rel_tol=0.01):
            return "tied"
        return "youtube" if yt_val > ig_val else "instagram"

    # ── Head-to-head comparisons ──────────────────────────────────────────────
    head_to_head: list[ComparisonResult] = [

        ComparisonResult(
            metric="Engagement Rate (%)",
            youtube_value=yt.engagement_rate_label,
            instagram_value=ig.engagement_rate_label,
            winner=_winner(yt.engagement_rate, ig.engagement_rate),
            delta=(
                f"{abs((yt.engagement_rate or 0) - (ig.engagement_rate or 0)):.2f}%"
                if yt.engagement_rate is not None and ig.engagement_rate is not None
                else "N/A"
            ),
        ),

        ComparisonResult(
            metric="Total Views",
            youtube_value=yt.views_formatted,
            instagram_value=ig.views_formatted,
            winner=_winner(
                float(yt.effective_views),
                float(ig.effective_views),
            ),
            delta=_format_count(abs(yt.effective_views - ig.effective_views)),
        ),

        ComparisonResult(
            metric="Total Likes",
            youtube_value=yt.likes_formatted,
            instagram_value=ig.likes_formatted,
            winner=_winner(
                float(yt.effective_likes),
                float(ig.effective_likes),
            ),
            delta=_format_count(abs(yt.effective_likes - ig.effective_likes)),
        ),

        ComparisonResult(
            metric="Total Comments",
            youtube_value=yt.comments_formatted,
            instagram_value=ig.comments_formatted,
            winner=_winner(
                float(yt.effective_comments),
                float(ig.effective_comments),
            ),
            delta=_format_count(abs(yt.effective_comments - ig.effective_comments)),
        ),

        ComparisonResult(
            metric="Total Interactions (Likes + Comments)",
            youtube_value=_format_count(yt.total_interactions),
            instagram_value=_format_count(ig.total_interactions),
            winner=_winner(
                float(yt.total_interactions),
                float(ig.total_interactions),
            ),
            delta=_format_count(abs(yt.total_interactions - ig.total_interactions)),
        ),

        ComparisonResult(
            metric="Like Rate (%)",
            youtube_value=f"{yt.like_rate:.2f}%" if yt.like_rate is not None else "N/A",
            instagram_value=f"{ig.like_rate:.2f}%" if ig.like_rate is not None else "N/A",
            winner=_winner(yt.like_rate, ig.like_rate),
            delta=(
                f"{abs((yt.like_rate or 0) - (ig.like_rate or 0)):.2f}%"
                if yt.like_rate is not None and ig.like_rate is not None else "N/A"
            ),
        ),

        ComparisonResult(
            metric="Comment Rate (%)",
            youtube_value=f"{yt.comment_rate:.2f}%" if yt.comment_rate is not None else "N/A",
            instagram_value=f"{ig.comment_rate:.2f}%" if ig.comment_rate is not None else "N/A",
            winner=_winner(yt.comment_rate, ig.comment_rate),
            delta=(
                f"{abs((yt.comment_rate or 0) - (ig.comment_rate or 0)):.2f}%"
                if yt.comment_rate is not None and ig.comment_rate is not None else "N/A"
            ),
        ),

        ComparisonResult(
            metric="Engagement Level",
            youtube_value=yt.engagement_level.value,
            instagram_value=ig.engagement_level.value,
            winner=None,    # qualitative — no numeric winner
            delta=None,
        ),
    ]

    # ── Overall winner (by engagement rate) ───────────────────────────────────
    overall_winner = _winner(yt.engagement_rate, ig.engagement_rate)

    # ── Narrative notes ───────────────────────────────────────────────────────
    notes: list[str] = []

    # Platform context note
    notes.append(
        "Note: YouTube and Instagram use different content formats and audience "
        "behaviours. Engagement benchmarks differ by platform — direct comparison "
        "should be interpreted with this in mind."
    )

    # Winner note
    if overall_winner == "youtube":
        notes.append(
            f"YouTube outperforms Instagram on engagement rate "
            f"({yt.engagement_rate_label} vs {ig.engagement_rate_label})."
        )
    elif overall_winner == "instagram":
        notes.append(
            f"Instagram outperforms YouTube on engagement rate "
            f"({ig.engagement_rate_label} vs {yt.engagement_rate_label})."
        )
    elif overall_winner == "tied":
        notes.append(
            "Both platforms show similar engagement rates — content performs "
            "comparably across audiences."
        )
    else:
        notes.append(
            "Overall winner cannot be determined due to insufficient data "
            "from one or both platforms."
        )

    # Reliability notes
    if not yt.is_engagement_reliable:
        notes.append(
            "⚠ YouTube engagement data is incomplete — some metrics were not "
            "reported and are treated as 0. True engagement may be higher."
        )
    if not ig.is_engagement_reliable:
        notes.append(
            "⚠ Instagram hides likes/comments for some accounts. "
            "Reported metrics may significantly understate actual engagement."
        )

    # Volume discrepancy note
    view_ratio = (
        yt.effective_views / ig.effective_views
        if ig.effective_views > 0 else None
    )
    if view_ratio and view_ratio > 10:
        notes.append(
            f"YouTube has {view_ratio:.0f}× more views — it may benefit from "
            "algorithmic recommendation while the Instagram Reel has a smaller audience."
        )
    elif view_ratio and view_ratio < 0.1:
        notes.append(
            f"Instagram has {1/view_ratio:.0f}× more views — the Reel may have "
            "benefited from stronger algorithmic distribution."
        )

    logger.info(
        "compare() complete | overall_winner=%s | yt_rate=%s | ig_rate=%s",
        overall_winner,
        yt.engagement_rate_label,
        ig.engagement_rate_label,
    )

    return ComparisonSummary(
        youtube=youtube_summary,
        instagram=instagram_summary,
        overall_winner=overall_winner,
        head_to_head=head_to_head,
        comparison_notes=notes,
        computed_at=_utc_now(),
    )
