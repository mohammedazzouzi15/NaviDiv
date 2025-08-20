"""Utility functions for monitoring scorer execution time and progress.

This module provides functionality to estimate runtime for different scorers
and monitor their execution progress in real-time.
"""

import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Constants for time formatting
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
MAX_DISPLAY_SCORERS = 5
LONG_JOB_THRESHOLD = 300  # 5 minutes
CONFIRMATION_THRESHOLD = 180  # 3 minutes


def estimate_scorer_runtime(file_path: Path) -> dict[str, float]:
    """Estimate runtime for each scorer based on dataset size and complexity.
    
    Args:
        file_path: Path to the CSV file containing molecules
        
    Returns:
        Dictionary mapping scorer names to estimated runtime in seconds
    """
    try:
        molecules_data = pd.read_csv(file_path)
        num_molecules = len(molecules_data)
        step_col = "step"
        if step_col in molecules_data.columns:
            num_steps = len(molecules_data[step_col].unique())
        else:
            num_steps = 1
        
        # Base times (seconds per molecule) derived from empirical observations
        base_times = {
            "Frequency": 0.01,
            "tSNE": 1.2,
            "Similarity": 0.5,
            "Activity": 0.8,
            "UMap": 0.7,
            "Ngram": 0.3,
            "Scaffold": 0.4,
            "Cluster": 1.0,
            "Original": 0.02,
            "RingScorer": 0.2,
            "FG": 0.15,
            "MPO": 0.3,
        }
        
        # Calculate estimates based on dataset complexity
        # Multi-step datasets are more complex
        complexity_factor = 1 + (num_steps - 1) * 0.2
        size_factor = max(1, num_molecules / 1000)  # Scale with dataset size
        
        estimates = {
            scorer: base_time * num_molecules * complexity_factor * size_factor
            for scorer, base_time in base_times.items()
        }
        
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        st.warning(f"Could not estimate runtime: {e}")
        scorer_names = [
            "Frequency", "tSNE", "Similarity", "Activity", "UMap",
            "Ngram", "Scaffold", "Cluster", "Original", "RingScorer",
            "FG", "MPO"
        ]
        estimates = dict.fromkeys(scorer_names, 30.0)
    
    return estimates


def format_time_estimate(seconds: float) -> str:
    """Format time estimate in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < SECONDS_PER_MINUTE:
        return f"{int(seconds)}s"
    if seconds < SECONDS_PER_HOUR:
        minutes = int(seconds // SECONDS_PER_MINUTE)
        remaining_seconds = int(seconds % SECONDS_PER_MINUTE)
        return f"{minutes}m {remaining_seconds}s"
    
    hours = int(seconds // SECONDS_PER_HOUR)
    minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
    return f"{hours}h {minutes}m"


def initialize_progress_monitoring(selected_scorers: list[str]) -> None:
    """Initialize progress monitoring for scorer execution.
    
    Args:
        selected_scorers: List of scorers that will be executed
    """
    if "scorer_progress" not in st.session_state:
        st.session_state.scorer_progress = {
            "completed": 0,
            "current_scorer": "Starting...",
            "is_running": False
        }
    
    st.session_state.scorer_progress.update({
        "total": len(selected_scorers),
        "scorers": selected_scorers,
        "completed": 0,
        "current_scorer": "Starting...",
        "is_running": True
    })


def monitor_scorer_progress(
    outputs_folder: Path, selected_scorers: list[str]
) -> None:
    """Monitor scorer execution progress by checking output files.

    Args:
        outputs_folder: Path to the outputs folder containing scorer results
        selected_scorers: List of scorers being executed
    """
    def progress_monitor() -> None:
        while st.session_state.scorer_progress["is_running"]:
            # Count completed scorers by checking for output files
            completed_count = 0
            current_scorer = "Starting..."

            for scorer in selected_scorers:
                scorer_file = outputs_folder / f"{scorer}.csv"
                if scorer_file.exists():
                    completed_count += 1
                elif completed_count == st.session_state.scorer_progress.get(
                    "completed", 0
                ):
                    current_scorer = scorer
                    break

            st.session_state.scorer_progress["completed"] = completed_count
            st.session_state.scorer_progress["current_scorer"] = current_scorer

            # Check if all scorers are complete
            if (completed_count >= len(selected_scorers)
                and st.session_state.scorer_progress["is_running"]):
                st.session_state.scorer_progress["is_running"] = False
                break

            time.sleep(2)  # Check every 2 seconds

    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()


def create_runtime_estimates_display(time_estimates: dict[str, float]) -> None:
    """Create a display for runtime estimates in the sidebar.
    
    Args:
        time_estimates: Dictionary of scorer names and estimated times
    """
    total_estimated_time = sum(time_estimates.values())
    
    with st.sidebar.container(), st.expander(
        "⏱️ **Runtime Estimates**", expanded=False
    ):
        st.markdown("**Estimated processing time by scorer:**")

        # Sort by estimated time (longest first)
        sorted_estimates = sorted(
            time_estimates.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for scorer, est_time in sorted_estimates[:MAX_DISPLAY_SCORERS]:
            st.markdown(f"• **{scorer}**: {format_time_estimate(est_time)}")

        if len(sorted_estimates) > MAX_DISPLAY_SCORERS:
            remaining_count = len(sorted_estimates) - MAX_DISPLAY_SCORERS
            st.markdown(f"• *...and {remaining_count} more*")

        st.markdown("---")
        total_time_str = format_time_estimate(total_estimated_time)
        st.markdown(f"**Total Estimated Time: {total_time_str}**")

        # Add warning for long runs
        if total_estimated_time > LONG_JOB_THRESHOLD:
            st.warning(
                "⚠️ This analysis may take a while. Consider running "
                "on a smaller subset first."
            )


def get_confirmation_for_long_jobs(total_estimated_time: float) -> bool:
    """Create confirmation checkbox for long-running analyses.
    
    Args:
        total_estimated_time: Total estimated time in seconds
        
    Returns:
        True if user confirmed or if confirmation not needed
    """
    show_confirmation = total_estimated_time > CONFIRMATION_THRESHOLD
    
    if not show_confirmation:
        return True
        
    st.sidebar.markdown("---")
    time_formatted = format_time_estimate(total_estimated_time)
    time_text = f"I understand this will take ~{time_formatted}"
    return st.sidebar.checkbox(time_text, key="confirm_long_run")
