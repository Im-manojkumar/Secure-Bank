"""Fraud cases: opening them for suspicious predictions and recording reviews."""

from models import CASE_OUTCOMES, CASE_STATUSES


class CaseUpdateError(ValueError):
    pass


def apply_case_update(case, payload, user):
    """Update status, notes and outcome from a JSON payload. Raises CaseUpdateError."""
    if not isinstance(payload, dict) or not payload:
        raise CaseUpdateError("Send at least one of: status, review_notes, outcome")

    unknown = set(payload) - {"status", "review_notes", "outcome"}
    if unknown:
        raise CaseUpdateError(f"Unknown fields: {', '.join(sorted(unknown))}")

    status = payload.get("status", case.status)
    if status not in CASE_STATUSES:
        raise CaseUpdateError(f"status must be one of: {', '.join(CASE_STATUSES)}")

    outcome = payload.get("outcome", case.outcome)
    if outcome is not None and outcome not in CASE_OUTCOMES:
        raise CaseUpdateError(f"outcome must be one of: {', '.join(CASE_OUTCOMES)}")
    if status == "Resolved" and outcome is None:
        raise CaseUpdateError("Set an outcome when resolving a case")
    if status != "Resolved" and "outcome" in payload and outcome is not None:
        raise CaseUpdateError("An outcome can only be set when the case is Resolved")
    if status != "Resolved":
        # Reopening a case clears its earlier finding.
        outcome = None

    if "review_notes" in payload:
        notes = payload["review_notes"]
        if not isinstance(notes, str) or len(notes) > 5000:
            raise CaseUpdateError("review_notes must be text of at most 5000 characters")
        case.review_notes = notes

    case.status = status
    case.outcome = outcome
    case.reviewed_by = user.id
    return case
