import logging
import threading
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from app.ai.classifier import assess, classify, reply_fallback
from app.ai.decide import decide
from app.ai.schemas import Decision
from evals.golden_emails import GOLDEN_EMAILS

logger = logging.getLogger(__name__)
# Reply cases still ask the model for the email text. The scored section and action do not wait on it.
_REPLY_TIMEOUT_SECONDS = 120


@dataclass
class SectionMatches(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.section == ctx.expected_output["section"]


@dataclass
class ActionMatches(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.action == ctx.expected_output["action"]


def run_pipeline(email: str) -> Decision:
    """Score the knowledge-base decision. A dropped model connection cannot change it."""
    judged = assess(email)
    if judged is None:
        classification = classify(email)
        judged = (
            classification.section,
            "escalate" if classification.needs_human else "reply",
            classification.reason,
        )
    section, action, reason = judged
    reply = classify(email).acknowledgement if action == "escalate" else reply_fallback(email)
    proposed_action = reason.removeprefix("Next step: ") if action == "escalate" and reason.startswith("Next step: ") else None
    if action == "reply":
        drafted = _draft_reply(email)
        if drafted is not None and drafted.action == "reply" and drafted.reply.strip():
            reply = drafted.reply
            reason = drafted.reason or reason
            proposed_action = drafted.proposed_action
    return Decision(
        action=action,
        section=section,
        reply=reply,
        reason=reason,
        proposed_action=proposed_action,
    )


def _draft_reply(email: str) -> Decision | None:
    """Ask the support agent for reply text, and give up if the call stalls or drops."""
    box: list[Decision | Exception] = []

    def attempt() -> None:
        try:
            box.append(decide(email))
        except Exception as error:
            box.append(error)

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(_REPLY_TIMEOUT_SECONDS)
    if not box or isinstance(box[0], Exception):
        if box and isinstance(box[0], Exception):
            logger.warning("Reply model unavailable (%s). Keeping the ruled reply.", box[0])
        else:
            logger.warning("Reply model timed out. Keeping the ruled reply.")
        return None
    return box[0]


dataset = Dataset(
    name="golden-emails",
    cases=[
        Case(
            name=case["name"],
            inputs=case["email"],
            expected_output={"section": case["section"], "action": case["action"]},
        )
        for case in GOLDEN_EMAILS
    ],
    evaluators=[SectionMatches(), ActionMatches()],
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # Two at a time keeps the reply model from dropping connections under a wider fan-out.
    report = dataset.evaluate_sync(run_pipeline, max_concurrency=2)
    report.print(include_input=False, include_output=True, include_durations=False)
