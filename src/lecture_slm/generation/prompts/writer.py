"""Final-artifact writer prompt construction."""

from lecture_slm.generation.models import GenerationRequest, TeachingPlanBase
from lecture_slm.generation.prompts.base import PromptPackage, json_block, request_blocks

WRITER_PROMPT_VERSION = "writer-v7"

WRITER_SYSTEM_PROMPT = (
    "You are the artifact writer for Lecture SLM. Follow the user's authoritative request and "
    "supplied teaching plan. Use your output budget to produce the requested educational artifact "
    "rather than explaining your planning process. Do not unnecessarily re-plan. Respect course "
    "level, pedagogy requirements, requested format, supplied sources, and prior-course context. "
    "Do not add unsupported source claims."
)

SOURCE_GROUNDING_INSTRUCTIONS = (
    "Because source material is supplied, treat it as authoritative for factual content. "
    "Base definitions, historical claims, applications, technical claims, relationships between "
    "concepts, and domain-specific assertions on these sources. Do not add factual claims from "
    "pretrained knowledge unless the supplied sources support them. If the sources do not support "
    "a factual point, omit it, keep the explanation narrower, or briefly note the gap when useful. "
    "This grounding also applies to introductory framing, motivation, conclusions, significance, "
    "importance, foundational status, historical purpose, influence, prevalence, and claims "
    "connecting the concept to a broader field. Do not call a concept foundational, central, "
    "widely used, important to a field, influential, transformative, a bridge between fields, or "
    "the basis of another discipline, and do not claim it was developed for a particular purpose, "
    "unless the sources support that characterization. Do not turn a supported technical "
    "relationship into a broader claim about importance, influence, or historical significance. "
    "Pedagogical motivation is allowed: explain how a source-supported distinction helps the "
    "learner understand the topic. Historical claims such as why a concept was developed or "
    "became important require source support. For example, if the sources explain quantifiers, "
    "it is supported to say, 'Predicate logic allows statements about all or some members of a "
    "domain.' Claims that it is 'foundational to computer science,' 'bridges logic and "
    "computation,' 'widely used in AI,' or was 'developed primarily to solve' a problem are "
    "unsupported unless the sources say so. Do not assert 'the primary motivation for' or 'the "
    "main reason' a concept exists, or conclude that it 'bridges the gap between logical "
    "reasoning and computational processes,' unless the sources explicitly support that claim. "
    "In particular, do not infer a concept's historical cause from what it can express. Avoid "
    "claims such as 'the primary motivation for moving from propositional logic' or 'the main "
    "reason it was developed' unless stated in the sources. Do not inflate a supported capability "
    "into a claim that it is 'essential,' or conclude that it 'bridges the gap between abstract "
    "logical reasoning and computational procedures,' without explicit support. Keep motivation "
    "and conclusions to the specific distinctions and relationships the sources establish. "
    "Source grounding takes precedence over the teaching plan. If the plan requests any factual "
    "section, claim, example, application, historical point, technical relationship, or domain use "
    "not supported by the supplied sources, omit or narrow that part of the plan. Do not fill "
    "unsupported plan requirements using pretrained knowledge. Follow this priority: supplied "
    "source factual scope, user request and course/pedagogy constraints, teaching plan, then "
    "pedagogical invention. Preserve supported pedagogical elements requested by the plan, "
    "including original examples and checks for understanding. Frame examples and questions "
    "around source-supported concepts; they need not appear verbatim in the sources. "
    "Keep the prose natural; do not repeatedly attribute statements to the sources. You may create "
    "original pedagogical examples, hypothetical examples, analogies, transitions, "
    "worked examples, illustrative names and numbers, practice and check-for-understanding "
    "questions, "
    "organization, "
    "and simplified explanations to teach supported facts. These teaching aids must not present "
    "unsupported real-world or domain-specific claims as facts."
)


def build_writer_prompt(
    request: GenerationRequest,
    plan: TeachingPlanBase | None,
) -> PromptPackage:
    blocks = request_blocks(request)
    if plan is not None:
        blocks.insert(1, json_block("Teaching plan to follow", plan))
    else:
        blocks.append(
            "## Writer task\nCreate the requested artifact directly. "
            "Do not provide a planning discussion."
        )
    return PromptPackage(
        version=WRITER_PROMPT_VERSION,
        system_message=(
            f"{WRITER_SYSTEM_PROMPT}\n\n{SOURCE_GROUNDING_INSTRUCTIONS}"
            if request.source_material
            else WRITER_SYSTEM_PROMPT
        ),
        user_message="\n\n".join(blocks),
    )
