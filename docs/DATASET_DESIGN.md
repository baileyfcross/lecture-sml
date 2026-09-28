# Dataset Design

`DatasetExample` is a structured Pydantic record, not arbitrary prompt/response JSON. It includes an id, split, task, course, level, instruction, optional context and source material, response, pedagogy tags, provenance, quality metadata, version, and extensible metadata.

## Provenance and quality

Every example preserves its origin and review state. Quality tiers are:

- **A:** instructor-created and explicitly approved.
- **B:** instructor-created but not recently reviewed.
- **C:** AI-generated and human-reviewed.
- **D:** synthetic and not yet human-reviewed.

The tier is metadata, not a substitute for actual review. Factual and pedagogy review fields can be recorded independently.

## Splits

Training, validation, and evaluation are explicit enum values. Evaluation prompts and examples must remain separate and must never be automatically copied into training data. Leakage would make baseline and fine-tuned comparisons misleading.

## What not to train

Do not turn the entire Obsidian vault into fine-tuning data. Fine-tuning should teach recurring instructional decisions and style; source knowledge should primarily stay in retrieval, where it can be updated, cited, and access-controlled. High-quality reviewed examples are more useful than uncontrolled bulk data.
