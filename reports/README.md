# Reports

- **`presentation/`** — the Slidev deck (source). Build locally:
  ```bash
  cd reports/presentation
  npm install            # restores node_modules (git-ignored)
  npm run dev            # live preview
  npm run export         # → PDF
  ```
  Covers data-imbalance handling, preprocessing/augmentation, model justification, and
  explainability. The explainability section is a known stub.

- **`ERRATA.md`** — post-submission corrections. **Read this alongside the slides** — two slide
  figures present numbers that the post-hoc verification corrected.
