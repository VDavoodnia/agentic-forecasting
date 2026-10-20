# Presentations (PPTX)

> **Status: PARKED — 2026-10-20 (Vandad's call).** The deck of record is
> the HTML paper report in [`docs/slides/`](../slides/). These PPTX
> decks are kept for provenance and are **not** updated as results land.

- `TD-One_Final_Presentation.pptx.b64` — the v1 showcase deck
  (2026-10-19), built by `build_deck.py`.
- `TD-One_Phase2_Update.pptx.b64` — the Phase-2 update deck
  (2026-10-20), built by `build_deck2.py`.
- The `.pptx` files are stored base64-encoded because the git push
  channel is text-only. Decode with:
  `base64 -d TD-One_Final_Presentation.pptx.b64 > TD-One_Final_Presentation.pptx`
  (sha256 of the decoded decks: Final
  `a283862c27f4b1834a595af8e89a61156674349415a6a9880a5fd0fac181a282`,
  Phase2 `25f9aaae40d2ad9a04f34515c85a0b87f155a8e6805f600b21a921f66a4917f0`.)
- `build_deck.py` / `build_deck2.py` — the python-pptx builders for the
  two decks (run with a throwaway venv that has python-pptx installed;
  it is not a repo dependency).

Speaker notes for the v1 deck live at
`~/workspace/your_files/TD-One_Showcase_Talk_Notes.md` (outside the
repo); the audited final showcase deck (Mehrazin's 12-slide version)
and its talking points are likewise kept with the project files, and
every number in them is carried in `docs/FINAL_RESULTS.md`.
