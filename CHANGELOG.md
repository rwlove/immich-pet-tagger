# Changelog

## v0.3.0

### Features
- **Import from Immich**: import a pet directly from an existing Immich person. The tool fetches up to 20 evenly distributed single-pet photos as refs automatically, skipping photos where multiple named people appear.
- **Find candidates for "not my pets"**: new button searches across all pets simultaneously, merges results, scores them by pet-likeness using the classifier, and shows the top 60 in the main grid for bulk review.
- **Tool-only delete**: when deleting a pet, a third option lets you remove it from the tool only (keeping the Immich person and all tagged photos intact). Assets are never deleted in either case.
- **Clear all refs / Clear all negatives**: bulk-clear buttons with confirmation, local only, no Immich changes.

### Fixes
- "Find similar photos" now skips the CLIP classifier stage when the pet has no refs, returning text search results immediately instead of waiting for model inference.
- Auto-select newly created pet after adding it.
- Stay on the edited pet after saving edits (was jumping to the first pet).
- Import no longer crashes on photos where Immich returns a null person in the faces list.

### Internal
- `app/` directory volume-mounted in docker-compose: Python, HTML, CSS, and JS changes apply after `docker compose restart` with no rebuild needed.

---

## v0.2.0

### Features
- **Similar photo suggestions**: new "Find similar photos" button runs a two-stage pipeline — Immich smart search pre-filters by pet description, then the classifier ranks candidates by pet class probability. Replaces the manual search bar.
- **Pet description field**: each pet now has a short description (2-4 keywords, e.g. "orange tabby cat") used as the CLIP text query for suggestions.
- **Shift+click range selection**: hold Shift and click a second photo to select the full range in the grid.
- **Tagged photos panel**: view photos already tagged for the active pet; remove a tag or mark as "not my pets" in bulk.
- **Scan status panel**: shows last scan time, badge (running/idle/error/never), and per-pet stats (tagged, skipped, errors) after each poll cycle.
### UI
- Refs and "Not my pets" grids: 3-column layout, equal height, independently scrollable.
- "Not my pets" panel (formerly Negatives): unified card style with refs, clickable thumbnails linking to Immich.
- "Find similar photos" button moved to the top of the right panel as the primary action.
- Scan and last scan controls moved to the bottom of the sidebar.
- Removed the manual search bar.

### Internal
- Negative samples subsampled to 3x pet refs in the classifier to keep class balance without discouraging large negative sets.
- Static files split into `style.css` and `app.js` (was a single `index.html`).
- Backend refactored: `data.py` for file I/O, `immich.py` for HTTP helpers (replaces `immich_apis.py`).

---

## v0.1.0

Initial release. Core tagging loop, pet enrollment UI, ref/negative management, logistic regression classifier on CLIP embeddings.
