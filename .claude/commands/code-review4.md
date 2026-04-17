---
description: Review pending changes on the current branch
---

Perform a code review of the pending changes on the current branch.

Steps:

1. Run `git status` and `git diff` (staged and unstaged) to identify the changes.
2. Compare the current branch against `main` with `git diff main...HEAD` to see the full scope.
3. Review the changes against the project conventions defined in `CLAUDE.md`:
    - SOLID design principles
    - German-language UI text
    - 4-space indentation
    - camelCase JS, kebab-case data attributes
    - Single-file architecture (`index.html`), no build step
    - Firebase Firestore + Auth patterns
    - `Intl.NumberFormat` with `de-DE` locale for currency
4. Flag any of the following:
    - Bugs or logic errors
    - Security issues (XSS, injection, exposed secrets beyond client-side Firebase keys)
    - Violations of project conventions
    - Missing error handling at system boundaries
    - Unnecessary abstractions or premature optimization
    - Dead code or unused variables
    - Accessibility or UX regressions
5. Summarize findings as a prioritized list: **Blocking**, **Should fix**, **Nit**. For each finding include file path and line number.
6. Do not make changes — report only.
