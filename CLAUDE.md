# CLAUDE.md

## Project Overview

Finanz-Tracker is a German-language personal finance web app for tracking wealth across multiple banks/brokers and calculating pension gaps (Rentenlücke). It is a single-file HTML application (`index.html`) with no build system.

## Tech Stack

- **Frontend**: Vanilla JavaScript, HTML, CSS (no framework)
- **Charting**: Chart.js 4.4.0 (CDN)
- **Backend**: Firebase Firestore (realtime database) + Firebase Auth (Google OAuth)
- **Fonts**: Google Fonts (Playfair Display, Work Sans)
- **Build**: None — single static HTML file, all code inline

## Running the App

Open `index.html` in a browser or serve via any static HTTP server. No build step required.

## Project Structure

Everything lives in `index.html` (~1400 lines):
- `<style>` block: All CSS with custom properties (`:root` variables)
- `<body>`: Two tabs — Vermögensübersicht (wealth overview) and Rentenlücke (pension gap)
- `<script>` block: All JS logic (~600+ lines) at the bottom

## Design Principles

- Always follow **SOLID** design principles:
    - **S**ingle Responsibility: Each function/module should have one reason to change
    - **O**pen/Closed: Code should be open for extension, closed for modification
    - **L**iskov Substitution: Subtypes must be substitutable for their base types
    - **I**nterface Segregation: Prefer small, focused interfaces over large ones
    - **D**ependency Inversion: Depend on abstractions, not concrete implementations

## Code Conventions

- **Language**: All UI text is in German
- **Indentation**: 4 spaces
- **Naming**: camelCase for JS variables/functions, kebab-case for data attributes
- **Bank IDs**: Short lowercase prefixes with account type suffix (e.g., `mlp_konto`, `consors_depot`, `tr_gesamt`)
- **DOM access**: Direct `getElementById` / `querySelector`, no abstraction layer
- **Event handlers**: Inline `onclick` attributes in HTML
- **Functions**: All global scope (no modules)
- **Async**: `async/await` for Firebase operations
- **Currency**: `Intl.NumberFormat` with `de-DE` locale
- **Dates**: `Date.toLocaleDateString('de-DE')`

## Key Globals

- `entries` — Array of all Firestore entries
- `currentUser` — Firebase auth user
- `wealthChart` / `rentenlueckeChart` — Chart.js instances
- `accounts` — Array of `{ id, label }` for all tracked bank accounts

## Architecture Patterns

- **Single-file**: All HTML/CSS/JS in one file, no modules or imports
- **Reactive refresh**: Data changes trigger `refreshDisplay()` which updates summary, chart, and history
- **Modals**: CSS class toggle (`active`) for show/hide
- **Tab navigation**: `switchTab(tabName)` with hash-based switching
- **State**: Firebase for persistence, `localStorage` for UI preferences (chart settings, pension form)

## Tracked Banks

MLP, Consors, Trade Republic, Volksbank, Union Investment, adidas, Commerzbank, Allianz Riester — each with account types (Konto, Depot, LTA).

## Important Notes

- Firebase config (API keys) is embedded inline — these are client-side keys, not secrets
- All financial calculations (totals, projections, pension gap) run client-side
- Chart uses dual Y-axes and a custom divider plugin for historical/forecast split
- Forecasting assumes 8% MSCI World return and 2.5% inflation
