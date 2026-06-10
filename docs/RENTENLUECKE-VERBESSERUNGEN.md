# Rentenlücke – Verbesserungsvorschläge

Ziel: den Rentenlücken-Rechner zu einem **generischen Werkzeug** ausbauen, das beliebige Vorsorge-Situationen abbildet und Daten nahtlos aus der Vermögensübersicht übernimmt.

## Ist-Zustand (Kurzanalyse)

Der Rechner ist fachlich bereits solide: Besteuerungsanteil nach Eintrittsjahr, Einkommensteuer-Grundtarif 2025 inkl. Splitting, Soli mit Milderungszone, KV/PV-Sätze für Rentner, Betriebsrenten-Freigrenze, jahresgenaue Projektion mit Rentensteigerung und Inflation. Die Übernahme aus der Vermögensübersicht existiert punktuell (zwei „Übernehmen"-Buttons: Gesamtvermögen ohne bAV, bAV → Betriebsrente).

Hauptschwächen für ein generisches Tool:

1. Steuer-/Abgabenparameter sind **hartkodiert auf 2025**.
2. Nur zwei Einkommensquellen (gesetzliche Rente, Betriebsrente) – keine Riester/Rürup/private Rentenversicherung/Mieteinnahmen.
3. Die Datenübernahme ist eine **einmalige Kopie** per Button, keine Live-Verknüpfung, und „alles oder nichts" (Gesamtvermögen minus bAV).
4. Eingaben liegen nur im `localStorage` – auf einem zweiten Gerät ist der Rechner leer, obwohl die Vermögensdaten in der Cloud liegen.
5. Entnahmen werden steuerfrei modelliert (keine Abgeltungsteuer auf Kursgewinne).

## A. Datenübernahme aus der Vermögensübersicht (höchste Priorität)

### A1. Konfigurierbare Konto-Zuordnung statt „alles minus bAV"
Analog zur bestehenden bAV-Einstellung eine Zuordnung je Konto/Depot:

- **Altersvorsorge-Kapital** (fließt in „Bestehendes Vermögen"),
- **Betriebsrente/bAV** (fließt annuitisiert in die Betriebsrente – existiert bereits),
- **Nicht für die Rente** (z. B. Girokonto/Notgroschen – wird ignoriert),
- optional je Konto eine **eigene Renditeerwartung** (Tagesgeld 2 % vs. Depot 7 %).

Umsetzung: das `Settings`-Objekt um `rentenAccounts`/`excludedAccounts` erweitern; `uebernahmeVermoegen2()` summiert dann nur die zugeordneten Konten. UI: gleiche Checkbox-Matrix wie im bAV-Dialog.

### A2. Live-Verknüpfung statt Kopie
Checkbox „Automatisch aus Vermögensübersicht übernehmen": bei jedem Tab-Wechsel/Berechnen wird der jüngste Eintrag neu eingelesen (Felder werden read-only mit Hinweis „verknüpft ✓"). Manuelles Überschreiben löst die Verknüpfung. Der Mechanismus existiert in Ansätzen schon (`autoFillBetriebsrente()` beim Tab-Wechsel) und müsste nur auf das Vermögensfeld ausgeweitet werden.

### A3. Verzinsung aus der eigenen Historie vorschlagen
Aus den vorhandenen Einträgen lässt sich die tatsächliche jährliche Wachstumsrate (CAGR) des Depots berechnen – als Vorschlagswert neben dem Feld „Verzinsung Vermögen" („Deine historische Rendite: 6,8 % p.a. – übernehmen?"). Achtung: Einzahlungen verzerren die reine Kurs-Rendite; als Näherung trotzdem wertvoll, mit entsprechendem Hinweis.

### A4. Sparrate mit Ist-Daten abgleichen
Die berechnete „benötigte monatliche Sparrate" mit der tatsächlichen durchschnittlichen monatlichen Vermögensveränderung aus dem Verlauf vergleichen: „Du sparst aktuell ø 850 €/Monat – benötigt: 1.100 €/Monat" – das macht das Ergebnis unmittelbar handlungsrelevant.

## B. Generischer Rechner

### B1. Steuer-/Abgabenparameter als Jahres-Tabelle
Alle Konstanten (Grundfreibetrag, Tarifformel-Koeffizienten, Soli-Freigrenze, KV-/PV-Sätze, bAV-Freigrenze) in ein `TAX_PARAMS`-Objekt pro Jahr auslagern (analog zum `BANKS`-Registry-Muster, OCP). Vorteile: jährliche Pflege ist ein Ein-Zeilen-Update, Tests können gegen ein fixes Jahr laufen, und perspektivisch kann der Tarif des Renteneintrittsjahres geschätzt werden (z. B. Grundfreibetrag mit Inflation fortschreiben statt 2025 einzufrieren – derzeit wird die Steuer in 30 Jahren systematisch überschätzt).

### B2. Beliebige Einkommensquellen (Renten-Registry)
Statt fester Felder „gesetzliche Rente" + „Betriebsrente" eine Liste von Einkommensquellen mit jeweils: Bruttobetrag/Monat, Beginn-Alter, jährliche Steigerung, Steuerart (Besteuerungsanteil / voll steuerpflichtig / Ertragsanteil / steuerfrei) und KV/PV-Pflicht. Damit lassen sich Riester, Rürup, private Rentenversicherung, Mieteinnahmen oder die Rente des Partners abbilden. `berechneNetto()` iteriert dann über die Quellen statt über zwei Parameter – die bestehende Funktion bleibt als Spezialfall erhalten.

### B3. Entnahme-Besteuerung (Abgeltungsteuer)
Depot-Entnahmen enthalten Kursgewinne, auf die ~26,375 % Abgeltungsteuer (zzgl. Sparer-Pauschbetrag, Teilfreistellung 30 % bei Aktienfonds) anfällt. Vorschlag: Eingabe „Kostenbasis-Anteil" oder vereinfachter „effektiver Steuersatz auf Entnahmen" – sonst ist die ausgewiesene Lücke zu optimistisch.

### B4. KV-Status konfigurierbar
Aktuell wird die Pflichtversicherung in der KVdR angenommen. Optionen: privat versichert (fester Monatsbeitrag) oder freiwillig gesetzlich versichert (Beiträge auch auf Kapitaleinkünfte/Betriebsrenten in voller Höhe).

### B5. Getrennte Renditen für Anspar- und Entnahmephase
Ein Zinssatz für beide Phasen ist unrealistisch (Umschichtung in sicherere Anlagen im Alter). Zwei Felder: „Rendite bis Rente" / „Rendite im Ruhestand".

### B6. bAV-Annuitisierung mit Verzinsung
`annuitizeMonthly()` verteilt das bAV-Kapital linear ohne Verzinsung; mit Restkapital-Verzinsung (Annuitätenformel) wäre die Betriebsrente realistischer. Außerdem: Wahl Einmalauszahlung (nachgelagerte Besteuerung, §34 EStG Fünftelregelung) vs. Verrentung.

### B7. Szenarien & Sensitivität
Drei Voreinstellungen (pessimistisch/realistisch/optimistisch) oder ±1 %-Schieberegler für Rendite/Inflation, dargestellt als Band im Diagramm. Gerade bei 30-Jahres-Projektionen ist die Bandbreite die eigentliche Information.

## C. Technik / UX

| Vorschlag | Nutzen |
|---|---|
| **C1.** Rentenlücken-Eingaben + bAV-Zuordnung in Firestore statt `localStorage` speichern | Geräteübergreifend konsistent; localStorage bleibt als Offline-Fallback |
| **C2.** Berechnung automatisch bei Eingabe-Änderung (debounced) statt „Berechnen"-Button | Sofortiges Feedback, weniger Klicks |
| **C3.** Ergebnis-Export (CSV/Druckansicht) analog zum bestehenden CSV-Export | Unterlage für Beratungsgespräche |
| **C4.** Mehrere benannte Profile (z. B. „Ich", „Partnerin", „gemeinsam") | Haushaltsbetrachtung |
| **C5.** Validierungshinweise inline statt `alert()` | Bessere UX, zeigt direkt das fehlerhafte Feld |

## Empfohlene Reihenfolge

1. **A1 + A2** (konfigurierbare, live Datenübernahme) – größter Nutzen, baut auf vorhandenem Settings-/bAV-Muster auf.
2. **B1** (Parameter-Tabelle pro Jahr) – Voraussetzung für alles Weitere, reiner Refactor.
3. **C1** (Firestore-Persistenz) – behebt die Geräte-Inkonsistenz.
4. **B2** (generische Einkommensquellen) – macht das Tool wirklich generisch.
5. **B3/B5** (Entnahme-Steuer, getrennte Renditen) – Genauigkeit.
6. Rest nach Bedarf.
