# Finanz-Tracker – Benutzerhandbuch

Finanz-Tracker ist eine persönliche Finanz-Web-App zur Verfolgung des Vermögens über mehrere Banken/Broker hinweg und zur Berechnung der Rentenlücke. Die App besteht aus einer einzigen HTML-Datei und benötigt keine Installation.

## Inhalt

1. [Erste Schritte](#erste-schritte)
2. [Anmeldung](#anmeldung)
3. [Tab „Vermögensübersicht"](#tab-vermögensübersicht)
4. [Tab „Rentenlücke"](#tab-rentenlücke)
5. [Datenspeicherung & Datenschutz](#datenspeicherung--datenschutz)
6. [Annahmen & Grenzen](#annahmen--grenzen)

---

## Erste Schritte

Öffne `index.html` direkt im Browser oder über einen beliebigen statischen Webserver. Es ist kein Build-Schritt nötig. Benötigt wird eine Internetverbindung (Chart.js, Google Fonts und Firebase werden per CDN geladen).

## Anmeldung

Beim ersten Aufruf erscheint der Anmeldebildschirm.

- **Mit Google anmelden**: Klick auf den Button öffnet das Google-Login-Popup.
- Nach der Anmeldung werden deine Daten automatisch aus der Cloud (Firebase Firestore) geladen. Jeder Nutzer sieht ausschließlich seine eigenen Einträge.
- **Abmelden**: Über den Button neben deinem Namen im Kopfbereich.

Der Verbindungsstatus wird unter den Buttons angezeigt (gelb = verbindet, grün = verbunden, rot = Fehler).

## Tab „Vermögensübersicht"

### Verfolgte Banken und Konten

Die App verfolgt feste Konten bei folgenden Instituten: MLP, Consors, Trade Republic, Volksbank, Union Investment, adidas, Commerzbank und Allianz Riester – jeweils mit Kontoarten wie Konto, Depot oder LTA. Konten, die als betriebliche Altersvorsorge (bAV) markiert sind, tragen das Symbol 🏢.

### Neuer Eintrag

1. Klicke auf **„+ Neuer Eintrag"**.
2. Wähle das Datum (vorbelegt mit heute).
3. Trage die aktuellen Kontostände ein. Leere Felder zählen als 0 €.
4. **„Alle Werte vom letzten Eintrag übernehmen"** kopiert sämtliche Werte des jüngsten Eintrags – praktisch, wenn sich nur wenige Positionen geändert haben.
5. Der Pfeil-Button (◀) neben einem Banknamen übernimmt nur die Werte dieser Bank.
6. **Speichern** legt den Eintrag in der Cloud ab; die Anzeige aktualisiert sich sofort.

### Eintrag bearbeiten oder löschen

- **Bearbeiten**: Klicke in der Tabelle „Verlauf" auf eine Zeile. Es öffnet sich derselbe Dialog mit den gespeicherten Werten.
- **Löschen**: Button „Löschen" in der jeweiligen Zeile (mit Sicherheitsabfrage).

### Übersichtskarten

Oben werden auf Basis des **neuesten Eintrags** angezeigt:

| Karte | Bedeutung |
|---|---|
| Gesamtvermögen | Summe aller Konten und Depots |
| Gesamt Konten | Summe aller Girokonten/Verrechnungskonten |
| Gesamt Depots | Summe aller Depots und LTA |
| Betriebliche Altersvorsorge 🏢 | Summe der als bAV markierten Konten/Depots |
| Letzte Veränderung | Differenz zum vorherigen Eintrag (absolut und in %) |
| Gesamt Veränderung | Differenz zum ältesten Eintrag (absolut und in %) |

### Vermögensverlauf (Diagramm)

- Über die Checkboxen lassen sich die Linien **Gesamtvermögen**, **Konten** und **Depots** ein-/ausblenden.
- **MSCI World (8 %)**: Vergleichslinie – was aus dem Startvermögen bei 8 % p.a. geworden wäre.
- **Inflation (2,5 %)**: Kaufkrafterhalt-Linie; aktiviert zusätzlich eine **10-Jahres-Prognose** (gestrichelt, mit „Prognose ▶"-Trennlinie).
- **„+ Einzelne Banken"** blendet pro Bank weitere Linien ein (Gesamt, Konto, Depot, LTA).
- **Zeitraum**: 3M / 6M / 1J / 3J / Alles filtert Diagramm und Verlaufstabelle.
- Alle Diagramm-Einstellungen werden lokal im Browser gespeichert und beim nächsten Besuch wiederhergestellt.

### Vergleichen

**„Vergleichen"** stellt zwei beliebige Einträge gegenüber: pro Konto der Wert an beiden Stichtagen und die Differenz (grün = Zuwachs, rot = Rückgang).

### Export CSV

**„Export CSV"** lädt alle Einträge als Semikolon-getrennte CSV-Datei herunter (deutsches Zahlenformat mit Komma) – direkt in Excel/LibreOffice nutzbar.

### Einstellungen (bAV & Rentenlücke)

Unter **„⚙ Einstellungen"** gibt es zwei Zuordnungen:

**Betriebliche Altersvorsorge (bAV) 🏢** – Konten/Depots, die als bAV zählen:

- werden in der Übersichtskarte „Betriebliche Altersvorsorge" separat summiert,
- tragen überall das 🏢-Symbol,
- fließen im Rentenlücken-Rechner als **Betriebsrente** ein (statt als frei verfügbares Vermögen) – das vermeidet Doppelzählung.

**Rentenlücke: Konten ausschließen 🚫** – Konten/Depots, die bei der Vermögens-Übernahme in den Rentenlücken-Rechner ignoriert werden sollen (z. B. Notgroschen oder zweckgebundenes Geld). Das übernommene Vermögen ist dann: **Gesamtvermögen − bAV − ausgeschlossene Konten**. Ist ein Konto gleichzeitig als bAV markiert, hat bAV Vorrang.

Beide Zuordnungen werden in der Cloud gespeichert und gelten damit auf allen Geräten.

## Tab „Rentenlücke"

Der Rechner ermittelt, wie viel **netto** von deiner Rente übrig bleibt und wie groß die Lücke zu deiner Wunschrente ist – inklusive deutscher Rentenbesteuerung und Sozialabgaben.

### Eingabefelder

| Feld | Bedeutung |
|---|---|
| Aktuelles Alter / Renteneintrittsalter | Bestimmen die Jahre bis zur Rente und das Renteneintrittsjahr |
| Familienstand | Ledig (Grundtarif) oder verheiratet (Splittingtarif) |
| Anzahl Kinder | Beeinflusst den Pflegeversicherungssatz (kinderlos 4,0 %, sonst 3,4 %) |
| Gesetzl. Rente brutto bei Eintritt | Erwartete monatliche Bruttorente laut Renteninformation |
| Rentensteigerung im Ruhestand | Jährliche Anpassung der gesetzlichen Rente (z. B. 1,5 %) |
| Betriebsrente brutto 🏢 | Monatliche Betriebsrente. **„Übernehmen"** verrentet dein aktuelles bAV-Vermögen über den Entnahmezeitraum (mit Verzinsung im Ruhestand, sofern angegeben) |
| Gewünschte Netto-Monatsrente | Dein Wunsch-Nettoeinkommen im Ruhestand in **heutiger Kaufkraft** (wird automatisch mit der Inflation hochgerechnet) |
| Jährliche Inflation | Annahme für die Kaufkraftanpassung (z. B. 2,5 %) |
| Bestehendes Vermögen (ohne bAV) | Heutiges Kapital. **„Übernehmen"** holt das Gesamtvermögen abzüglich bAV und abzüglich der in den Einstellungen ausgeschlossenen Konten |
| Verzinsung bis Rente | Erwartete Rendite p.a. in der Ansparphase |
| Verzinsung im Ruhestand | Optional: Rendite in der Entnahmephase (z. B. niedriger durch Umschichtung in sicherere Anlagen). Leer = wie bis Rente |
| Steuersatz auf Entnahmen | Optional: effektiver Steuersatz auf Depot-Entnahmen (z. B. Abgeltungsteuer auf den Gewinnanteil). Die Entnahme wird entsprechend „brutto" hochgerechnet. 0 = steuerfrei |
| Entnahmezeitraum | Wie viele Jahre das Kapital reichen soll |

**Automatische Übernahme**: Mit der Checkbox „Vermögen & Betriebsrente automatisch aus der Vermögensübersicht übernehmen" werden beide Felder live mit dem jeweils neuesten Eintrag verknüpft – sie aktualisieren sich bei jedem Tab-Wechsel, bei neuen Einträgen und vor jeder Berechnung. Verknüpfte Felder sind schreibgeschützt; zum manuellen Eingeben einfach die Checkbox deaktivieren.

Alle Eingaben werden lokal **und** in der Cloud gespeichert und beim nächsten Besuch (auch auf anderen Geräten) wieder vorbelegt. Beim Wechsel auf den Tab wird die Betriebsrente automatisch aus dem bAV-Vermögen vorbefüllt, sofern das Feld noch leer ist (manuelle Eingaben werden nie überschrieben).

### Was wird berechnet?

1. **Besteuerungsanteil** der gesetzlichen Rente nach deinem Renteneintrittsjahr (gesetzliche Übergangsregelung, ab 2058 = 100 %).
2. **Sozialabgaben**: Kranken- (≈ 8,15 %) und Pflegeversicherung auf die gesetzliche Rente; auf die Betriebsrente voller KV-Satz (≈ 16,3 %) oberhalb der Freigrenze (176,75 €/Monat) plus PV.
3. **Einkommensteuer** nach Grundtarif 2025 (§ 32a EStG) bzw. Splittingtarif, plus Solidaritätszuschlag (mit Freigrenze und Milderungszone).
4. **Monatliche Rentenlücke (netto)**: Wunschrente (inflationsbereinigt auf den Renteneintritt) minus Netto-Rente.
5. **Benötigtes Kapital**: Barwert aller monatlichen Entnahmen über den Entnahmezeitraum (diskontiert mit der Ruhestand-Verzinsung), wobei die Lücke jedes Jahr neu berechnet wird (Rente steigt mit der Rentensteigerung, Bedarf mit der Inflation). Ist ein Steuersatz auf Entnahmen angegeben, wird die Entnahme entsprechend hochgerechnet.
6. **Vermögenslücke**: Benötigtes Kapital minus dein auf den Renteneintritt hochgerechnetes Vermögen (Ansparphasen-Verzinsung).
7. **Benötigte monatliche Sparrate**: Welche Sparrate (bei der Verzinsung bis Rente) die Vermögenslücke bis zum Renteneintritt schließt.

### Ergebnis-Diagramm

Das Diagramm zeigt pro Ruhestandsjahr:

- **Vermögen** (linke Achse): Kapitalverlauf inkl. Verzinsung und Entnahmen,
- **Netto-Bedarf** (rechte Achse): inflationsbereinigte Wunschrente,
- **Netto-Rente**: gesetzliche plus Betriebsrente nach Steuern/Abgaben,
- **Entnahme/Monat**: die jährlich neu berechnete Lücke.

Einzelne Linien lassen sich per Checkbox ausblenden.

## Datenspeicherung & Datenschutz

| Daten | Speicherort |
|---|---|
| Vermögenseinträge | Firebase Firestore (Cloud), pro Nutzer getrennt |
| bAV-Zuordnung, ausgeschlossene Konten, Rentenlücken-Eingaben | Firestore (Cloud, geräteübergreifend) mit `localStorage` als lokalem Cache/Offline-Fallback; beim Login gewinnt die Cloud-Version |
| Diagramm-Einstellungen (Checkboxen, Zeitraum) | `localStorage` des Browsers (nur lokal) |

Die Anmeldung erfolgt per Google OAuth; ein Passwort wird in der App nicht gespeichert. Die im Quellcode sichtbaren Firebase-Schlüssel sind clientseitige Identifikatoren, keine Geheimnisse.

## Annahmen & Grenzen

- **Steuer-/Abgabenwerte gelten für 2025** (Grundfreibetrag 12.084 €, KV-Zusatzbeitrag ≈ 0,85 %, bAV-Freigrenze 176,75 €/Monat). Die Parameter sind intern pro Jahr hinterlegt (`TAX_PARAMS_BY_YEAR`) und können bei Rechtsänderungen ergänzt werden; bis dahin gilt der jeweils jüngste bekannte Stand.
- Die MSCI-World-Vergleichslinie nimmt pauschal **8 % p.a.** an, die Kaufkraftlinie **2,5 % Inflation**.
- Kapitalertragsteuer auf Entnahmen wird nur berücksichtigt, wenn du einen **Steuersatz auf Entnahmen** angibst (vereinfachter Effektivsatz, keine Berechnung des tatsächlichen Gewinnanteils).
- Die Betriebsrente bleibt im Ruhestand konstant (keine Anpassung); die bAV-Übernahme verrentet das Kapital mit der Ruhestand-Verzinsung (ohne Angabe: linear).
- Pflegeversicherung vereinfacht: kinderlos vs. mit Kind(ern); Kinderzahl-Staffel für Kinder unter 25 wird nicht abgebildet.
- Alle Berechnungen sind Modellrechnungen und ersetzen keine Steuer- oder Rentenberatung.

## Tests (für Entwickler)

```bash
node tests/bav.test.js
node tests/rentenluecke.test.js
```

Die Tests laden den echten Inline-Code aus `index.html` in eine isolierte Node-VM und prüfen die reine Geschäftslogik.
