"""Serbian for the contracts pages: the first page, the drawer, the document pane and its
search, the uploads and samples, the comparison and the portfolio, with the texts
web/library.py registers in TRANSLATED.

The words the other catalogues already have (Pronađeno, Proverite, Potvrdi, Ugovori, Uporedi,
the clause names) are not repeated here.
"""

CATALOGUE = {
    # The first page
    "Read a contract with a local assistant": "Pročitajte ugovor uz pomoć lokalnog asistenta",
    (
        "Upload a contract. The assistant reads it once on this computer, finds the clauses a "
        "lawyer checks first, and then answers your questions with quotes."
    ): (
        "Otpremite ugovor. Asistent ga jednom pročita na ovom računaru, pronađe klauzule koje "
        "pravnik prvo proverava, a zatim odgovara na vaša pitanja citatima iz ugovora."
    ),
    # Adding a contract
    "PDF or text file": "PDF ili tekstualna datoteka",
    (
        "A PDF or a .txt file. Scanned pages are read with OCR, which takes longer. "
        "The file stays on this computer."
    ): (
        "PDF ili .txt datoteka. Skenirane strane se čitaju pomoću OCR-a, što traje duže. "
        "Datoteka ostaje na ovom računaru."
    ),
    "Upload and read": "Otpremi i pročitaj",
    "Choose a PDF or TXT file first.": "Prvo izaberite PDF ili TXT datoteku.",
    "Reading the file": "Čitam datoteku",
    "Finding the clauses · {done} of {total} · {elapsed} s": (
        "Tražim klauzule · {done} od {total} · {elapsed} s"
    ),
    "Saving": "Čuvam",
    "{name} was read before; opening it": "Ugovor „{name}“ je već pročitan; otvaram ga",
    "Read again for the clauses added since; opening it": (
        "Ugovor je ponovo pročitan zbog klauzula dodatih u međuvremenu; otvaram ga"
    ),
    "Done; opening the contract": "Gotovo; otvaram ugovor",
    (
        "New contracts cannot be read on this computer yet: the model that reads them is not "
        "installed. The README says how to add it."
    ): (
        "Novi ugovori još ne mogu da se čitaju na ovom računaru: model koji ih čita nije "
        "instaliran. U datoteci README piše kako da ga dodate."
    ),
    "The samples need that model too.": "I primerima je potreban taj model.",
    "There is no sample contract by that name.": "Ne postoji primer ugovora sa tim nazivom.",
    # Why a file could not be read
    (
        "This PDF is a scan. Reading scans needs Tesseract with its language packs: "
        "brew install tesseract tesseract-lang."
    ): (
        "Ovaj PDF je sken. Za čitanje skenova potreban je Tesseract sa jezičkim paketima: "
        "brew install tesseract tesseract-lang."
    ),
    "This PDF has no text the assistant can read.": (
        "Ovaj PDF nema tekst koji asistent može da pročita."
    ),
    "This file cannot be read: choose a PDF or a TXT file.": (
        "Ova datoteka ne može da se pročita: izaberite PDF ili TXT datoteku."
    ),
    "This text file is not saved as UTF-8; save it as UTF-8 and add it again.": (
        "Ova tekstualna datoteka nije sačuvana kao UTF-8; sačuvajte je kao UTF-8 i dodajte je "
        "ponovo."
    ),
    "The saved analysis of this contract is damaged and cannot be opened.": (
        "Sačuvana analiza ovog ugovora je oštećena i ne može da se otvori."
    ),
    "No page image: the PDF of this contract is not kept on this computer.": (
        "Nema slike strane: PDF ovog ugovora nije sačuvan na ovom računaru."
    ),
    "This page is not in the PDF.": "Ova strana ne postoji u PDF-u.",
    "The analysis file of this contract was not found.": (
        "Datoteka sa analizom ovog ugovora nije pronađena."
    ),
    # The drawer and the sizes of a contract
    "PDF": "PDF",
    "Text": "Tekst",
    "{count} pages": "{count} str.",
    "{count} words": "{count} reči",
    "1 scanned page read with OCR": "1 skenirana strana pročitana pomoću OCR-a",
    "{count} scanned pages read with OCR": "Skenirane strane pročitane pomoću OCR-a: {count}",
    "{count} to check": "{count} za proveru",
    "nothing to check": "ništa za proveru",
    "More": "Više",
    "Download what the assistant found (JSON)": "Preuzmi ono što je asistent pronašao (JSON)",
    # Review severities as tags; the words of the overview's severities, with a count
    "{count} high": "Važno: {count}",
    "{count} medium": "Vredi pogledati: {count}",
    "{count} low": "Manje važno: {count}",
    "None": "Nema",
    # The document pane
    "Show the contract as": "Prikaži ugovor kao",
    "Page": "Strana",
    "Previous page": "Prethodna strana",
    "Next page": "Sledeća strana",
    "of {pages}": "od {pages}",
    "Pages {first}–{last}": "Strane {first}–{last}",
    "Excerpt around the highlighted passage": "Izvod oko označenog odlomka",
    "Beginning of the contract": "Početak ugovora",
    (
        "This page is a scan read with OCR; the words are shown as read, and the page image "
        "cannot box them."
    ): (
        "Ova strana je sken pročitan pomoću OCR-a; reči su prikazane onako kako su pročitane, "
        "a na slici strane ne mogu da se uokvire."
    ),
    "Page {page} of the contract": "Strana {page} ugovora",
    "Page {page} of the contract with the {label} passage boxed": (
        "Strana {page} ugovora sa uokvirenim odlomkom: {label}"
    ),
    "Page {page} of the contract with each occurrence of “{term}” boxed": (
        "Strana {page} ugovora na kojoj je uokvireno svako pojavljivanje izraza „{term}“"
    ),
    "Pick a passage in the overview or ask a question to see it here.": (
        "Izaberite odlomak u pregledu ili postavite pitanje da biste ga videli ovde."
    ),
    "Quote copied.": "Citat je kopiran.",
    "The quote could not be copied; select it and copy it by hand.": (
        "Citat nije mogao da se kopira; označite ga i kopirajte ručno."
    ),
    # The search in the document
    "Search the contract": "Pretraga ugovora",
    "Find a word or a phrase": "Pronađite reč ili izraz",
    "Find": "Traži",
    "Type at least two characters": "Upišite najmanje dva znaka",
    "No occurrences": "Nema pojavljivanja",
    "1 occurrence": "1 pojavljivanje",
    "{count} occurrences": "Broj pojavljivanja: {count}",
    "1 occurrence on 1 page": "1 pojavljivanje na 1 strani",
    "{count} occurrences on 1 page": "Broj pojavljivanja: {count}, sva na 1 strani",
    "{count} occurrences on {pages} pages": (
        "Broj pojavljivanja: {count} · broj strana: {pages}"
    ),
    "Occurrence {index} of {count}": "Pojavljivanje {index} od {count}",
    "Showing the first {limit}.": "Prikazano je prvih {limit}.",
    "Previous occurrence": "Prethodno pojavljivanje",
    "Next occurrence": "Sledeće pojavljivanje",
    "List them all": "Prikaži spisak",
    "Clear": "Obriši",
    # The comparison
    "Breadcrumb": "Putanja",
    "Workspace": "Radni prostor",
    "Compare two contracts": "Poređenje dva ugovora",
    "First contract": "Prvi ugovor",
    "Second contract": "Drugi ugovor",
    "Open": "Otvori",
    "Swap the two contracts": "Zameni mesta ugovorima",
    "Comparing the two contracts": "Poredim dva ugovora",
    "Comparing the clauses": "Poredim klauzule",
    "Reading the key terms of {name}": "Čitam ključne činjenice ugovora „{name}“",
    "Choose a contract to compare with.": "Izaberite ugovor sa kojim poredite.",
    "Choose two different contracts: the same one is on both sides.": (
        "Izaberite dva različita ugovora: isti ugovor je na obe strane."
    ),
    "The other contract has not been analysed, or its analysis is damaged.": (
        "Drugi ugovor nije analiziran ili je njegova analiza oštećena."
    ),
    "Add a second contract to compare it with the first.": (
        "Dodajte drugi ugovor da biste ga uporedili sa prvim."
    ),
    "of {total} clauses differ": "od {total} klauzula se razlikuje",
    "{count} of {total} key terms differ.": (
        "Razlikuje se {count} od {total} ključnih činjenica."
    ),
    "Each value is copied word for word from the contract.": (
        "Svaka vrednost je doslovno prepisana iz ugovora."
    ),
    "The key terms are not compared.": "Ključne činjenice nisu upoređene.",
    # Each reason follows "nisu upoređene:" or "jer"
    "The key terms are not compared: {reason}.": "Ključne činjenice nisu upoređene: {reason}.",
    "they need the local AI, which is off": "za njih je potreban lokalni AI, koji je isključen",
    "no clause they are read from was found in {name}": (
        "u ugovoru „{name}“ nije pronađena nijedna klauzula iz koje se čitaju"
    ),
    "the local AI did not answer": "lokalni AI nije odgovorio",
    "The key terms of {name} are not shown because {reason}.": (
        "Ključne činjenice ugovora „{name}“ nisu prikazane jer {reason}."
    ),
    "Key terms": "Ključne činjenice",
    "Differs": "Razlikuje se",
    "Term": "Stavka",
    "not verified": "nije provereno",
    "Important:": "Važno:",
    "None of the rules applies to this contract.": (
        "Nijedno pravilo se ne odnosi na ovaj ugovor."
    ),
    "Clauses": "Klauzule",
    "Show clauses": "Prikaži klauzule",
    "Differences": "Razlike",
    "Same": "Iste",
    "1 clause": "1 klauzula",
    "{count} clauses": "Broj klauzula: {count}",
    "Only in {name}": "Samo u ugovoru „{name}“",
    "In both, found in one and to check in the other": (
        "U oba, u jednom pronađeno, a u drugom za proveru"
    ),
    "{left} first, {right} second": "prva oznaka: {left}, druga: {right}",
    "Same in both": "Isto u oba",
    # The portfolio
    "Export CSV": "Izvezi CSV",
    "No contracts yet.": "Još nema ugovora.",
    "Add one or try a sample": "Dodajte jedan ili isprobajte primer",
    "1 PDF": "1 PDF",
    "{count} PDFs": "PDF: {count}",
    "1 text file": "1 tekstualna datoteka",
    "{count} text files": "tekstualne datoteke: {count}",
    "Clauses found": "Pronađene klauzule",
    "of {total} looked for in each contract": "od {total} koje se traže u svakom ugovoru",
    "Passages to check": "Odlomci za proveru",
    "read each, then confirm or reject it": "pročitajte svaki, pa ga potvrdite ili odbacite",
    "High-priority points": "Važne tačke",
    "{medium} medium, {low} low": "vredi pogledati: {medium}, manje važno: {low}",
    "Next notice deadline": "Sledeći rok za otkaz",
    "{name}, {period} before the term ends": "{name}, {period} pre isteka trajanja",
    "Search contracts": "Pretražite ugovore",
    "Search by name, party or governing law": (
        "Tražite po nazivu, ugovornoj strani ili merodavnom pravu"
    ),
    "Filter the contracts": "Filtriraj ugovore",
    "With high-priority points": "Sa važnim tačkama",
    "With a renewal term": "Sa periodom obnavljanja",
    "No governing law": "Bez merodavnog prava",
    "Key terms are copied word for word from the contract.": (
        "Ključne činjenice su doslovno prepisane iz ugovora."
    ),
    "The key terms need the local AI, which is off.": (
        "Za ključne činjenice potreban je lokalni AI, koji je isključen."
    ),
    "Contract": "Ugovor",
    "Clauses found, of {total}": "Pronađene klauzule, od {total}",
    "not extracted": "nije izdvojeno",
    "No contract matches the search.": "Nijedan ugovor ne odgovara pretrazi.",
    "Open a row to work on that contract.": "Otvorite red da biste radili na tom ugovoru.",
    # Dates, as "1. jul 2027."
    "{day} {month} {year}": "{day}. {month} {year}.",
    "Jan": "jan",
    "Feb": "feb",
    "Mar": "mar",
    "Apr": "apr",
    "May": "maj",
    "Jun": "jun",
    "Jul": "jul",
    "Aug": "avg",
    "Sep": "sep",
    "Oct": "okt",
    "Nov": "nov",
    "Dec": "dec",
}
