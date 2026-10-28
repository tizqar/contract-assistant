"""Serbian for the vocabulary of copy.py and the shared layout (base, workspace, components)."""

CATALOGUE = {
    # copy.STATUS and copy.STATUS_HELP
    "Found": "Pronađeno",
    "Check this": "Proverite",
    "Not found": "Nije pronađeno",
    "Confirmed": "Potvrđeno",
    "Rejected": "Odbačeno",
    "Defined term": "Definisan pojam",
    "Closest passage": "Najbliži odlomak",
    "The assistant is confident this is the passage for this clause.": (
        "Asistent je siguran da je ovo odlomak za ovu klauzulu."
    ),
    "A likely passage. Read it, then confirm it or reject it.": (
        "Verovatan odlomak. Pročitajte ga, pa ga potvrdite ili odbacite."
    ),
    "The contract was checked and no passage for this clause was found.": (
        "Ugovor je proveren i nije pronađen odlomak za ovu klauzulu."
    ),
    "A reviewer confirmed this passage; it now counts as found.": (
        "Ovaj odlomak je potvrđen u pregledu; sada se računa kao pronađen."
    ),
    "A reviewer rejected this passage; the next candidate, if any, is shown.": (
        "Ovaj odlomak je odbačen u pregledu; prikazan je sledeći kandidat, ako ga ima."
    ),
    "by {name}": "— {name}",
    "The sentence of the contract that defines the term, copied word for word.": (
        "Rečenica ugovora koja definiše pojam, prepisana doslovno."
    ),
    "A passage close to your question; it is not one of the clauses checked.": (
        "Odlomak blizak vašem pitanju; nije jedna od proveravanih klauzula."
    ),
    # AI
    "AI": "AI",
    (
        "AI text can be wrong. Quotes are copied word for word from the contract; "
        "check everything else against it."
    ): (
        "Tekst koji piše AI može biti pogrešan. Citati su prepisani doslovno iz ugovora; "
        "sve ostalo proverite u njemu."
    ),
    "The assistant's text was held back because {reason}.": (
        "Tekst asistenta je zadržan jer {reason}."
    ),
    # Buttons and labels
    "Confirm": "Potvrdi",
    "Not this": "Nije to",
    "Undo": "Poništi",
    "Show in document": "Prikaži u dokumentu",
    "Explain": "Objasni",
    "Explain in Serbian": "Objasni na srpskom",
    "Explain in English": "Objasni na engleskom",
    "Copy quote": "Kopiraj citat",
    "strong match": "jako poklapanje",
    "possible match": "moguće poklapanje",
    "Page {page}": "Strana {page}",
    # Sections
    "Summary": "Sažetak",
    "What the contract says, written from the clauses found.": (
        "Šta ugovor kaže, napisano na osnovu pronađenih klauzula."
    ),
    "Key facts": "Ključne činjenice",
    "Parties, dates, term, notice and governing law, each copied from the contract.": (
        "Strane, datumi, trajanje, otkazni rok i merodavno pravo, svako prepisano iz ugovora."
    ),
    "Things to check": "Šta proveriti",
    "Points a reader usually looks at, from a fixed rule list.": (
        "Tačke koje čitalac obično proverava, prema stalnoj listi pravila."
    ),
    "Who must do what": "Ko šta mora",
    "Obligations read from the clauses, in the contract's words.": (
        "Obaveze pročitane iz klauzula, rečima ugovora."
    ),
    "All clauses": "Sve klauzule",
    "Every clause the assistant looks for, with what it found.": (
        "Svaka klauzula koju asistent traži, sa onim što je našao."
    ),
    "Defined terms": "Definisani pojmovi",
    "Terms the contract defines, with the defining sentence.": (
        "Pojmovi koje ugovor definiše, sa rečenicom koja ih definiše."
    ),
    # Navigation and status
    "Overview": "Pregled",
    "Document": "Dokument",
    "Contracts": "Ugovori",
    "Add a contract": "Dodaj ugovor",
    "Or try a sample contract": "Ili isprobajte primer ugovora",
    "Reading the contract": "Čitam ugovor",
    "Local AI ready": "Lokalni AI je spreman",
    "AI off": "AI je isključen",
    "Report": "Izveštaj",
    "How to read this": "Kako ovo čitati",
    "What the assistant did": "Šta je asistent uradio",
    (
        "It read the contract once and looked for the clauses a lawyer checks first: who "
        "the parties are, how long it lasts, how it ends, who is liable, and about forty "
        "more."
    ): (
        "Pročitao je ugovor jednom i potražio klauzule koje pravnik prvo proverava: ko su "
        "strane, koliko traje, kako prestaje, ko odgovara, i još četrdesetak."
    ),
    "Found, Check this, Not found": "Pronađeno, Proverite, Nije pronađeno",
    (
        "Found means the assistant is confident about the passage. Check this means it "
        "found a likely passage but is not sure; read it, then press Confirm or Not this. "
        "Not found means the contract was checked and nothing matched."
    ): (
        "Pronađeno znači da je asistent siguran u odlomak. Proverite znači da je našao "
        "verovatan odlomak, ali nije siguran; pročitajte ga, pa pritisnite Potvrdi ili Nije "
        "to. Nije pronađeno znači da je ugovor proveren i ništa se nije poklopilo."
    ),
    "Quotes and AI text": "Citati i AI tekst",
    (
        "Everything in a quote block is copied word for word from the contract. Text "
        "marked AI was written by a language model on this computer from those quotes and "
        "checked against them; it can still be wrong."
    ): (
        "Sve u bloku citata prepisano je doslovno iz ugovora. Tekst označen sa AI napisao je "
        "jezički model na ovom računaru na osnovu tih citata i proveren je prema njima; ipak "
        "može biti pogrešan."
    ),
    "Nothing leaves this computer": "Ništa ne napušta ovaj računar",
    "Reviews and approval": "Pregled i odobrenje",
    (
        "Confirm or reject each passage to check, then send the contract for approval. An "
        "administrator approves it, which locks the decisions, or sends it back with a "
        "note. Every step is kept in the contract's history, with who did it and when."
    ): (
        "Potvrdite ili odbacite svaki odlomak za proveru, pa pošaljite ugovor na odobrenje. "
        "Administrator ga odobrava, čime se odluke zaključavaju, ili ga vraća uz napomenu. "
        "Svaki korak ostaje u istoriji ugovora, s tim ko ga je uradio i kada."
    ),
    (
        "The contracts, their analyses, the decisions and the activity log are kept in the "
        "app's own database; nothing is sent to an outside service."
    ): (
        "Ugovori, njihove analize, odluke i dnevnik aktivnosti čuvaju se u bazi podataka same "
        "aplikacije; ništa se ne šalje spoljnoj usluzi."
    ),
    (
        "The assistant checked this contract for {total} clauses: {found} found, "
        "{check} to check, {not_found} not found."
    ): (
        "Asistent je proverio {total} klauzula u ovom ugovoru: {found} pronađeno, "
        "{check} za proveru, {not_found} nije pronađeno."
    ),
    # base.html
    "Contract Assistant": "Asistent za ugovore",
    "Open another contract": "Otvorite drugi ugovor",
    "Compare": "Uporedi",
    "Portfolio": "Portfolio",
    "Advanced": "Napredno",
    "Advanced settings": "Napredna podešavanja",
    (
        "How sure the assistant must be before it marks a passage. Above the first value a "
        "passage is Found; between the two it is Check this; below the second the clause "
        "counts as not found. The defaults were chosen on validation data; most readers "
        "never change them."
    ): (
        "Koliko siguran asistent mora biti pre nego što označi odlomak. Iznad prve vrednosti "
        "odlomak je Pronađeno; između dve je Proverite; ispod druge klauzula se računa kao "
        "nepronađena. Podrazumevane vrednosti izabrane su na validacionim podacima; većina "
        "čitalaca ih nikad ne menja."
    ),
    "Found from": "Pronađeno od",
    "Check this from": "Proverite od",
    "Reset to defaults": "Vrati podrazumevano",
    "Technical details": "Tehnički detalji",
    "Language": "Jezik",
    "Write a DOCX and a Markdown report from what is saved": (
        "Napiši DOCX i Markdown izveštaj od onoga što je sačuvano"
    ),
    "No local language model: summaries and explanations are off": (
        "Nema lokalnog jezičkog modela: sažeci i objašnjenja su isključeni"
    ),
    "{model} through Ollama, on this computer": "{model} kroz Ollama, na ovom računaru",
    # workspace.html
    "Show": "Prikaz",
    "Chat": "Razgovor",
    "Suggested questions": "Predložena pitanja",
    "Ask about this contract": "Pitajte o ovom ugovoru",
    "Ask": "Pitaj",
    "Stop": "Zaustavi",
    # components.html
    "Written by a language model on this computer from the quotes; it can be wrong": (
        "Napisao jezički model na ovom računaru na osnovu citata; može biti pogrešno"
    ),
    # The suggested questions of ui/views.py
    "Who are the parties?": "Ko su ugovorne strane?",
    "Which law governs the contract?": "Koje pravo je merodavno za ugovor?",
    "When does the agreement expire, and does it renew?": (
        "Kada ugovor ističe i da li se obnavlja?"
    ),
    "Can either party terminate without cause?": (
        "Može li bilo koja strana da raskine ugovor bez razloga?"
    ),
    "Is there a non-compete or exclusivity obligation?": (
        "Postoji li zabrana konkurencije ili obaveza ekskluzivnosti?"
    ),
    "What are the payment terms?": "Koji su uslovi plaćanja?",
}
