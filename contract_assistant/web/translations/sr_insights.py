"""Serbian for the overview: its templates, the texts web/insights.py registers in TRANSLATED
and the fixed list of things to check (data/review_rules.json)."""

CATALOGUE = {
    # The overview and what happened
    "Jump to a section": "Idi na odeljak",
    "Loading": "Učitavam",
    (
        "Passages marked Check this are likely but not certain: "
        "read them and press Confirm or Not this."
    ): (
        "Odlomci označeni sa Proverite su verovatni, ali nisu sigurni: "
        "pročitajte ih i pritisnite Potvrdi ili Nije to."
    ),
    "You confirmed 1 passage.": "Potvrdili ste 1 odlomak.",
    "You confirmed {count} passages.": "Broj odlomaka koje ste potvrdili: {count}.",
    "Show the 1 passage to check": "Prikaži odlomak za proveru",
    "Show the {count} passages to check": "Prikaži odlomke za proveru ({count})",
    # Status lines while the local AI writes
    "This part is written by the local AI, which is not running on this computer.": (
        "Ovaj deo piše lokalni AI, koji nije pokrenut na ovom računaru."
    ),
    "The first draft did not match the contract; writing it again": (
        "Prva verzija se nije slagala sa ugovorom; pišem je ponovo"
    ),
    "Extracting the key terms. This can take up to half a minute.": (
        "Izdvajam ključne činjenice. To može da potraje do pola minuta."
    ),
    "Writing the note": "Pišem belešku",
    "Reading who must do what. This can take half a minute.": (
        "Čitam ko šta mora. To može da potraje pola minuta."
    ),
    "Writing the overview of the whole text: {stage}": "Pišem pregled celog teksta: {stage}",
    "section {number} of {count}": "odeljak {number} od {count}",
    "putting the sections together": "spajam odeljke",
    "Stopped.": "Zaustavljeno.",
    # Notices
    "The summary is written in English or Serbian.": (
        "Sažetak se piše na engleskom ili srpskom."
    ),
    "Show all clauses, or those that are Found, Check this or Not found.": (
        "Prikažite sve klauzule ili samo one sa statusom Pronađeno, Proverite ili Nije pronađeno."
    ),
    "The things to check changed since the page was shown; open the overview again.": (
        "Tačke za proveru su se promenile otkako je stranica prikazana; ponovo otvorite pregled."
    ),
    "This point has no passage in the contract to show.": (
        "Za ovu tačku u ugovoru nema odlomka koji bi mogao da se prikaže."
    ),
    "This is about something the contract does not say.": (
        "Ovo se odnosi na nešto što ugovor ne navodi."
    ),
    "Nothing on the list of things to check applies to this contract.": (
        "Nijedna tačka sa liste za proveru ne odnosi se na ovaj ugovor."
    ),
    "The key facts of this contract have not been read yet.": (
        "Ključne činjenice ovog ugovora još nisu pročitane."
    ),
    "There is no report of this contract yet; press Report first.": (
        "Za ovaj ugovor još nema izveštaja; prvo pritisnite Izveštaj."
    ),
    "No clause has this status.": "Nijedna klauzula nema ovaj status.",
    "The contract does not seem to define any terms.": (
        "Izgleda da ugovor ne definiše nijedan pojam."
    ),
    "None of the clauses the key terms are read from was found in this contract.": (
        "U ovom ugovoru nije pronađena nijedna od klauzula iz kojih se čitaju ključne činjenice."
    ),
    "The assistant has less practice with this kind of clause.": (
        "Asistent ima manje iskustva sa ovom vrstom klauzule."
    ),
    "The assistant has less practice with these, so it never marks them Found on its own.": (
        "Asistent ima manje iskustva sa ovim klauzulama, pa ih nikad sam ne označava kao "
        "Pronađeno."
    ),
    # Summary
    "Language of the summary": "Jezik sažetka",
    "English": "Engleski",
    "Serbian": "Srpski",
    "1 quote did not match the contract word for word.": (
        "Jedan citat se nije doslovno poklopio sa ugovorom."
    ),
    "{count} quotes did not match the contract word for word.": (
        "Broj citata koji se nisu doslovno poklopili sa ugovorom: {count}."
    ),
    # Overview of the whole text
    "Overview of the whole text": "Pregled celog teksta",
    (
        "A longer overview that reads every section of the contract, not only the clauses; "
        "a long contract takes a few minutes."
    ): (
        "Duži pregled koji čita svaki odeljak ugovora, a ne samo klauzule; za dug ugovor "
        "potrebno je nekoliko minuta."
    ),
    "Write the overview of the whole text": "Napiši pregled celog teksta",
    "Section by section · {count}": "Odeljak po odeljak · {count}",
    "Section {number}": "Odeljak {number}",
    # Key facts and key dates
    "Agreement date": "Datum ugovora",
    "Effective date": "Datum stupanja na snagu",
    "Expiration date": "Datum isteka",
    "Initial term": "Početno trajanje",
    "Renewal": "Obnavljanje",
    "Notice period": "Otkazni rok",
    "Governing law": "Merodavno pravo",
    "not stated": "nije navedeno",
    "not confirmed in the contract": "nije potvrđeno u ugovoru",
    "Copied from: {names}": "Prepisano iz: {names}",
    "Download the key facts as JSON": "Preuzmi ključne činjenice kao JSON",
    "Key dates": "Ključni datumi",
    "Dates read from the key facts above, or worked out from them.": (
        "Datumi pročitani iz ključnih činjenica iznad ili izračunati na osnovu njih."
    ),
    "No dates could be read from the key facts.": (
        "Iz ključnih činjenica nije moguće pročitati nijedan datum."
    ),
    "could not be read": "nije moguće pročitati",
    "Stated as “{text}”": "Navedeno kao „{text}“",
    "Worked out: {formula}": "Izračunato: {formula}",
    "Initial term ends": "Kraj početnog trajanja",
    "First renewal ends": "Kraj prvog obnavljanja",
    "Last day to give notice": "Poslednji dan za otkaz",
    # Things to check
    "Important": "Važno",
    "Worth a look": "Vredi pogledati",
    "Minor": "Manje važno",
    "Based on: {clauses}": "Na osnovu: {clauses}",
    "Show passage": "Prikaži odlomak",
    "Write a note": "Napiši belešku",
    # Who must do what
    "must": "mora",
    "may": "može",
    "must not": "ne sme",
    "1 entry could not be matched to the contract and was left out.": (
        "Jedna stavka nije pronađena u ugovoru, pa je izostavljena."
    ),
    "{count} entries could not be matched to the contract and were left out.": (
        "Izostavljene stavke koje nisu pronađene u ugovoru: {count}."
    ),
    "None of the clauses found says who must do what.": (
        "Nijedna pronađena klauzula ne kaže ko šta mora."
    ),
    "No clause was found in this contract, so there is nobody to list.": (
        "U ovom ugovoru nije pronađena nijedna klauzula, pa nema koga da se navede."
    ),
    # All clauses and defined terms
    "All": "Sve",
    "Show the clauses by status": "Prikaži klauzule po statusu",
    "You pressed Not this on a passage here.": "Ovde ste za jedan odlomak pritisnuli Nije to.",
    "Show every occurrence": "Prikaži sva pojavljivanja",
    # Report
    "Your report is ready.": "Izveštaj je spreman.",
    (
        "Your report is ready, but it leaves out {missing}. "
        "Once it appears in the overview, press Report again."
    ): (
        "Izveštaj je spreman, ali ne sadrži {missing}. "
        "Kada se pojavi u pregledu, ponovo pritisnite Izveštaj."
    ),
    (
        "Your report is ready, but it leaves out {missing}. "
        "Once they appear in the overview, press Report again."
    ): (
        "Izveštaj je spreman, ali ne sadrži {missing}. "
        "Kada se pojave u pregledu, ponovo pritisnite Izveštaj."
    ),
    "Your report is ready. It leaves out {missing}, which need the local AI.": (
        "Izveštaj je spreman. Ne sadrži {missing}, za šta je potreban lokalni AI."
    ),
    "the summary": "sažetak",
    "the key facts": "ključne činjenice",
    "who must do what": "odeljak Ko šta mora",
    "the overview of the whole text": "pregled celog teksta",
    "{rest} and {last}": "{rest} i {last}",
    "Download Word file": "Preuzmi Word datoteku",
    "Download Markdown": "Preuzmi Markdown",
    # Why a generated text was held back; each follows "because" ("jer")
    "the model gave an empty reply": "model nije ništa odgovorio",
    "it quotes text that is not in the contract": "citira tekst koji ne postoji u ugovoru",
    "it quotes text that is not in the clauses": "citira tekst koji ne postoji u klauzulama",
    "it quotes text that is not in the section": "citira tekst koji ne postoji u odeljku",
    "it quotes text that is not in the section summaries": (
        "citira tekst koji ne postoji u sažecima odeljaka"
    ),
    "it names numbers not in the contract": "navodi brojeve kojih nema u ugovoru",
    "it names numbers not in the clauses": "navodi brojeve kojih nema u klauzulama",
    "it names numbers not in the section": "navodi brojeve kojih nema u odeljku",
    "it names numbers not in the section summaries": (
        "navodi brojeve kojih nema u sažecima odeljaka"
    ),
    "the reply is not the JSON object the term sheet needs": (
        "odgovor nije JSON objekat potreban za ključne činjenice"
    ),
    "these values are not in the clauses word for word": (
        "ove vrednosti ne postoje doslovno u klauzulama"
    ),
    "the reply is not the JSON list the obligations need": (
        "odgovor nije JSON lista potrebna za obaveze"
    ),
    "these entries are not in the clauses word for word": (
        "ove stavke ne postoje doslovno u klauzulama"
    ),
    "no section could be summarized": "nijedan odeljak nije mogao da se sažme",
    # The things to check of data/review_rules.json: title, then note
    "No governing law clause": "Nema klauzule o merodavnom pravu",
    "Without a governing law clause the applicable law is left to the courts to decide.": (
        "Bez klauzule o merodavnom pravu, o tome koje se pravo primenjuje odlučuju sudovi."
    ),
    "Liability is uncapped for some breaches": "Za neke povrede odgovornost nije ograničena",
    "Check which breaches are carved out of the cap and whether insurance covers them.": (
        "Proverite koje su povrede izuzete iz ograničenja i da li ih pokriva osiguranje."
    ),
    "No cap on liability found": "Nije pronađeno ograničenje odgovornosti",
    "A missing cap means each party is exposed to the full damages the law allows.": (
        "Bez ograničenja svaka strana odgovara za punu štetu koju zakon dozvoljava."
    ),
    "No right to terminate without cause": "Nema prava na raskid bez razloga",
    "The parties may be bound for the whole term unless the other side breaches.": (
        "Strane mogu biti vezane tokom celog trajanja, osim ako druga strana prekrši ugovor."
    ),
    "Automatic renewal with a notice window": "Automatsko obnavljanje sa rokom za otkaz",
    "Calendar the notice deadline; missing it renews the contract for another term.": (
        "Upišite rok za otkaz u kalendar; ako ga propustite, ugovor se obnavlja za još jedan "
        "period."
    ),
    "Renewal term without a stated notice period": "Obnavljanje bez navedenog otkaznog roka",
    "Check how and when the renewal can be stopped.": (
        "Proverite kako i kada obnavljanje može da se zaustavi."
    ),
    "Non-compete obligation": "Obaveza zabrane konkurencije",
    "Check the scope, the territory and the duration, and whether it survives termination.": (
        "Proverite obim, teritoriju i trajanje, i da li važi i posle prestanka ugovora."
    ),
    "Exclusivity obligation": "Obaveza ekskluzivnosti",
    "Check which party is bound and for how long.": (
        "Proverite koja strana je vezana i koliko dugo."
    ),
    "Most favored nation pricing": "Cene po klauzuli najpovlašćenije strane",
    "Pricing given to others may have to be matched.": (
        "Cene koje su date drugima možda će morati da se ponude i ovde."
    ),
    "Change of control clause": "Klauzula o promeni kontrole",
    "A sale or merger may give the other side a right to terminate or consent.": (
        "Prodaja ili spajanje mogu drugoj strani dati pravo na raskid ili na davanje "
        "saglasnosti."
    ),
    "Liquidated damages": "Ugovorna kazna",
    "Check the amounts and whether they could be read as a penalty.": (
        "Proverite iznose i da li bi mogli da se tumače kao kazna, a ne kao naknada štete."
    ),
    "Irrevocable or perpetual licence": "Neopoziva ili trajna licenca",
    "The licence may outlive the contract; check what it covers.": (
        "Licenca može da traje duže od ugovora; proverite šta obuhvata."
    ),
    "Minimum commitment": "Minimalna obaveza",
    "Check the amounts, the period and what happens on a shortfall.": (
        "Proverite iznose, period i šta se dešava ako se minimum ne dostigne."
    ),
    "Assignment needs consent": "Za ustupanje je potrebna saglasnost",
    "Relevant for a restructuring or a sale of the business.": (
        "Važno kod restrukturiranja ili prodaje poslovanja."
    ),
    "Audit rights": "Pravo na reviziju",
    "Check who may audit, how often and at whose cost.": (
        "Proverite ko sme da vrši reviziju, koliko često i o čijem trošku."
    ),
    "Covenant not to sue": "Obaveza netuženja",
    "Check what claims are waived.": "Proverite kojih se zahteva strane odriču.",
    "No insurance obligation found": "Nije pronađena obaveza osiguranja",
    "Consider whether one side should carry cover for the risks of the deal.": (
        "Razmislite da li bi jedna strana trebalo da ima osiguranje za rizike posla."
    ),
    "No dispute resolution clause detected": "Nije pronađena klauzula o rešavanju sporova",
    "Disputes go to the courts of whichever forum has jurisdiction.": (
        "Sporove rešava onaj sud koji je nadležan."
    ),
}
