"""Serbian for the chat: its templates and the texts web/chat.py registers in TRANSLATED."""

CATALOGUE = {
    # Status lines while a reply is written
    "Writing the answer": "Pišem odgovor",
    "Writing the summary": "Pišem sažetak",
    "Writing the explanation": "Pišem objašnjenje",
    # Notices and replies that say what to do
    "Type a question first.": "Prvo upišite pitanje.",
    "Ask about a clause first, then say 'explain that'.": (
        "Prvo pitajte o nekoj klauzuli, pa napišite „objasni to“."
    ),
    "No clause was found in this contract, so there is nothing to summarize.": (
        "U ovom ugovoru nije pronađena nijedna klauzula, pa nema šta da se sažme."
    ),
    "There is no passage to explain.": "Nema odlomka za objašnjenje.",
    "Explanations are written in English or Serbian.": (
        "Objašnjenja se pišu na engleskom ili srpskom."
    ),
    "Summaries and explanations need the local AI, which is off on this computer.": (
        "Za sažetke i objašnjenja potreban je lokalni AI, koji je na ovom računaru isključen."
    ),
    "The assistant did not answer. Please try again.": (
        "Asistent nije odgovorio. Pokušajte ponovo."
    ),
    "Something went wrong. Please try again.": "Došlo je do greške. Pokušajte ponovo.",
    "This answer is no longer in the chat.": "Ovog odgovora više nema u razgovoru.",
    # The reader's bubble for an Explain button
    "Explain: {label}": "Objasni: {label}",
    "Explain in English: {label}": "Objasni na engleskom: {label}",
    "Explain in Serbian: {label}": "Objasni na srpskom: {label}",
    # Headings of the AI texts
    "{label} in plain English": "{label} na jednostavnom engleskom",
    "{label} in plain Serbian": "{label} na jednostavnom srpskom",
    "Summary in English": "Sažetak na engleskom",
    "Summary in Serbian": "Sažetak na srpskom",
    "Based on": "Na osnovu",
    "and {count} more": "i još {count}",
    # Answers
    (
        "This is not one of the clauses the assistant checks, "
        "so here are the closest passages of the contract."
    ): (
        "Ovo nije jedna od klauzula koje asistent proverava, "
        "pa su ovde najbliži odlomci ugovora."
    ),
    (
        "This question is not about one of the clauses the assistant checks, "
        "and no passage of the contract comes close to it."
    ): (
        "Ovo pitanje se ne odnosi ni na jednu od klauzula koje asistent proverava, "
        "a nijedan odlomak ugovora mu nije blizak."
    ),
    "You can ask about:": "Možete pitati o sledećem:",
    "For example:": "Na primer:",
    "The contract was checked for this clause; no passage matches it.": (
        "Ugovor je proveren za ovu klauzulu; nijedan odlomak joj ne odgovara."
    ),
    (
        "A matching passage was found but it could not be verified word for word "
        "against the contract, so it is not shown."
    ): (
        "Pronađen je odgovarajući odlomak, ali nije mogao doslovno da se proveri prema "
        "ugovoru, pa nije prikazan."
    ),
    # The topics of ui/views.CLAUSE_GROUPS, listed when a question matches no clause
    # The empty chat, the feedback and the tools
    "Ask anything about this contract: who signed it, when it ends, how to cancel it.": (
        "Pitajte bilo šta o ovom ugovoru: ko ga je potpisao, kada ističe, kako se raskida."
    ),
    "Every answer quotes the contract word for word, so you can check it.": (
        "Svaki odgovor doslovno citira ugovor, pa ga možete proveriti."
    ),
    "Helpful": "Korisno",
    "Not helpful": "Nije korisno",
    "Noted, thank you.": "Zabeleženo, hvala.",
    "Stopped. This question was not answered.": (
        "Zaustavljeno. Na ovo pitanje nije odgovoreno."
    ),
    "Clear chat": "Obriši razgovor",
}
