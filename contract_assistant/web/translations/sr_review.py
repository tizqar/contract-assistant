"""Serbian for a contract's review bar, its steps and its history (web/review.py, review/*.html).

Serbian past tense carries the speaker's gender, which the page does not know, so who took a
step reads "What happened: Name" with a participle that agrees with the contract instead.
"""

CATALOGUE = {
    # web/review.py: who changed the review last
    "Sent for approval by {name}, {when}": "Poslat na odobrenje: {name}, {when}",
    "Approved by {name}, {when}": "Odobren: {name}, {when}",
    "Sent back by {name}, {when}": "Vraćen na doradu: {name}, {when}",
    "Reopened by {name}, {when}": "Ponovo otvoren: {name}, {when}",
    # web/review.py: after a step
    "Sent for approval. The decisions are locked until an administrator answers.": (
        "Poslato na odobrenje. Odluke su zaključane dok administrator ne odgovori."
    ),
    "Approved. The decisions stay locked.": "Odobreno. Odluke ostaju zaključane.",
    "Sent back with your note.": "Vraćeno na doradu uz vašu napomenu.",
    "Reopened. Passages can be confirmed or rejected again.": (
        "Ponovo otvoreno. Odlomci se ponovo mogu potvrditi ili odbaciti."
    ),
    # review/bar.html
    "Send for approval": "Pošalji na odobrenje",
    "Send it for approval?": "Poslati ga na odobrenje?",
    (
        "An administrator will approve it or send it back. Until then, passages cannot be "
        "confirmed or rejected."
    ): (
        "Administrator će ga odobriti ili vratiti na doradu. Do tada se odlomci ne mogu "
        "potvrditi ni odbaciti."
    ),
    "Approve": "Odobri",
    "Approve the decisions as they are; they stay locked": (
        "Odobri odluke takve kakve su; ostaju zaključane"
    ),
    "Send back": "Vrati na doradu",
    "What should change?": "Šta treba promeniti?",
    "Everyone who opens the contract sees this note.": (
        "Napomenu vidi svako ko otvori ugovor."
    ),
    "Reopen": "Ponovo otvori",
    "Unlock the decisions so passages can be changed again": (
        "Otključaj odluke da bi se odlomci ponovo mogli menjati"
    ),
    "History": "Istorija",
    "What to change": "Šta treba promeniti",
    # review/panel.html and review/history.html
    "Everything done with this contract, the newest first.": (
        "Sve što je urađeno sa ovim ugovorom, od najnovijeg."
    ),
    "Close": "Zatvori",
    "Nothing has happened to this contract yet.": "Sa ovim ugovorom se još ništa nije desilo.",
    "Older": "Starije",
}
