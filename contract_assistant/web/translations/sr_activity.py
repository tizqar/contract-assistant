"""Serbian for the permission messages (access.py) and the activity log (audit.py).

Serbian past tense carries the speaker's gender, which the log does not know, so each line
reads "Name: what happened" with a passive or a noun instead of a verb in the past tense.
"""

CATALOGUE = {
    # access.py
    "You do not have permission to do that.": "Nemate dozvolu za to.",
    "That folder or contract does not exist, or it is not shared with you.": (
        "Ta fascikla ili ugovor ne postoji, ili nije podeljen sa vama."
    ),
    "This contract is sent for approval or approved, so its passages cannot be changed.": (
        "Ovaj ugovor je poslat na odobrenje ili je odobren, pa se njegovi odlomci ne mogu menjati."
    ),
    "This contract's review is not at a step where that can be done.": (
        "Pregled ovog ugovora nije u koraku u kom se to može uraditi."
    ),
    # audit.py: accounts
    "{actor} signed in": "{actor}: prijava",
    "A sign-in as {email} failed": "Neuspešna prijava kao {email}",
    "{actor} signed out": "{actor}: odjava",
    "{actor} changed their password": "{actor}: promenjena lozinka",
    "{actor} created the account of {subject} ({role})": (
        "{actor}: napravljen nalog za {subject} ({role})"
    ),
    "{actor} renamed {old_name} to {subject}": "{actor}: {old_name} preimenovan u {subject}",
    "{actor} made {subject} {role}": "{actor}: {subject} dobija ulogu {role}",
    "{actor} deactivated the account of {subject}": "{actor}: deaktiviran nalog {subject}",
    "{actor} activated the account of {subject}": "{actor}: aktiviran nalog {subject}",
    "{actor} reset the password of {subject}": "{actor}: resetovana lozinka za {subject}",
    # audit.py: folders
    "{actor} created the folder {folder}": "{actor}: napravljena fascikla {folder}",
    "{actor} renamed the folder {old_name} to {folder}": (
        "{actor}: fascikla {old_name} preimenovana u {folder}"
    ),
    "{actor} moved the folder {folder} to {target}": (
        "{actor}: fascikla {folder} premeštena u {target}"
    ),
    "{actor} moved the folder {folder} to the trash": (
        "{actor}: fascikla {folder} premeštena u otpad"
    ),
    "{actor} restored the folder {folder}": "{actor}: vraćena fascikla {folder}",
    "{actor} deleted the folder {name} for good": "{actor}: fascikla {name} trajno obrisana",
    "{actor} shared the folder {folder} with {subject} ({level})": (
        "{actor}: fascikla {folder} podeljena sa {subject} ({level})"
    ),
    "{actor} stopped sharing the folder {folder} with {subject}": (
        "{actor}: fascikla {folder} više nije podeljena sa {subject}"
    ),
    # audit.py: contracts
    "{actor} added {contract}": "{actor}: dodat ugovor {contract}",
    "{contract} was imported from the cache folder": "{contract} je uvezen iz keš fascikle",
    "{actor} renamed {old_name} to {contract}": "{actor}: {old_name} preimenovan u {contract}",
    "{actor} moved {contract} to {target}": "{actor}: {contract} premešten u {target}",
    "{actor} moved {contract} to the trash": "{actor}: {contract} premešten u otpad",
    "{actor} restored {contract}": "{actor}: vraćen ugovor {contract}",
    "{actor} deleted {name} for good": "{actor}: {name} trajno obrisan",
    "{actor} had {contract} read again for new clauses": (
        "{actor}: {contract} ponovo pročitan zbog novih klauzula"
    ),
    "{actor} downloaded {what} of {contract}": "{actor}: preuzeto {what} za {contract}",
    # audit.py: passages and reviews
    "{actor} confirmed the {clause} passage of {contract}": (
        "{actor}: potvrđen odlomak {clause} u ugovoru {contract}"
    ),
    "{actor} rejected the {clause} passage of {contract}": (
        "{actor}: odbačen odlomak {clause} u ugovoru {contract}"
    ),
    "{actor} undid the decision on {clause} in {contract}": (
        "{actor}: poništena odluka o klauzuli {clause} u ugovoru {contract}"
    ),
    "{actor} sent {contract} for approval": "{actor}: {contract} poslat na odobrenje",
    "{actor} approved {contract}": "{actor}: odobren ugovor {contract}",
    "{actor} sent {contract} back: {note}": "{actor}: {contract} vraćen na doradu: {note}",
    "{actor} reopened the review of {contract}": (
        "{actor}: ponovo otvoren pregled ugovora {contract}"
    ),
    "{actor} rated an answer about {contract}": "{actor}: ocenjen odgovor o ugovoru {contract}",
    "{actor} cleared their chat about {contract}": "{actor}: obrisan razgovor o ugovoru {contract}",
    "The system": "Sistem",
    "Someone": "Neko",
    # web/activity.py
    "can view": "može da pregleda",
    "can edit": "može da menja",
    "{date}, {time}": "{date}, {time}",
    "Accounts and sign-ins": "Nalozi i prijave",
    "Folders and sharing": "Fascikle i deljenje",
    "Passage decisions": "Odluke o odlomcima",
    # copy.REVIEW_STATUS and copy.REVIEW_HELP
    "In review": "U pregledu",
    "Sent for approval": "Poslat na odobrenje",
    "Approved": "Odobren",
    "Passages can be confirmed or rejected; send it for approval when done.": (
        "Odlomci se mogu potvrditi ili odbaciti; kada završite, pošaljite ga na odobrenje."
    ),
    "Waiting for an administrator; the decisions are locked meanwhile.": (
        "Čeka administratora; odluke su u međuvremenu zaključane."
    ),
    "An administrator approved it; the decisions are locked.": (
        "Administrator ga je odobrio; odluke su zaključane."
    ),
}
