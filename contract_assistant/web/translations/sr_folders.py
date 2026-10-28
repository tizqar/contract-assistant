"""Serbian for the file manager (web/folders.py, templates/files) and the drawer's place.

Folder is "fascikla", the trash "otpad" and a file "fajl", as in the other catalogues. The
messages after a change describe the thing ("Premešteno u otpad: ...") rather than who did it,
so no past-tense verb has to guess anyone's gender. The words other catalogues already have
(Otkaži, Zatvori, Ime, Pristup, Vlasnik: {name}, može da pregleda, može da menja, Imejl
adresa) are not repeated here.
"""

CATALOGUE = {
    # The side list and the breadcrumb
    "Places": "Mesta",
    "My files": "Moji fajlovi",
    "Shared with me": "Deljeno sa mnom",
    "Trash": "Otpad",
    "All spaces": "Svi prostori",
    "{name} (top level)": "{name} (najviši nivo)",
    # A place
    "You can read and change the contracts here.": "Ovde možete da čitate i menjate ugovore.",
    "You can read the contracts here.": "Ovde možete da čitate ugovore.",
    "New folder": "Nova fascikla",
    "It goes into {folder}.": "Dodaje se u fasciklu {folder}.",
    "It goes into My files.": "Dodaje se u Moje fajlove.",
    "Review": "Status pregleda",
    "What it is, what was found": "Šta je, šta je pronađeno",
    "Last change": "Poslednja izmena",
    "This folder is empty.": "Ova fascikla je prazna.",
    "Other people share folders with you.": "Drugi dele fascikle sa vama.",
    "Open Shared with me": "Otvori Deljeno sa mnom",
    # The rows and their menus
    "Actions for {name}": "Radnje za {name}",
    "Other people can open this folder": "I drugi mogu da otvore ovu fasciklu",
    "Shared": "Deljeno",
    "Folder": "Fascikla",
    "Rename": "Preimenuj",
    "Move": "Premesti",
    "Share": "Podeli",
    "Move to the trash": "Premesti u otpad",
    # Shared with me
    (
        "Folders other people share with you. What you can do in each depends on the access "
        "they gave you."
    ): (
        "Fascikle koje drugi dele sa vama. Šta u kojoj možete da radite zavisi od pristupa "
        "koji vam je dat."
    ),
    "Owner": "Vlasnik",
    "Nobody shares a folder with you yet.": "Još niko ne deli fasciklu sa vama.",
    # The trash
    "Deleted folders and contracts wait here and can be restored.": (
        "Obrisane fascikle i ugovori čekaju ovde i mogu da se vrate."
    ),
    "Deleting them for good cannot be undone.": "Trajno brisanje ne može da se poništi.",
    "Only an administrator can delete them for good.": (
        "Samo administrator može da ih trajno obriše."
    ),
    "What it is": "Šta je",
    "Deleted": "Obrisano",
    "The trash is empty.": "Otpad je prazan.",
    "Folder, with everything in it": "Fascikla, sa svim što je u njoj",
    "Deleted {when}": "Obrisano {when}",
    "Restore": "Vrati",
    "Delete for good": "Obriši trajno",
    "Delete {name} for good?": "Trajno obrisati {name}?",
    (
        "The folder and everything in it go: its folders, its contracts, their decisions, "
        "chats and summaries."
    ): (
        "Nestaje fascikla i sve u njoj: njene fascikle, njeni ugovori, njihove odluke, "
        "razgovori i sažeci."
    ),
    "The contract goes, with its decisions, chats and summaries.": (
        "Nestaje ugovor, zajedno sa odlukama, razgovorima i sažecima."
    ),
    "This cannot be undone. The activity log keeps a line about it.": (
        "Ovo ne može da se poništi. Dnevnik aktivnosti čuva zapis o tome."
    ),
    # All spaces
    "The files of every account. As an administrator you can open and tidy any of them.": (
        "Fajlovi svih naloga. Kao administrator možete da otvorite i sredite bilo koji "
        "od njih."
    ),
    "you": "vi",
    "1 contract": "1 ugovor",
    "{count} contracts": "Ugovora: {count}",
    # The dialogs
    "Rename {name}": "Preimenuj {name}",
    "Folder name": "Naziv fascikle",
    "New name": "Novi naziv",
    "Create the folder": "Napravi fasciklu",
    "Move {name}": "Premesti {name}",
    (
        "Choose the folder it goes into. Things stay within the files of the person they "
        "belong to."
    ): "Izaberite fasciklu u koju ide. Sve ostaje među fajlovima osobe kojoj pripada.",
    "Move to": "Premesti u",
    "it is here now": "sada je ovde",
    "Move here": "Premesti ovde",
    "Share {name}": "Podeli {name}",
    "The people you add see everything in this folder, the folders inside it included.": (
        "Osobe koje dodate vide sve u ovoj fascikli, uključujući fascikle u njoj."
    ),
    (
        "Can view: read the contracts and ask about them. Can edit: also confirm passages, "
        "rename, and add contracts and folders."
    ): (
        "Može da pregleda: čita ugovore i postavlja pitanja o njima. Može da menja: i "
        "potvrđuje odlomke, preimenuje i dodaje ugovore i fascikle."
    ),
    "People with access": "Osobe sa pristupom",
    "Access of {name}": "Pristup za {name}",
    "Remove the access of {name}": "Ukloni pristup za {name}",
    "Remove access": "Ukloni pristup",
    "Only you and the administrators can open it now.": (
        "Sada mogu da je otvore samo vi i administratori."
    ),
    # The drawer
    "All files": "Svi fajlovi",
    "Where this contract is": "Gde se ovaj ugovor nalazi",
    # What a change says (web/folders.py)
    "Folder created: {name}.": "Fascikla je napravljena: {name}.",
    "Renamed: {name}.": "Preimenovano: {name}.",
    "Moved to {target}.": "Premešteno u: {target}.",
    "Moved to the trash: {name}.": "Premešteno u otpad: {name}.",
    "Restored: {name}.": "Vraćeno iz otpada: {name}.",
    "Deleted for good: {name}.": "Trajno obrisano: {name}.",
    "{name} now has access to the folder.": "{name} sada ima pristup fascikli.",
    "Access changed: {name}.": "Pristup je promenjen: {name}.",
    "{name} no longer has access to the folder.": "{name} više nema pristup fascikli.",
    "Confirm that it should be deleted for good.": "Potvrdite da treba trajno obrisati.",
}
