"""Serbian for the admin pages and the account page (web/admin.py, web/account.py, admin/*.html,
account/*.html).

Serbian past tense carries the speaker's gender, which the pages do not know, so the lines
about a person use the present tense, a passive, or a participle that agrees with a noun
("osoba", "nalog") instead.
"""

CATALOGUE = {
    # admin/layout.html and admin/notice.html
    "Admin pages": "Stranice administracije",
    "Users": "Korisnici",
    "Activity": "Aktivnost",
    "Approvals": "Odobrenja",
    "Waiting for approval": "Čeka odobrenje",
    "This page is for administrators": "Ova stranica je za administratore",
    "Your account cannot open it. If something needs doing here, ask an administrator.": (
        "Vaš nalog ne može da je otvori. Ako ovde nešto treba uraditi, obratite se "
        "administratoru."
    ),
    "Go to your files": "Idi na svoje fajlove",
    "No such account": "Nalog ne postoji",
    "Back to the users": "Nazad na korisnike",
    # admin/users.html
    "Accounts are deactivated, never deleted, so their history stays.": (
        "Nalozi se deaktiviraju, nikad ne brišu, pa njihova istorija ostaje."
    ),
    "Name": "Ime",
    "Role": "Uloga",
    "Status": "Stanje",
    "Last sign-in": "Poslednja prijava",
    "Created": "Napravljen",
    "(you)": "(vi)",
    "Active": "Aktivan",
    "Deactivated": "Deaktiviran",
    "Temporary password": "Privremena lozinka",
    "The person must choose a new password at the next sign-in.": (
        "Osoba mora da izabere novu lozinku pri sledećoj prijavi."
    ),
    "Never": "Nikad",
    "Create an account": "Napravite nalog",
    (
        "The new account gets a temporary password, shown here once. The person chooses "
        "their own at the first sign-in."
    ): (
        "Novi nalog dobija privremenu lozinku, koja se ovde prikazuje samo jednom. Osoba bira "
        "svoju pri prvoj prijavi."
    ),
    "Create the account": "Napravi nalog",
    # admin/macros.html: a temporary password shown once, and a list of events
    "Temporary password for {name}": "Privremena lozinka za {name}",
    "Password copied.": "Lozinka je kopirana.",
    "The password could not be copied; select it and copy it by hand.": (
        "Lozinka nije mogla da se kopira; označite je i kopirajte ručno."
    ),
    "Copy": "Kopiraj",
    (
        "It is shown only this once. Give it to {name} ({email}); they must change it at the "
        "first sign-in."
    ): (
        "Prikazuje se samo ovaj put. Dajte je osobi {name} ({email}); pri prvoj prijavi mora "
        "da je promeni."
    ),
    "Open the contract": "Otvori ugovor",
    "Network address": "Mrežna adresa",
    # admin/user.html
    "Save the name": "Sačuvaj ime",
    (
        "An administrator also manages the accounts, approves reviews, reads the whole "
        "activity log and deletes from the trash for good."
    ): (
        "Administrator još upravlja nalozima, odobrava preglede, čita ceo dnevnik aktivnosti "
        "i trajno briše iz otpada."
    ),
    "Save the role": "Sačuvaj ulogu",
    "Access": "Pristup",
    (
        "Deactivating the account signs out its browsers and stops it from signing in. Its "
        "contracts and history stay."
    ): (
        "Deaktiviranjem naloga odjavljuju se njegovi pregledači i prijava više nije moguća. "
        "Njegovi ugovori i istorija ostaju."
    ),
    "Deactivate the account of {name}? Its browsers are signed out at once.": (
        "Deaktivirati nalog {name}? Njegovi pregledači se odmah odjavljuju."
    ),
    "Deactivate the account": "Deaktiviraj nalog",
    "This account is deactivated. Activating it lets the person sign in again.": (
        "Ovaj nalog je deaktiviran. Kada se aktivira, osoba ponovo može da se prijavi."
    ),
    "Activate the account": "Aktiviraj nalog",
    "To change your own password, use the password page.": (
        "Svoju lozinku menjate na stranici za lozinku."
    ),
    (
        "For a person who forgot their password: they get a temporary one from you, and "
        "choose their own at the next sign-in. The old password stops working and the "
        "account's browsers are signed out."
    ): (
        "Za osobu koja je zaboravila lozinku: od vas dobija privremenu, a svoju bira pri "
        "sledećoj prijavi. Stara lozinka prestaje da važi i pregledači naloga se odjavljuju."
    ),
    "Reset the password of {name}? The current password stops working.": (
        "Resetovati lozinku za {name}? Trenutna lozinka prestaje da važi."
    ),
    "Reset the password": "Resetuj lozinku",
    "Recent activity": "Nedavne aktivnosti",
    "What this person did, and what was done to their account and contracts.": (
        "Šta je ova osoba uradila i šta je urađeno sa njenim nalogom i ugovorima."
    ),
    "Everything this person did": "Sve što je ova osoba uradila",
    "No activity yet.": "Još nema aktivnosti.",
    # web/admin.py: after a change
    "The name is saved.": "Ime je sačuvano.",
    "The role is saved.": "Uloga je sačuvana.",
    "The account is deactivated and its browsers are signed out.": (
        "Nalog je deaktiviran i njegovi pregledači su odjavljeni."
    ),
    "The account is activated; the person can sign in again.": (
        "Nalog je aktiviran; osoba ponovo može da se prijavi."
    ),
    "Account created for {name} ({email}).": "Napravljen nalog za {name} ({email}).",
    "The password is reset and the account's browsers are signed out.": (
        "Lozinka je resetovana i pregledači naloga su odjavljeni."
    ),
    # admin/activity.html
    "Everything done in the app, newest first. Nothing here can be changed or deleted.": (
        "Sve što je urađeno u aplikaciji, od najnovijeg. Ovde se ništa ne može izmeniti ni "
        "obrisati."
    ),
    "Filter the activity": "Filtriranje aktivnosti",
    "Person": "Osoba",
    "Everyone": "Svi",
    "Kind of activity": "Vrsta aktivnosti",
    "All kinds": "Sve vrste",
    "Contract name contains": "Naziv ugovora sadrži",
    "From": "Od",
    "To": "Do",
    "Filter": "Filtriraj",
    "Clear the filters": "Ukloni filtere",
    "Back to the newest": "Nazad na najnovije",
    "No activity matches these filters.": "Nijedna aktivnost ne odgovara ovim filterima.",
    # admin/approvals.html
    (
        "The longest waiting first. Approving keeps the decisions locked; sending back "
        "reopens the review with your note."
    ): (
        "Prvo ugovori koji najduže čekaju. Posle odobrenja odluke ostaju zaključane; vraćanjem "
        "na doradu pregled se ponovo otvara uz vašu napomenu."
    ),
    "No contract is waiting for approval.": "Nijedan ugovor ne čeka odobrenje.",
    "Owner: {name}": "Vlasnik: {name}",
    "For example: the notice period on page 4 is not the one we agreed.": (
        "Na primer: otkazni rok na 4. strani nije onaj koji je dogovoren."
    ),
    "Send back with this note": "Vrati na doradu uz ovu napomenu",
    # account/account.html
    "Your details": "Vaši podaci",
    "What your role lets you do": "Šta vam uloga omogućava",
    "What you are responsible for": "Vaše odgovornosti",
    "Signed-in browsers": "Prijavljeni pregledači",
    "This browser": "Ovaj pregledač",
    "Signed in {when}": "Prijava: {when}",
    "Last seen {when}": "Poslednja aktivnost: {when}",
    "Sign out the other browsers": "Odjavi ostale pregledače",
    "Only this browser is signed in.": "Prijavljen je samo ovaj pregledač.",
    "Your recent activity": "Vaše nedavne aktivnosti",
    # web/account.py: what a role lets its holder do and what it is responsible for
    (
        "Add contracts to your own space and keep them in folders: rename, move, delete to "
        "the trash and restore them."
    ): (
        "Dodajete ugovore u svoj prostor i držite ih u fasciklama: preimenujete ih, premeštate, "
        "brišete u otpad i vraćate."
    ),
    "Share your folders with chosen people, to view or to edit, and stop sharing them.": (
        "Delite svoje fascikle sa izabranim osobama, za pregled ili za izmene, i prestajete da "
        "ih delite."
    ),
    (
        "In a folder shared with you to view: read the contracts, ask about them, have "
        "summaries written and download them."
    ): (
        "U fascikli podeljenoj sa vama za pregled: čitate ugovore, postavljate pitanja o "
        "njima, tražite sažetke i preuzimate ih."
    ),
    (
        "In a folder shared with you to edit: also confirm or reject passages, rename, add "
        "contracts and folders, and send a contract for approval."
    ): (
        "U fascikli podeljenoj sa vama za izmene: još potvrđujete ili odbacujete odlomke, "
        "preimenujete, dodajete ugovore i fascikle i šaljete ugovor na odobrenje."
    ),
    "Change your name and your password, and see the history of the contracts you can open.": (
        "Menjate svoje ime i lozinku i vidite istoriju ugovora koje možete da otvorite."
    ),
    "Everything a user can do, in every person's space.": (
        "Sve što može korisnik, u prostoru svake osobe."
    ),
    (
        "Create accounts for users and administrators, rename them, change their roles, "
        "deactivate and activate them, and reset passwords. Accounts are never deleted, so "
        "their history stays."
    ): (
        "Pravite naloge za korisnike i administratore, preimenujete ih, menjate im uloge, "
        "deaktivirate ih i aktivirate i resetujete lozinke. Nalozi se nikad ne brišu, pa "
        "njihova istorija ostaje."
    ),
    (
        "Approve a contract sent for approval, send it back with a note, or reopen an "
        "approved one."
    ): (
        "Odobravate ugovor poslat na odobrenje, vraćate ga na doradu uz napomenu ili ponovo "
        "otvarate odobren ugovor."
    ),
    "Read the whole activity log, filter it and download it as a CSV file.": (
        "Čitate ceo dnevnik aktivnosti, filtrirate ga i preuzimate kao CSV fajl."
    ),
    "Delete folders and contracts in the trash for good.": (
        "Trajno brišete fascikle i ugovore iz otpada."
    ),
    "Check the passages of a contract while it is In review, then send it for approval.": (
        "Proveravate odlomke ugovora dok je U pregledu, a zatim ga šaljete na odobrenje."
    ),
    (
        "Share a folder only with the people who need it, and stop sharing when they no "
        "longer do."
    ): (
        "Fasciklu delite samo sa osobama kojima je potrebna i prestajete da je delite kada im "
        "više ne treba."
    ),
    (
        "Keep your password to yourself. If you forget it, an administrator can give you a "
        "temporary one."
    ): (
        "Lozinku čuvate za sebe. Ako je zaboravite, administrator vam može dati privremenu."
    ),
    "Approve or send back the contracts waiting for approval.": (
        "Odobravate ili vraćate na doradu ugovore koji čekaju odobrenje."
    ),
    "Give each person the account they need, and deactivate the accounts no longer needed.": (
        "Svakoj osobi dajete nalog koji joj treba i deaktivirate naloge koji više nisu potrebni."
    ),
    "Keep a second administrator, so that someone can always manage the accounts.": (
        "Imate još jednog administratora, da bi uvek neko mogao da upravlja nalozima."
    ),
    # web/account.py: notices and browsers
    "Your name is saved.": "Vaše ime je sačuvano.",
    "Other browsers signed out: {count}.": "Odjavljeni drugi pregledači: {count}.",
    "Unknown browser": "Nepoznat pregledač",
    "{browser} on {system}": "{browser} na sistemu {system}",
}
