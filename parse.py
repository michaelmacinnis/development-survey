#!/usr/bin/env python3

import csv
import re
import sys

import matplotlib.pyplot as plt

from scipy.stats import kendalltau
from scipy.stats import linregress
from scipy.stats import spearmanr

from sklearn.metrics import mean_squared_error

label = {
    "academic_exp": "Years of experience making software in an academic setting",
    "awesome": "Platforms/languages/programs/tools/technologies that are awesome and why",
    "awful": "Platforms/languages/programs/tools/technologies that are awful and why",
    "identifier": "Please provide an identifier of your choosing. This can be your real name, email address, discord username, etc. If you would prefer not to provide identifying information please skip this question.",
    "personal_exp": "Years of experience making software in a personal setting",
    "professional_exp": "Years of experience making software in a professional setting",
    "languages_by_pref": "Programming languages by preference (from most to least preferred)",
    "languages_by_use": "Programming languages by use (from most to least frequently used)",
    "platforms_by_pref": "Platforms by preference (from most to least preferred)",
    "platforms_by_use": "Platforms by use (from most to least frequently used)",
    "other_by_pref": "Other tools/technologies by preference (from most to least preferred)",
    "other_by_use": "Other tools/technologies by use (from most to least frequently used)",
    "tools_by_pref": "Programs used to make software by preference (from most to least preferred)",
    "tools_by_use": "Programs used to make software by use (from most to least frequently used)",
}

min_freq = 5

pattern = re.compile("[,/\n]")

# It's a small data set. Read it all into memory.
rows = [row for row in csv.DictReader(sys.stdin)]


def age(r):
    if r == "20-29":
        return 0.2
    elif r == "30-39":
        return 0.3
    elif r == "40-49":
        return 0.4


def cleaned(s):
    s = s.strip().lower()

    # Replace anything beginning with these prefixes with the prefix.
    for r in [
        "...",
        "abandoned",
        "and ",
        "azure",
        "c++",
        "discord",
        "each fills a niche",
        "github",
        "google cloud",
        "google docs",
        "intellij",
        "it's hard to give",
        "javascript",
        "jetbrains",
        "js",
        "jupyter",
        "linux (centos",
        "linux (currently 18.04 and 20.04",
        "md",
        "none - all",
        "note: they",
        "omnigraffle",
        "pharo",
        "rest of the pack",
        "rider",
        "ruby",
        "search engine",
        "she'll",
        "shell",
        "smalltalk",
        "sniffers",
        "sql",
        "used to deploy",
        "used to develop",
        "vmware",
        "windows",
        "writing",
        "wsl ",
    ]:
        if s.startswith(r) and s != r:
            print("replacing", s, "with", r, file=sys.stderr)
            s = r

    sentinel = "not found"

    # Replace the following.
    r = {
        "...": None,
        "a": None,
        "abandoned": None,
        "ai frameworks": None,
        "and ": None,
        "android studio": "jetbrains",
        "assignments": None,
        "assembler": "assembly",
        "asynchronous tools (e.g.": None,
        "bash": "shell",
        "bicep": "azure",
        "bitbucket": "atlassian",
        "bitbucket)": "atlassian",
        "bsd": "bsd",
        "centos": "linux",
        "chrome os": "ChromeOS",
        "clang": "llvm",
        "clion": "jetbrains",
        "collaboration platforms (e.g.": None,
        "confluence": "atlassian",
        "databases (e.g.": None,
        "datagrip": "jetbrains",
        "debugging": None,
        "each fills a niche": None,
        "freebsd": "bsd",
        "gcs": "google cloud",
        "gimp)": "gimp",
        "goland": "jetbrains",
        "golang": "go",
        "googledoc": "googledocs",
        "google docs": "googledocs",
        "google sheets": "googledocs",
        "(html": "html",
        "image manipulators (e.g.": None,
        "imovie (video editing) (skype": "imovie",  # Skip skype it only came up once.
        "intellij": "jetbrains",
        "ios": "iOS",
        "irc)": "irc",
        "it's hard to give": None,
        "java script": "javascript",
        "jira": "atlassian",
        "js": "javascript",
        "linux (centos": "linux",
        "linux (currently 18.04 and 20.04": "linux",
        "linux (debian)": "linux",
        "linux (desktop)": "linux",
        "linux (server)": "linux",
        "linux (ubuntu)": "linux",
        "mac": "macos",
        "mac os": "macos",
        "md": "markdown",
        "minukube": "minikube",
        "mint)": "linux",
        "msteams)": "msteams",
        "mvn": "maven",
        "mysql)": "mysql",
        "n": None,
        "node.js)": "node.js",
        "none - all": None,
        "note: they": None,
        "note: same as before": None,
        "plain bash (ad hoc scripts)": "shell",
        "planning": None,
        "platforms (tensorflow": "tensorflow",
        "plugins": None,
        "power bi": "powerbi",
        "prototypes": None,
        "pycharm": "jetbrains",
        "rancher linux": "linux",
        "rider": "jetbrains",
        "rest of the pack": None,
        "scikitlearn": "scikit-learn",
        "sh": "shell",
        "she'll": "shell",
        "simple documents": None,
        "synchronous tools (e.g.": None,
        "trello)": "trello",
        "used to deploy": None,
        "used to develop": None,
        "vim": "vi",
        "virtual machines (e.g.": None,
        "virtualbox)": "virtualbox",
        "visual studio code": "vscode",
        "vs code": "vscode",
        "web frameworks (e.g.": None,
        "writing": None,
        "wsl ": "wsl",
        "x86": "assembly",
    }.get(s, sentinel)

    if r is not sentinel:
        print("replacing", s, "with", r, file=sys.stderr)
        return r

    return s


def correlation(title, k1, k2, v1, v2, plot=False):
    kt, kp = map(lambda n: round(n, 3), kendalltau(v1, v2))
    sr, sp = map(lambda n: round(n, 3), spearmanr(v1, v2))

    m, b, lr, lp, _ = linregress(v1, v2)

    if nan(kt) or nan(sr):  # or nan(pr):
        return

    file = sys.stderr
    if plot or (
        max(kp, lp, sp) <= 0.012 and all(map(lambda x: x > 0.3, (kt, lr, sr)))
    ):
        file = sys.stdout

    y = [m * x + b for x in v1]
    rmsd = mean_squared_error(y, v2, squared=False)

    scores = ", ".join(
        [
            r"Kendall: %.3f (%.3f)" % (kt, kp),
            r"R-Squared: %.3f (%.3f)" % (lr, lp),
            r"Spearman: %.3f (%.3f)" % (sr, sp),
            r"RMSD: %.3f" % rmsd,
        ]
    )
    print("%s %s and %s have " % (scores, k1, k2), end="", file=file)
    print("%s correlation" % (strength(kt, lr, sr)), file=file)

    if file == sys.stderr:
        return

    for e in [
        ("Kendall", r"$\tau$"),
        ("Spearman", r"$\rho$"),
        ("R-Squared", r"$R^{2}$"),
    ]:
        scores = scores.replace(*e)

    plt.scatter(v1, v2)
    plt.title("\n".join([title, scores]))
    plt.xlabel(k1)
    plt.ylabel(k2)

    plt.plot(v1, y)

    plt.savefig(filename("img/", title, "-", k1, "-", k2, ".svg"), dpi=350)
    plt.close()


def correlations(title, k1, k2, k3, k4, k5, k6, v1, v2, v3, v4, v5, v6):
    uv4 = unit(v4)
    uv5 = unit(v5)
    uv6 = unit(v6)

    print("corr:", k1, ", ".join(map(str, v1)), file=sys.stderr)
    print("corr:", k2, ", ".join(map(str, v2)), file=sys.stderr)
    print("corr:", k3, ", ".join(map(str, v3)), file=sys.stderr)
    print(
        "corr:",
        k4,
        ", ".join(map(str, v4)),
        ", ".join(map(str, uv4)),
        file=sys.stderr,
    )
    print(
        "corr:",
        k5,
        ", ".join(map(str, v5)),
        ", ".join(map(str, uv5)),
        file=sys.stderr,
    )
    print(
        "corr:",
        k6,
        ", ".join(map(str, v6)),
        ", ".join(map(str, uv6)),
        file=sys.stderr,
    )

    correlation(title, k2, k1, v2, v1)
    correlation(title, k3, k1, v3, v1)
    correlation(title, k4, k1, uv4, v1)
    correlation(title, k5, k1, uv5, v1)
    correlation(title, k6, k1, uv6, v1)


def counts(display_name, points, prefix="", tally={}):
    for e in sorted(
        points.items(), key=lambda e: (-e[1], display(display_name, e[0]))
    ):
        n = tally.get(e[0], 0)
        print(
            prefix
            + display(display_name, e[0])
            + (" (" + str(n) + ")" if n else "")
            + ":",
            e[1],
        )

    print()


def display(display_name, name):
    return display_name.get(name, name)


def filename(*args):
    return "".join(args).replace(" ", "-")


def find(s, patterns, size=76):
    if s is None:
        return None

    for p in patterns:
        idx = s.lower().find(p)
        if idx != -1:
            if idx > 60:
                s = s[idx:]
            return s[:size]

    return None


def float_field(text):
    return float(text.split(" ")[0] or "0")


def gender(r):
    r = r.lower()
    if r == "male":
        return 0.0
    elif r == "female":
        return 1.0


def inc(d, k, v=1):
    d[k] = d.get(k, 0) + v


def list_field(text):
    return list(filter(None, [cleaned(s) for s in pattern.split(text)]))


def nan(n):
    return n != n


def normalized(prefs, k):
    return (len(prefs) + 1 - prefs.get(k, len(prefs) + 1)) / len(prefs)


def ranked(lst):
    rank = {}

    done = {}

    idx = 1
    for elem in lst:
        if done.get(elem, False):
            continue

        done[elem] = True
        rank[elem] = idx
        idx += 1

    return rank


def sentiment(row, terms):
    if find(row[label["awesome"]], terms):
        return 1.0
    elif find(row[label["awful"]], terms):
        return 0.0
    else:
        return 0.5


def size(n):
    n = abs(n)

    # From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6107969/table/tbl1/?report=objectonly
    # Quinnipiac University (Politics) but with 'none' replaced by 'no'.
    if n < 0.1:
        return "no"
    if n < 0.2:
        return "negligible"
    elif n < 0.3:
        return "weak"
    elif n < 0.4:
        return "moderate"
    elif n < 0.7:
        return "strong"
    elif n < 1.0:
        return "very strong"

    return "perfect"


def strength(*ns):
    neg = 0
    pos = 0

    maximum = 0.0
    minimum = 1.0

    for n in ns:
        if n <= 0.0:
            neg += 1
        if n >= 0.0:
            pos += 1

        # There are min and max functions but we're looping already.
        n = abs(n)
        if n < minimum:
            minimum = n
        if n > maximum:
            maximum = n

    if neg and pos:
        return "no"

    d = "positive"
    if neg:
        d = "negative"

    max_size = size(maximum)

    s = size(minimum)
    if max_size != s:
        s = " to ".join((s, max_size))

    return "a " + " ".join((s, d))


def unique(*kss):
    s = set()
    for ks in kss:
        for k in ks:
            s.add(k)
    return s


def unit(lst):
    maximum = max(lst)
    minimum = min(lst)
    r = float(maximum - minimum)

    return [(float(e) - float(minimum)) / r for e in lst]


def which(s, patterns):
    for p in patterns:
        if s.lower().find(p) != -1:
            return p

    return "other"


def languages(rows):
    print("languages:\n")

    combined_pref = {}
    combined_used = {}

    display_name = {
        "assembly": "Assembly",
        "awk": "awk",
        "brainfuck": "Brainfuck",
        "c": "C",
        "c#": "C#",
        "c++": "C++",
        "crystal": "Crystal",
        "css": "CSS",
        "go": "Go",
        "haskell": "Haskell",
        "helm": "Helm",
        "html": "HTML",
        "java": "Java",
        "javascript": "JavaScript",
        "kotlin": "Kotlin",
        "latex": "LaTeX",
        "lisp": "Lisp",
        "lua": "Lua",
        "markdown": "Markdown",
        "matlab": "MATLAB",
        "ocaml": "OCaml",
        "octave": "Octave",
        "perl": "Perl",
        "php": "PHP",
        "powershell": "PowerShell",
        "python": "Python",
        "r": "R",
        "racket": "Racket",
        "ruby": "Ruby",
        "rust": "Rust",
        "scala": "Scala",
        "shell": "Shell",
        "smalltalk": "Smalltalk",
        "sql": "SQL",
        "typescript": "TypeScript",
        "verilog": "Verilog",
        "vhdl": "VHDL",
        "visual basic": "Visual Basic",
        "xml": "XML",
    }

    most_pref = {}
    most_used = {}
    name_count = {}
    used_vs_pref = {}

    ages = []
    genders = []
    academic_exp = []
    personal_exp = []
    professional_exp = []

    langs = sorted(display_name.keys())

    for lang in langs:
        combined_pref[lang] = []
        combined_used[lang] = []

    print("\"unhappy\":")

    for row in rows:
        pref = list_field(row[label["languages_by_pref"]])
        used = list_field(row[label["languages_by_use"]])

        if len(pref) == 0:
            continue

        print("language preferences:", " ".join(pref), file=sys.stderr)

        if pref[0] == "same as above":
            pref = used

        for i in range(len(pref) - 1):
            n1 = display_name[pref[i]]
            n2 = display_name[pref[i + 1]]
            print('pref graph: "%s" -> "%s"' % (n1, n2), file=sys.stderr)

        for i in range(len(used) - 1):
            n1 = display_name[used[i]]
            n2 = display_name[used[i + 1]]
            print('used graph: "%s" -> "%s"' % (n1, n2), file=sys.stderr)

        inc(most_pref, pref[0])
        inc(most_used, used[0])

        pref_ranked = ranked(pref)
        used_ranked = ranked(used)

        for lang in langs:
            combined_pref[lang].append(normalized(pref_ranked, lang))
            combined_used[lang].append(normalized(used_ranked, lang))

        ages.append(age(row["Age"] or row["age2"]))
        genders.append(gender(row["Gender"] or row["gender2"]))
        academic_exp.append(float_field(row[label["academic_exp"]]))
        personal_exp.append(float_field(row[label["personal_exp"]]))
        professional_exp.append(float_field(row[label["professional_exp"]]))

        print("\nused vs pref:", file=sys.stderr)
        print(", ".join(pref), file=sys.stderr)
        print(", ".join(used), file=sys.stderr)

        names = unique(pref_ranked.keys(), used_ranked.keys())
        for name in names:
            inc(name_count, name)

            used_rank = used_ranked.get(name, len(used) + 1)
            pref_rank = pref_ranked.get(name, max(used_rank, len(pref) + 1))

            delta = pref_rank - used_rank
            dist = abs(delta)
            print(
                "lang delta:",
                name,
                used_rank,
                pref_rank,
                delta,
                delta * dist,
                row[label["languages_by_use"]],
                "vs",
                row[label["languages_by_pref"]],
                file=sys.stderr,
            )
            if delta:
                inc(used_vs_pref, name, delta * dist)

        print(file=sys.stderr)

        if pref[0] not in used[:3]:
            print("   ", pref[0], "vs", ", ".join(used))

    print()

    for k, v in name_count.items():
        if k in used_vs_pref:
            if v < 3:
                del used_vs_pref[k]
            else:
                used_vs_pref[k] = round(used_vs_pref[k] / v, 3)

    print("most mentioned:")
    counts(display_name, name_count, "    ")

    print("most preferred:")
    counts(display_name, most_pref, "    ")

    print("most used:")
    counts(display_name, most_used, "    ")

    print("used vs preferred:")
    counts(display_name, used_vs_pref, "    ", name_count)

    print("age and experience correlations:")
    correlation(
        "Age and Experience",
        "Age",
        "Academic",
        ages,
        unit(academic_exp),
        plot=True,
    )
    correlation(
        "Age and Experience", "Age", "Personal", ages, unit(personal_exp)
    )
    correlation(
        "Age and Experience",
        "Age",
        "Professional",
        ages,
        unit(professional_exp),
    )
    print()

    print("experience correlations:")
    correlation(
        "Experience",
        "Academic",
        "Personal",
        unit(academic_exp),
        unit(personal_exp),
    )
    correlation(
        "Experience",
        "Academic",
        "Professional",
        unit(academic_exp),
        unit(professional_exp),
    )
    correlation(
        "Experience",
        "Personal",
        "Professional",
        unit(personal_exp),
        unit(professional_exp),
    )
    print()

    print("language preference and demographic/experience correlations:")
    for lang in langs:
        if len(list(filter(None, combined_pref[lang]))) < min_freq:
            continue

        print("popular language:", lang, file=sys.stderr)

        correlations(
            "Language Preference and Demographic",
            display_name[lang],
            "Age",
            "Gender",
            "Academic",
            "Personal",
            "Professional",
            combined_pref[lang],
            ages,
            genders,
            academic_exp,
            personal_exp,
            professional_exp,
        )

    print()

    print("language use and demographic/experience correlations:")
    for lang in langs:
        if len(list(filter(None, combined_used[lang]))) < min_freq:
            continue

        print("used language:", lang, file=sys.stderr)

        correlations(
            "Language Use and Demographic",
            display_name[lang],
            "Age",
            "Gender",
            "Academic",
            "Personal",
            "Professional",
            combined_used[lang],
            ages,
            genders,
            academic_exp,
            personal_exp,
            professional_exp,
        )

    print()

    print("language preference correlations:")
    for i in range(len(langs) - 1):
        l1 = langs[i]
        if len(list(filter(None, combined_pref[l1]))) < min_freq:
            continue

        for j in range(i + 1, len(langs)):
            l2 = langs[j]
            if len(list(filter(None, combined_pref[l2]))) < min_freq:
                continue

            correlation(
                "Language Preference",
                display_name[l1],
                display_name[l2],
                combined_pref[l1],
                combined_pref[l2],
            )

    print()

    print("language use correlations:")
    for i in range(len(langs) - 1):
        l1 = langs[i]
        if len(list(filter(None, combined_used[l1]))) < min_freq:
            continue

        for j in range(i + 1, len(langs)):
            l2 = langs[j]
            if len(list(filter(None, combined_used[l2]))) < min_freq:
                continue

            correlation(
                "Language Use",
                display_name[l1],
                display_name[l2],
                combined_used[l1],
                combined_used[l2],
            )

    print()


def platforms(rows):
    print("platforms:\n")

    combined_pref = {}
    combined_used = {}

    defeats = [
        "macos and windows (same preference)",
        "it depends on the purpose. for a server",
    ]

    display_name = {
        "aix": "AIX",
        "android": "Android",
        "bsd": "BSD",
        "linux": "Linux",
        "macos": "macOS",
        "windows": "Windows",
        "web": "Web",
        "wsl": "WSL",
    }

    most_pref = {"linux": 0.5, "macos": 1, "windows": 0.5}
    most_used = {"linux": 0.5, "macos": 1, "windows": 0.5}
    name_count = {}

    ages = []
    genders = []
    academic_exp = []
    personal_exp = []
    professional_exp = []

    pfs = sorted(display_name.keys())

    for pf in pfs:
        combined_pref[pf] = []
        combined_used[pf] = []

    print("\"unhappy\":")

    for row in rows:
        pref = list_field(row[label["platforms_by_pref"]])
        used = list_field(row[label["platforms_by_use"]])

        if pref[0] == "same as above":
            pref = used

        if pref[0] in defeats:
            continue

        for name in unique(pref, used):
            inc(name_count, name)

        inc(most_pref, pref[0])
        inc(most_used, used[0])

        for pf in pfs:
            combined_pref[pf].append(normalized(ranked(pref), pf))
            combined_used[pf].append(normalized(ranked(used), pf))

        ages.append(age(row["Age"] or row["age2"]))
        genders.append(gender(row["Gender"] or row["gender2"]))
        academic_exp.append(float_field(row[label["academic_exp"]]))
        personal_exp.append(float_field(row[label["personal_exp"]]))
        professional_exp.append(float_field(row[label["professional_exp"]]))

        if pref[0] == used[0]:
            continue

        print("   ", pref[0], "vs", ", ".join(used))

    print()

    print("most mentioned:")
    counts(display_name, name_count, "    ")

    print("most preferred:")
    counts(display_name, most_pref, "    ")

    print("most used:")
    counts(display_name, most_used, "    ")

    print("platform preference and demographic/experience correlations:")
    for pf in pfs:
        correlations(
            "Platform Preference",
            display_name[pf],
            "Age",
            "Gender",
            "Academic",
            "Personal",
            "Professional",
            combined_pref[pf],
            ages,
            genders,
            academic_exp,
            personal_exp,
            professional_exp,
        )

    print()

    print("platform use and demographic/experience correlations:")
    for pf in pfs:
        correlations(
            "Platform Use",
            display_name[pf],
            "Age",
            "Gender",
            "Academic",
            "Personal",
            "Professional",
            combined_used[pf],
            ages,
            genders,
            academic_exp,
            personal_exp,
            professional_exp,
        )

    print()


def term(rows, terms):
    print(terms[0] + ":")

    points = {}

    for row in rows:
        for item in row.items():
            text = find(item[1], terms)
            if text:
                print("    " + item[0] + ":", file=sys.stderr)
                print("    " + text, file=sys.stderr)
                print(file=sys.stderr)

                category = which(
                    item[0], ["awesome", "awful", "preferred", "used"]
                )
                points[category] = points.get(category, 0) + 1

    counts({}, points, "    ")


def tools(rows):
    print("tools:\n")

    combined_pref = {}
    combined_used = {}

    display_name = {
        "adobe illustrator": "Adobe Illustrator",
        "anaconda": "Anaconda",
        "ant": "And",
        "apollo client": "Apollo Client",
        "atlassian": "Atlassian",
        "azure": "Azure",
        "cmake": "CMake",
        "colab": "Google Colab",
        "discord": "Discord",
        "docker": "Docker",
        "dockerhub": "Docker Hub",
        "eclipse": "Eclipse",
        "gcc": "GCC",
        "gdb": "GDB",
        "git": "git",
        "github": "GitHub",
        "gitlab": "GitLab",
        "google cloud": "Google Cloud",
        "googledocs": "Google Docs",
        "ida pro": "IDA Pro",
        "ide": "IDE",
        "jenkins": "Jenkins",
        "jetbrains": "JetBrains",
        "jupyter": "Jupyter Notebooks",
        "kubernetes": "Kubernetes",
        "liquibase": "Liquibase",
        "llvm": "LLVM",
        "make": "make",
        "maven": "Maven",
        "microsoft word": "MS Word",
        "miro": "Miro",
        "mode (sql)": "Mode",
        "nginx": "Nginx",
        "notepad": "Notepad",
        "notepad++": "Notepad++",
        "omnigraffle": "Omnigraffle",
        "overleaf": "Overleaf",
        "pharo": "Pharo",
        "postman": "Postman",
        "powerbi": "Power BI",
        "puppet": "Puppet",
        "python": "Python",
        "rstudio": "RStudio",
        "ruby": "Ruby",
        "scout-app": "Scout-App",
        "spyder": "Spyder",
        "svn": "Subversion",
        "terminal": "Terminal",
        "terminator": "Terminator",
        "testrail": "TestRail",
        "trello": "Trello",
        "unity 3d": "Unity",
        "valgrind": "Valgrind",
        "vhdl": "VHDL",
        "vi": "vi",
        "virtualbox": "VirtualBox",
        "visual studio": "Visual Studio",
        "vmware": "VMware",
        "vscode": "Visual Studio Code",
        "webratio": "WebRatio",
        "windbg": "WinDbg",
        "windows": "Windows",
        "wsl": "WSL",
        "zsh": "Z Shell",
    }

    most_pref = {}
    most_used = {}
    name_count = {}
    used_vs_pref = {}

    ages = []
    genders = []
    academic_exp = []
    personal_exp = []
    professional_exp = []

    tls = sorted(display_name.keys())

    for tl in tls:
        combined_pref[tl] = []
        combined_used[tl] = []

    for row in rows:
        pref = list_field(row[label["tools_by_pref"]])
        used = list_field(row[label["tools_by_use"]])

        other_pref = list_field(row[label["other_by_pref"]])
        other_used = list_field(row[label["other_by_use"]])
        if len(other_pref) > 0 and other_pref[0] == "same as above":
            other_pref = other_used

        # A few participants reported tools as other. Move to tools.
        special = row["special"] or ""

        # In one case, append a missing preference that results in the
        # same score and ensures that extending both lists doesn't result
        # in elements in one list appearing one rank higher.
        if special.startswith("append"):
            print("appending", file=sys.stderr)
            pref.append("vhdl")

        # Extend any reported tools with other.
        if special.endswith("extend"):
            print("extending", file=sys.stderr)
            pref.extend(other_pref)
            used.extend(other_used)

        if len(pref) == 0 or len(used) == 0:
            continue

        if pref[0] == "same as above":
            pref = used

        for t in pref:
            print("tool:", t, file=sys.stderr)
        for t in used:
            print("tool:", t, file=sys.stderr)

        inc(most_pref, pref[0])
        inc(most_used, used[0])

        if pref[0] == "python":
            print("Language in tools:", row, file=sys.stderr)

        pref_ranked = ranked(pref)
        used_ranked = ranked(used)

        new = {}
        for tl in tls:
            if new.get(tl, True):
                combined_pref[tl].append(normalized(pref_ranked, tl))
                combined_used[tl].append(normalized(used_ranked, tl))
                new[tl] = False

        ages.append(age(row["Age"] or row["age2"]))
        genders.append(gender(row["Gender"] or row["gender2"]))
        academic_exp.append(float_field(row[label["academic_exp"]]))
        personal_exp.append(float_field(row[label["personal_exp"]]))
        professional_exp.append(float_field(row[label["professional_exp"]]))

        done = {}
        names = unique(pref_ranked.keys(), used_ranked.keys())
        for name in names:
            if done.get(name, False):
                continue

            new[name] = True

            inc(name_count, name)

            used_rank = used_ranked.get(name, len(used) + 1)
            pref_rank = pref_ranked.get(name, max(used_rank, len(pref) + 1))

            delta = used_rank - pref_rank
            dist = abs(delta)

            print(
                "tool delta:",
                name,
                used_rank,
                pref_rank,
                delta,
                delta * dist,
                file=sys.stderr,
            )

            if delta:
                inc(used_vs_pref, name, delta * dist)

    for k, v in name_count.items():
        if k in used_vs_pref:
            if v < min_freq:
                del used_vs_pref[k]
            else:
                used_vs_pref[k] = round(used_vs_pref[k] / v, 3)

    nc = {}
    for k in used_vs_pref.keys():
        nc[k] = name_count[k]
    name_count = nc

    print("most mentioned:")
    counts(display_name, name_count, "    ")

    print("most preferred:")
    counts(display_name, most_pref, "    ")

    print("most used:")
    counts(display_name, most_used, "    ")

    print("used vs preferred:")
    counts(display_name, used_vs_pref, "    ", name_count)

    # Combine popular IDEs to check demographic/experience correlations.
    combined_pref["ide"] = [
        min(e)
        for e in zip(combined_pref["jetbrains"], combined_pref["vscode"])
    ]
    combined_used["ide"] = [
        min(e)
        for e in zip(combined_used["jetbrains"], combined_used["vscode"])
    ]

    print("tool preference and demographic/experience correlations:")
    for tl in tls:
        if len(list(filter(None, combined_pref[tl]))) < min_freq:
            continue

        correlations(
            "Tool Preference and Demographic",
            display_name[tl],
            "Age",
            "Gender",
            "Academic",
            "Personal",
            "Professional",
            combined_pref[tl],
            ages,
            genders,
            academic_exp,
            personal_exp,
            professional_exp,
        )

    print()

    print("tool use and demographic/experience correlations:")
    for tl in tls:
        if len(list(filter(None, combined_used[tl]))) < min_freq:
            continue

        correlations(
            "Tool Use and Demographic",
            display_name[tl],
            "Age",
            "Genders",
            "Academic",
            "Personal",
            "Professional",
            combined_used[tl],
            ages,
            genders,
            academic_exp,
            personal_exp,
            professional_exp,
        )

    print()

    print("tool preference correlations:")
    for i in range(len(tls) - 1):
        t1 = tls[i]

        # Ignore our synthetic addition.
        if t1 == "ide":
            continue

        if len(list(filter(None, combined_pref[t1]))) < min_freq:
            continue

        for j in range(i + 1, len(tls)):
            t2 = tls[j]
            if len(list(filter(None, combined_pref[t2]))) < min_freq:
                continue

            correlation(
                "Tool Preference",
                display_name[t1],
                display_name[t2],
                combined_pref[t1],
                combined_pref[t2],
            )

    print()

    print("tool use correlations:")
    for i in range(len(tls) - 1):
        t1 = tls[i]

        # Ignore our synthetic addition.
        if t1 == "ide":
            continue

        if len(list(filter(None, combined_used[t1]))) < min_freq:
            continue

        for j in range(i + 1, len(tls)):
            t2 = tls[j]
            if len(list(filter(None, combined_used[t2]))) < min_freq:
                continue

            correlation(
                "Tool Use",
                display_name[t1],
                display_name[t2],
                combined_used[t1],
                combined_used[t2],
            )

    print()


print("responses:")
print(len(rows))
print()

languages(rows)
platforms(rows)
tools(rows)

print("specific terms:")
print()

term(rows, ["browser", "chrome", "firefox", "edge"])
term(rows, ["docker", "containerization"])
term(rows, ["git"])
term(rows, ["kubernetes", "k8s"])
term(rows, ["terminal"])

print("done")
