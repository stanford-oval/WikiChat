# Converted from wtf_wikipedia JavaScript package at https://github.com/spencermountain/wtf_wikipedia/tree/1e875e020df12fdb02731bbbf7b68337c4016a04/src/01-document
#  wikipedia special terms lifted and augmented from parsoid parser april 2015
#  and then manually on March 2020

# The redirect detection code in wtf_wikipedia does not work if applied to the wikitext field of Wikipedia HTML dumps.
# This is because the HTML dumps contain the wikitext of the page after redirection is applied, meaning that they look like a replica of the non-redirect page.

import re

# used in titles to denote disambiguation pages
# e.g. 'Football_(disambiguation)'

disambiguation_indicator_in_titles = (
    [
        "disambigua",  # it
        "消歧義",  # zh
        "значения",  # ru
        "dubbelsinnig",  # af
        "այլ կիրառումներ",  # hy
        "ujednoznacznienie",  # pl
    ]
    + [  # These are obtained from translations of https://en.wikipedia.org/wiki/Wikipedia:Disambiguation in 114 languages (then deduplicated). Might not be actually what is used in individual article's titles
        "Dubbelsinnigheid",
        "Pachina de desambigación",
        "توضيح",
        "দ্ব্যৰ্থতা দূৰীকৰণ",
        "Páxina de dixebra",
        "Dəqiqləşdirmə",
        "دقیقلشدیرمه",
        "Күп мәғәнәлелек",
        "Begriffsklearung",
        "Неадназначнасць",
        "Неадназначнасьць",
        "Пояснителна страница",
        "बहुअर्थी-शब्द स्पष्टीकरण",
        "দ্ব্যর্থতা নিরসন",
        "Disheñvelout",
        "Pàgina de desambiguació",
        "Чолхалла",
        "Mga pulong nga may labaw pa sa usa ka kahulogan",
        "ڕوونکردنەوە",
        "Rozcestníky",
        "Gwahaniaethu",
        "Flertydige titler",
        "Begriffsklärung",
        "Αποσαφήνιση",
        "Ambigüedad en títulos",
        "Täpsustuslehekülg",
        "Argipen orri",
        "ابهام‌زدایی",
        "Täsmennyssivu",
        "Homonymie",
        "Homónimos",
        "સંદિગ્ધ શીર્ષક",
        "Reddaghey",
        "פירושונים",
        "बहुविकल्पी शब्द",
        "Razdvojba",
        "Wjacezmyslnosć",
        "Egyértelműsítő lapok",
        "Բազմիմաստության փարատում",
        "Disambiguation",
        "Disambiguasi",
        "Panangilawlawag",
        "Массехк маӀан хилар",
        "Homonimo",
        "Aðgreiningarsíður",
        "曖昧さ回避",
        "Dhisambiguasi",
        "მრავალმნიშვნელოვანი",
        "Айрық",
        "អសង្ស័យកម្ម",
        "ದ್ವಂದ್ವ ನಿವಾರಣೆ",
        "동음이의어 문서",
        "Watt ėßß mėd „(Watt ėßß datt?)“-Sigge?",
        "Discretiva",
        "Homonymie",
        "Verdudelikingspazjena",
        "Nuorodiniai straipsniai",
        "Nozīmju atdalīšana",
        "Disambiguasi",
        "Лама смусть",
        "Појаснување",
        "വിവക്ഷകൾ",
        "Disambiguation",
        "Nyahkekaburan",
        "Zambiguaçon",
        "گجگجی بیتن",
        "Mehrdüdig Begreep",
        "अस्पष्ट पृष्ठहरू",
        "Doorverwijspagina",
        "Fleirtyding",
        "Flertydige titler",
        "Frouque",
        "Strona ujednoznaczniająca",
        "Ròbe parej",
        "څومانيز",
        "Desambiguação",
        "Dudalipen",
        "Dezambiguizare",
        "Неоднозначность",
        "Disambiguazzioni",
        "سلجھائپ",
        "Razvrstavanje",
        "බහුරුත්හරණය",
        "Disambiguation",
        "Rozlišovacia stránka",
        "Razločitev",
        "Kthjellime",
        "Вишезначна одредница",
        "Begriepskloorenge",
        "Disambiguasi",
        "Särskiljning",
        "Ujednoznaczńajůnco zajta",
        "பக்கவழி நெறிப்படுத்தல்",
        "అయోమయ నివృత్తి",
        "Ибҳомзудоӣ",
        "การแก้ความกำกวม",
        "Paglilinaw",
        "Anlam ayrımı",
        "Күп мәгънәле мәкаләләр",
        "Омонимия",
        "Неоднозначність",
        "ضد ابہام",
        "Định hướng",
        "Telplänov",
        "Omonimeye",
        "分清爽",
        "באדייטן",
        "Deurverwiespagina",
        "消歧义",
        "指津辨義",
        "Khu-pia̍t",
        "搞清楚",
    ]
)
disambiguation_indicator_in_titles = [
    i.lower() for i in disambiguation_indicator_in_titles
]  # disambiguation indicators in titles are often lowercase
disambiguation_indicator_in_titles = list(
    set(disambiguation_indicator_in_titles)
)  # deduplicate

multilingual_disambiguation_templates = {
    "dab",  # en
    "disamb",  # en
    "disambig",  # en
    "disambiguation",  # en
    "aðgreining",
    "aðgreining",  # is
    "aimai",  # ja
    "airport disambiguation",
    "ałtsʼáʼáztiin",  # nv
    "anlam ayrımı",  # gag
    "anlam ayrımı",  # tr
    "apartigilo",  # eo
    "argipen",  # eu
    "begriepskloorenge",  # stq
    "begriffsklärung",  # als
    "begriffsklärung",  # de
    "begriffsklärung",  # pdc
    "begriffsklearung",  # bar
    "biology disambiguation",
    "bisongidila",  # kg
    "bkl",  # pfl
    "bokokani",  # ln
    "caddayn",  # so
    "call sign disambiguation",
    "caselaw disambiguation",
    "chinese title disambiguation",
    "clerheans",  # kw
    "cudakirin",  # ku
    "čvor",  # bs
    "db",  # vls
    "desambig",  # nov
    "desambigación",  # an
    "desambiguação",  # pt
    "desambiguació",  # ca
    "desambiguación",  # es
    "desambiguáncia",  # ext
    "desambiguasion",  # lad
    "desambiguassiù",  # lmo
    "desambigui",  # lfn
    "dezambiguizare",  # ro
    "dezanbìgua",
    "dəqiqləşdirmə",
    "dəqiqləşdirmə",  # az
    "disamb-term",
    "disamb-terms",
    "disamb2",
    "disamb3",
    "disamb4",
    "disambigua",  # it
    "disambìgua",  # sc
    "disambiguasi",
    "disambiguation cleanup",
    "disambiguation lead name",
    "disambiguation lead",
    "disambiguation name",
    "disambiguazion",
    "disambigue",
    "discretiva",
    "discretiva",  # la
    "disheñvelout",  # br
    "disingkek",  # min
    "dixanbigua",  # vec
    "dixebra",  # ast
    "diżambigwazzjoni",  # mt
    "dmbox",
    "doorverwijspagina",  # nl
    "dp",  # nl
    "dubbelsinnig",
    "dubbelsinnig",  # af
    "dudalipen",  # rmy
    "dv",  # nds_nl
    "egyért",  # hu
    "faaleaogaina",
    "fleiri týdningar",  # fo
    "fleirtyding",  # nn
    "flertydig",  # da
    "förgrening",  # sv
    "genus disambiguation",
    "gì-ngiê",  # cdo
    "giklaro",  # ceb
    "gwahaniaethu",  # cy
    "homonimo",  # io
    "homónimos",  # gl
    "homonymie",  # fr
    "hospital disambiguation",
    "huaʻōlelo puana like",
    "huaʻōlelo puana like",  # haw
    "human name disambiguation cleanup",
    "human name disambiguation",
    "idirdhealú",  # ga
    "khu-pia̍t",  # zh_min_nan
    "kthjellim",  # sq
    "kujekesa",  # sn
    "letter-number combination disambiguation",
    "letter-numbercombdisambig",
    "maana",  # sw
    "maneo bin",  # diq
    "mathematical disambiguation",
    "mehrdüdig begreep",  # nds
    "menm non",  # ht
    "military unit disambiguation",
    "muardüüdag artiikel",  # frr
    "music disambiguation",
    "myesakãrã",
    "neibetsjuttings",  # fy
    "nozīmju atdalīšana",  # lv
    "number disambiguation",
    "nuorodinis",  # lt
    "nyahkekaburan",  # ms
    "omonimeye",  # wa
    "omonimi",
    "omonimia",  # oc
    "opus number disambiguation",
    "page dé frouque",  # nrm
    "paglilinaw",  # tl
    "panangilawlawag",  # ilo
    "pansayod",  # war
    "pejy mitovy anarana",  # mg
    "peker",  # no
    "phonetics disambiguation",
    "place name disambiguation",
    "portal disambiguation",
    "razdvojba",  # hr
    "razločitev",  # sl
    "razvrstavanje",  # sh
    "reddaghey",  # gv
    "road disambiguation",
    "rozcestník",  # cs
    "rozlišovacia stránka",  # sk
    "school disambiguation",
    "sclerir noziun",  # rm
    "selvendyssivu",  # olo
    "soilleireachadh",  # gd
    "species latin name abbreviation disambiguation",
    "species latin name disambiguation",
    "station disambiguation",
    "suzmunski",  # jbo
    "synagogue disambiguation",
    "täpsustuslehekülg",  # et
    "täsmennyssivu",  # fi
    "taxonomic authority disambiguation",
    "taxonomy disambiguation",
    "telplänov",  # vo
    "template disambiguation",
    "tlahtolmelahuacatlaliztli",  # nah
    "trang định hướng",  # vi
    "ujednoznacznienie",  # pl
    "verdudeliking",  # li
    "wěcejwóznamowosć",  # dsb
    "wjacezmyslnosć",  # hsb
    "z",  # vep
    "zambiguaçon",  # mwl
    "zeimeibu škiršona",  # ltg
    "αποσαφήνιση",  # el
    "айрық",  # kk
    "аҵакырацәа",  # ab
    "бир аайы јок",
    "вишезначна одредница",  # sr
    "ибҳомзудоӣ",  # tg
    "кёб магъаналы",  # krc
    "күп мәгънәләр",  # tt
    "күп мәғәнәлелек",  # ba
    "массехк маӏан хилар",
    "мъногосъмꙑслиѥ",  # cu
    "неадназначнасць",  # be
    "неадназначнасьць",  # be_x_old
    "неоднозначность",  # ru
    "олон удхатай",  # bxr
    "појаснување",  # mk
    "пояснение",  # bg
    "са шумуд манавал",  # lez
    "салаа утгатай",  # mn
    "суолталар",  # sah
    "текмаанисиздик",  # ky
    "цо магіна гуреб",  # av
    "чеперушка",  # rue
    "чолхалла",  # ce
    "шуко ончыктымаш-влак",  # mhr
    "მრავალმნიშვნელოვანი",  # ka
    "բազմիմաստութիւն",  # hyw
    "բազմիմաստություն",  # hy
    "באדייטן",  # yi
    "פירושונים",  # he
    "ابهام‌زدایی",  # fa
    "توضيح",  # ar
    "توضيح",  # arz
    "دقیقلشدیرمه",  # azb
    "ڕوونکردنەوە",  # ckb
    "سلجهائپ",  # sd
    "ضد ابہام",  # ur
    "گجگجی بیری",  # mzn
    "نامبهمېدنه",  # ps
    "መንታ",  # am
    "अस्पष्टता",  # ne
    "बहुअर्थी",  # bh
    "बहुविकल्पी शब्द",  # hi
    "দ্ব্যর্থতা নিরসন",  # bn
    "ਗੁੰਝਲ-ਖੋਲ੍ਹ",  # pa
    "સંદિગ્ધ શીર્ષક",  # gu
    "பக்கவழி நெறிப்படுத்தல்",  # ta
    "అయోమయ నివృత్తి",  # te
    "ದ್ವಂದ್ವ ನಿವಾರಣೆ",  # kn
    "വിവക്ഷകൾ",  # ml
    "වක්‍රෝත්ති",  # si
    "แก้ความกำกวม",  # th
    "သံတူကြောင်းကွဲ",  # my
    "သဵင်မိူၼ် တူၼ်ႈထႅဝ်ပႅၵ်ႇ",  # shn
    "ណែនាំ",  # km
    "អសង្ស័យកម្ម",
    "동음이의",  # ko
    "扤清楚",  # gan
    "搞清楚",  # zh_yue
    "曖昧さ回避",  # ja
    "消歧义",  # zh
    "釋義",  # zh_classical
    "gestion dj'omònim",  # pms
    "sut'ichana qillqa",  # qu
    "gestion dj'omònim",
    "sut'ichana qillqa",
}

# alternative disambiguation templates that the English wikipedia uses
d = " disambiguation"
english_disambiguation_templates = {
    "dab",
    "dab",
    "disamb",
    "disambig",
    "geodis",
    "hndis",
    "setindex",
    "ship index",
    "split dab",
    "sport index",
    "wp disambig",
    "disambiguation cleanup",
    "airport" + d,
    "biology" + d,
    "call sign" + d,
    "caselaw" + d,
    "chinese title" + d,
    "genus" + d,
    "hospital" + d,
    "lake index",
    "letter" + d,
    "letter-number combination" + d,
    "mathematical" + d,
    "military unit" + d,
    "mountainindex",
    "number" + d,
    "phonetics" + d,
    "place name" + d,
    "portal" + d,
    "road" + d,
    "school" + d,
    "species latin name abbreviation" + d,
    "species latin name" + d,
    "station" + d,
    "synagogue" + d,
    "taxonomic authority" + d,
    "taxonomy" + d,
}

all_disambiguation_templates = set()
all_disambiguation_templates.update(multilingual_disambiguation_templates)
all_disambiguation_templates.update(english_disambiguation_templates)

# templates that signal page is not a disambiguation
not_disambiguation_templates = {
    "about",
    "for",
    "for multi",
    "other people",
    "other uses of",
    "distinguish",
}

# TODO add more languages to this regex
may_also_refer_to_regex = re.compile(r". may (also )?refer to\b", re.IGNORECASE)
# Regex to detect disambiguation indicators in the title
in_title_regex = re.compile(
    r". \((" + "|".join(disambiguation_indicator_in_titles) + r")\)$", re.IGNORECASE
)


def is_disambiguation(article: dict) -> bool:
    # check for a {{disambig}} template
    article_templates = set(
        [
            template["name"].split(":")[-1]
            for template in article.get("templates", [])
            if "name" in template
        ]
    )
    if len(article_templates.intersection(all_disambiguation_templates)) > 0:
        return True
    # check for (disambiguation) in title
    title = article["name"]
    if bool(in_title_regex.match(title)):
        return True

    # does it have a non-disambig template?
    if article_templates.intersection(not_disambiguation_templates):
        return False

    # try 'may refer to' on first line for en-wiki?
    if "wikitext" in article and bool(
        may_also_refer_to_regex.match(article["wikitext"][:100])
    ):
        return True

    return False
