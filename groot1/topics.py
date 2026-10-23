"""Topic-classification data with label-space diversification, for zero-shot generalization.

    python -m groot1.topics build --out data/topics.jsonl
    python -m groot1.topics diversify --rows mix/train.jsonl --sources banking77,clinc150,massive --out div.jsonl

Groot 1 S v0.1 learned label priors instead of reading options: on unseen news taxonomies it
answered "world" for most articles. Every row built here randomizes how the label set is
presented, so the only stable signal is the meaning of each option:

* a random subset of the labels (gold always kept), so label sets vary from row to row
* label keys in English, in Portuguese, bare, or replaced by neutral ids that keep only the description
* descriptions in English or Portuguese, questions from a pool of templates in both languages
* states as plain text or as a small JSON object with English or Portuguese keys

No Folha de S.Paulo data is used (including the IARA corpus, which is Folha): Folha is the held-out
zero-shot benchmark. Portuguese news comes from CNN Brasil and BBC Brasil.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

from .data import make_row

# key -> (English description, Portuguese key, Portuguese description)
DBPEDIA = {
    "Company": ("A business or company", "empresa", "Uma empresa ou negócio"),
    "EducationalInstitution": ("A school, university or other educational institution", "instituicao_de_ensino", "Escola, universidade ou outra instituição de ensino"),
    "Artist": ("An artist, musician, writer or performer", "artista", "Artista, músico, escritor ou intérprete"),
    "Athlete": ("An athlete or sports player", "atleta", "Atleta ou esportista"),
    "OfficeHolder": ("A politician or holder of public office", "politico", "Político ou ocupante de cargo público"),
    "MeanOfTransportation": ("A vehicle, ship, aircraft or other means of transportation", "meio_de_transporte", "Veículo, navio, avião ou outro meio de transporte"),
    "Building": ("A building or structure", "edificio", "Edifício ou construção"),
    "NaturalPlace": ("A natural place such as a mountain, river or lake", "lugar_natural", "Lugar natural como montanha, rio ou lago"),
    "Village": ("A village or small settlement", "vilarejo", "Vilarejo ou pequena localidade"),
    "Animal": ("An animal species", "animal", "Uma espécie animal"),
    "Plant": ("A plant species", "planta", "Uma espécie de planta"),
    "Album": ("A music album", "album", "Álbum musical"),
    "Film": ("A film or movie", "filme", "Filme"),
    "WrittenWork": ("A book, magazine or other written work", "obra_escrita", "Livro, revista ou outra obra escrita"),
}
YAHOO = {
    "Society & Culture": ("Society, culture, traditions and religion", "sociedade_e_cultura", "Sociedade, cultura, tradições e religião"),
    "Science & Mathematics": ("Science and mathematics", "ciencia_e_matematica", "Ciência e matemática"),
    "Health": ("Health, medicine and the body", "saude", "Saúde, medicina e corpo humano"),
    "Education & Reference": ("Education, school and reference questions", "educacao", "Educação, escola e referência"),
    "Computers & Internet": ("Computers, software and the internet", "computadores_e_internet", "Computadores, software e internet"),
    "Sports": ("Sports and athletes", "esportes", "Esportes e atletas"),
    "Business & Finance": ("Business, money and finance", "negocios_e_financas", "Negócios, dinheiro e finanças"),
    "Entertainment & Music": ("Entertainment, music, movies and TV", "entretenimento_e_musica", "Entretenimento, música, filmes e TV"),
    "Family & Relationships": ("Family, dating and relationships", "familia_e_relacionamentos", "Família, namoro e relacionamentos"),
    "Politics & Government": ("Politics, government and law", "politica_e_governo", "Política, governo e leis"),
}
HUFFPOST_MAP = {
    "POLITICS": "POLITICS", "WELLNESS": "WELLNESS", "HEALTHY LIVING": "WELLNESS", "ENTERTAINMENT": "ENTERTAINMENT",
    "TRAVEL": "TRAVEL", "STYLE & BEAUTY": "STYLE & BEAUTY", "STYLE": "STYLE & BEAUTY", "PARENTING": "PARENTING",
    "PARENTS": "PARENTING", "FOOD & DRINK": "FOOD & DRINK", "TASTE": "FOOD & DRINK", "BUSINESS": "BUSINESS",
    "SPORTS": "SPORTS", "HOME & LIVING": "HOME & LIVING", "WEDDINGS": "WEDDINGS", "CRIME": "CRIME", "DIVORCE": "DIVORCE",
    "WORLD NEWS": "WORLD NEWS", "THE WORLDPOST": "WORLD NEWS", "WORLDPOST": "WORLD NEWS", "MEDIA": "MEDIA",
    "RELIGION": "RELIGION", "SCIENCE": "SCIENCE", "TECH": "TECH", "MONEY": "MONEY", "ARTS": "ARTS & CULTURE",
    "ARTS & CULTURE": "ARTS & CULTURE", "CULTURE & ARTS": "ARTS & CULTURE", "GREEN": "ENVIRONMENT",
    "ENVIRONMENT": "ENVIRONMENT", "COLLEGE": "EDUCATION", "EDUCATION": "EDUCATION",
}
HUFFPOST = {
    "POLITICS": ("Politics, government and elections", "politica", "Política, governo e eleições"),
    "WELLNESS": ("Wellness, healthy living and mental health", "bem_estar", "Bem-estar, vida saudável e saúde mental"),
    "ENTERTAINMENT": ("Entertainment, celebrities, movies and TV", "entretenimento", "Entretenimento, celebridades, filmes e TV"),
    "TRAVEL": ("Travel and tourism", "viagem", "Viagens e turismo"),
    "STYLE & BEAUTY": ("Fashion, style and beauty", "moda_e_beleza", "Moda, estilo e beleza"),
    "PARENTING": ("Parenting and raising children", "filhos", "Criação de filhos e maternidade"),
    "FOOD & DRINK": ("Food, recipes, restaurants and drinks", "comida_e_bebida", "Comida, receitas, restaurantes e bebidas"),
    "BUSINESS": ("Business, companies and the economy", "negocios", "Negócios, empresas e economia"),
    "SPORTS": ("Sports and athletes", "esportes", "Esportes e atletas"),
    "HOME & LIVING": ("Home, decoration and daily living", "casa", "Casa, decoração e vida doméstica"),
    "WEDDINGS": ("Weddings and engagements", "casamentos", "Casamentos e noivados"),
    "CRIME": ("Crime, police and courts", "crime", "Crimes, polícia e tribunais"),
    "DIVORCE": ("Divorce and separation", "divorcio", "Divórcio e separação"),
    "WORLD NEWS": ("International news from other countries", "mundo", "Notícias internacionais de outros países"),
    "MEDIA": ("News media, journalism and the press", "midia", "Mídia, jornalismo e imprensa"),
    "RELIGION": ("Religion and faith", "religiao", "Religião e fé"),
    "SCIENCE": ("Science and research", "ciencia", "Ciência e pesquisa"),
    "TECH": ("Technology, gadgets and the internet", "tecnologia", "Tecnologia, aparelhos e internet"),
    "MONEY": ("Personal finance and money", "dinheiro", "Finanças pessoais e dinheiro"),
    "ARTS & CULTURE": ("Arts, books, theater and culture", "arte_e_cultura", "Artes, livros, teatro e cultura"),
    "ENVIRONMENT": ("Environment, climate and nature", "meio_ambiente", "Meio ambiente, clima e natureza"),
    "EDUCATION": ("Education, schools and universities", "educacao", "Educação, escolas e universidades"),
}
NYT = {
    "Politics": ("Politics, government and elections", "politica", "Política, governo e eleições"),
    "Business and Finance": ("Business, companies, markets and the economy", "economia", "Economia, empresas, mercados e finanças"),
    "Sports": ("Sports, teams and athletes", "esporte", "Esportes, times e atletas"),
    "Science and Technology": ("Science, technology and the internet", "ciencia_e_tecnologia", "Ciência, tecnologia e internet"),
    "Health and Wellness": ("Health, medicine and wellness", "saude", "Saúde, medicina e bem-estar"),
    "Arts, Culture, and Entertainment": ("Arts, culture, music, film and TV", "cultura", "Arte, cultura, música, cinema e TV"),
    "Crime": ("Crime, police and the courts", "policia", "Crimes, polícia e justiça criminal"),
    "Lifestyle and Fashion": ("Lifestyle, fashion, food and travel", "estilo_de_vida", "Estilo de vida, moda, comida e viagens"),
}
SIB = {
    "science/technology": ("Science and technology", "ciencia_e_tecnologia", "Ciência e tecnologia"),
    "travel": ("Travel and tourism", "viagem", "Viagens e turismo"),
    "politics": ("Politics and government", "politica", "Política e governo"),
    "sports": ("Sports", "esportes", "Esportes"),
    "health": ("Health and medicine", "saude", "Saúde e medicina"),
    "entertainment": ("Entertainment and culture", "entretenimento", "Entretenimento e cultura"),
    "geography": ("Geography, places and nature", "geografia", "Geografia, lugares e natureza"),
}
BBC = {
    "business": ("Business and the economy", "economia", "Economia e negócios"),
    "sport": ("Sports", "esporte", "Esportes"),
    "politics": ("Politics and government", "politica", "Política e governo"),
    "tech": ("Technology", "tecnologia", "Tecnologia"),
    "entertainment": ("Entertainment, film and music", "entretenimento", "Entretenimento, cinema e música"),
}
NEWSGROUPS = {
    "alt.atheism": ("Atheism and religious debate", "ateismo", "Ateísmo e debate religioso"),
    "comp.graphics": ("Computer graphics", "computacao_grafica", "Computação gráfica"),
    "comp.os.ms-windows.misc": ("Microsoft Windows", "windows", "Microsoft Windows"),
    "comp.sys.ibm.pc.hardware": ("PC hardware", "hardware_pc", "Hardware de PC"),
    "comp.sys.mac.hardware": ("Apple Mac hardware", "hardware_mac", "Hardware Apple Mac"),
    "comp.windows.x": ("The X Window System on Unix", "x_window", "Sistema X Window no Unix"),
    "misc.forsale": ("Items for sale", "vendas", "Itens à venda"),
    "rec.autos": ("Cars", "carros", "Carros"),
    "rec.motorcycles": ("Motorcycles", "motos", "Motocicletas"),
    "rec.sport.baseball": ("Baseball", "beisebol", "Beisebol"),
    "rec.sport.hockey": ("Hockey", "hoquei", "Hóquei"),
    "sci.crypt": ("Cryptography and security", "criptografia", "Criptografia e segurança"),
    "sci.electronics": ("Electronics", "eletronica", "Eletrônica"),
    "sci.med": ("Medicine and health", "medicina", "Medicina e saúde"),
    "sci.space": ("Space and astronomy", "espaco", "Espaço e astronomia"),
    "soc.religion.christian": ("Christianity", "cristianismo", "Cristianismo"),
    "talk.politics.guns": ("Gun politics", "armas", "Política de armas"),
    "talk.politics.mideast": ("Middle East politics", "oriente_medio", "Política do Oriente Médio"),
    "talk.politics.misc": ("General politics", "politica", "Política em geral"),
    "talk.religion.misc": ("General religion", "religiao", "Religião em geral"),
}
AGNEWS = {
    "World": ("International news from other countries", "mundo", "Notícias internacionais de outros países"),
    "Sports": ("Sports", "esportes", "Esportes"),
    "Business": ("Business, companies and the economy", "negocios", "Negócios, empresas e economia"),
    "Sci/Tech": ("Science and technology", "ciencia_e_tecnologia", "Ciência e tecnologia"),
}
EMOTION = {
    "sadness": ("Sad or unhappy", "tristeza", "Triste ou infeliz"),
    "joy": ("Happy or joyful", "alegria", "Feliz ou alegre"),
    "love": ("Affection or love", "amor", "Afeto ou amor"),
    "anger": ("Angry or irritated", "raiva", "Com raiva ou irritado"),
    "fear": ("Afraid or anxious", "medo", "Com medo ou ansioso"),
    "surprise": ("Surprised", "surpresa", "Surpreso"),
}

CNN_BR = {
    "esportes": ("Sports", "esportes", "Esportes, futebol e atletas"),
    "nacional": ("National news from Brazil: cities, public safety and daily life", "nacional", "Notícias do Brasil: cidades, segurança e cotidiano"),
    "internacional": ("International news from other countries", "internacional", "Notícias internacionais de outros países"),
    "economia": ("Economy, business and markets", "economia", "Economia, empresas e mercado"),
    "politica": ("Politics, government and elections", "politica", "Política, governo e eleições"),
    "entretenimento": ("Entertainment, celebrities, film, TV and music", "entretenimento", "Entretenimento, celebridades, cinema, TV e música"),
    "tecnologia": ("Technology and the internet", "tecnologia", "Tecnologia e internet"),
    "saude": ("Health and medicine", "saude", "Saúde e medicina"),
}
BBC_BR = {
    "brasil": ("News about Brazil", "brasil", "Notícias sobre o Brasil"),
    "internacional": ("International news", "internacional", "Notícias internacionais"),
    "economia": ("Economy and business", "economia", "Economia e negócios"),
    "saude": ("Health and medicine", "saude", "Saúde e medicina"),
    "america_latina": ("News from Latin America", "america_latina", "Notícias da América Latina"),
    "ciencia": ("Science and research", "ciencia", "Ciência e pesquisa"),
}

WEB_NEWS = {
    "politica": ("Politics, government, Congress and elections", "politica", "Política, governo, Congresso e eleições"),
    "economia": ("Economy, companies, markets and personal finance", "economia", "Economia, empresas, mercado e finanças"),
    "mundo": ("International news from other countries", "internacional", "Notícias internacionais de outros países"),
    "esporte": ("Sports, football and athletes", "esportes", "Esportes, futebol e atletas"),
    "cidades": ("Cities, urban life, transport and local services", "cidades", "Cidades, vida urbana, transporte e serviços locais"),
    "policia": ("Crime, police and public safety", "policia", "Crimes, polícia e segurança pública"),
    "cultura": ("Culture and entertainment: film, music, TV and celebrities", "cultura", "Cultura e entretenimento: cinema, música, TV e famosos"),
    "tecnologia": ("Technology, internet, apps and games", "tecnologia", "Tecnologia, internet, aplicativos e games"),
    "saude": ("Health, medicine and wellbeing", "saude", "Saúde, medicina e bem-estar"),
    "ciencia": ("Science, research and space", "ciencia", "Ciência, pesquisa e espaço"),
    "educacao": ("Education, schools, universities and exams", "educacao", "Educação, escolas, universidades e vestibular"),
    "meio_ambiente": ("Environment, climate and nature", "meio_ambiente", "Meio ambiente, clima e natureza"),
    "turismo": ("Travel and tourism", "turismo", "Viagens e turismo"),
    "carros": ("Cars and the auto industry", "carros", "Carros e indústria automotiva"),
    "comida": ("Food, recipes and restaurants", "gastronomia", "Gastronomia, receitas e restaurantes"),
}

QUESTIONS = {
    "topic": {
        "en": ["What is the main topic of this text?", "Which category best describes this text?", "What is this text about?",
               "Classify the text into one of the categories."],
        "pt": ["Qual é o assunto principal deste texto?", "Qual categoria melhor descreve este texto?", "Sobre o que é este texto?",
               "Classifique o texto em uma das categorias."],
    },
    "news": {
        "en": ["Which section of the newspaper does this article belong to?", "What is the main category of this news article?",
               "Which desk should publish this story?", "What is this article mainly about?"],
        "pt": ["Qual é a categoria principal desta matéria?", "Em qual editoria esta notícia deve ser publicada?",
               "Qual é o assunto principal desta notícia?", "A qual seção do jornal esta matéria pertence?"],
    },
    "entity": {
        "en": ["What kind of entity does this text describe?", "Which type best describes the subject of this article?"],
        "pt": ["Que tipo de entidade este texto descreve?", "Qual tipo melhor descreve o assunto deste artigo?"],
    },
    "emotion": {
        "en": ["Which emotion does the text express?", "How does the author feel?"],
        "pt": ["Qual emoção o texto expressa?", "Como o autor se sente?"],
    },
}


def render_options(rng: random.Random, labels: list[tuple[str, str, str, str]], gold: int, max_k: int = 16):
    """Return (option strings, gold index) for a random presentation of the label set."""
    idx = list(range(len(labels)))
    k_all = len(idx)
    if k_all > max_k or rng.random() < 0.5:
        k = rng.randint(2, min(k_all, max_k))
        others = [i for i in idx if i != gold]
        rng.shuffle(others)
        idx = sorted([gold] + others[: k - 1])
    mode = rng.choices(["en", "pt", "bare_en", "bare_pt", "neutral", "mixed"], weights=[25, 25, 10, 10, 15, 15])[0]
    opts = []
    for j, i in enumerate(idx):
        key, en, pkey, pt = labels[i]
        if mode == "en":
            opts.append(f"{key}: {en}")
        elif mode == "pt":
            opts.append(f"{pkey}: {pt}")
        elif mode == "bare_en":
            opts.append(key)
        elif mode == "bare_pt":
            opts.append(pkey)
        elif mode == "neutral":
            tag = rng.choice(["opt_{}", "c{}", "{}", "label_{}"]).format(j + 1)
            opts.append(f"{tag}: {rng.choice([en, pt])}")
        else:
            opts.append(f"{rng.choice([key, pkey])}: {rng.choice([en, pt])}")
    names = [o.split(": ", 1)[0] for o in opts]
    if len(set(names)) != len(names):  # colliding keys (e.g. two Portuguese keys alike): fall back to English keys
        opts = [f"{labels[i][0]}: {labels[i][1]}" for i in idx]
    return opts, idx.index(gold)


def render_state(rng: random.Random, title: str | None, text: str, max_chars: int = 4000):
    text = (text or "").strip()[:max_chars]
    title = (title or "").strip()
    r = rng.random()
    if not title:
        return text
    if r < 0.2:
        return f"{title}\n\n{text}".strip()
    if r < 0.6:
        return {"title": title, "text": text}
    return {"titulo": title, "texto": text}


def make_choice(rng, source, rid, family, labels_map, gold_key, title, text, split="train"):
    keys = list(labels_map)
    labels = [(k, *labels_map[k]) for k in keys]
    opts, g = render_options(rng, labels, keys.index(gold_key))
    lang = rng.choice(["en", "pt"])
    q = rng.choice(QUESTIONS[family][lang])
    target = [0.0] * len(opts)
    target[g] = 1.0
    return make_row(source, rid, render_state(rng, title, text), q, "choice", opts, target, split, lang)


def _load(name, config=None, split="train"):
    from datasets import load_dataset

    return load_dataset(name, config, split=split)


def build(caps: dict[str, int], seed: int = 0) -> Iterator[dict]:
    rng = random.Random(seed)

    def take(ds, n):
        order = list(range(len(ds)))
        rng.shuffle(order)
        return ds.select(order[:n])

    ds = _load("fancyzhx/dbpedia_14")
    names = ds.features["label"].names
    for i, r in enumerate(take(ds, caps["dbpedia"])):
        yield make_choice(rng, "topic_dbpedia", f"dbpedia:{i}", "entity", DBPEDIA, names[r["label"]], r["title"], r["content"])
    ds = _load("community-datasets/yahoo_answers_topics")
    names = ds.features["topic"].names
    for i, r in enumerate(take(ds, caps["yahoo"])):
        text = f"{r['question_content'] or ''}\n\n{r['best_answer'] or ''}".strip()
        yield make_choice(rng, "topic_yahoo", f"yahoo:{i}", "topic", YAHOO, names[r["topic"]], r["question_title"], text)
    ds = _load("heegyu/news-category-dataset")
    n = 0
    for r in take(ds, len(ds)):
        cat = HUFFPOST_MAP.get(r["category"])
        if cat is None:
            continue
        yield make_choice(rng, "topic_huffpost", f"huffpost:{n}", "news", HUFFPOST, cat, r["headline"], r["short_description"])
        n += 1
        if n >= caps["huffpost"]:
            break
    for i, r in enumerate(take(_load("dstefa/New_York_Times_Topics"), caps["nyt"])):
        yield make_choice(rng, "topic_nyt", f"nyt:{i}", "news", NYT, r["topic_name"], None, r["text"])
    for i, r in enumerate(take(_load("SetFit/20_newsgroups"), caps["newsgroups"])):
        if len((r["text"] or "").strip()) < 40:
            continue
        yield make_choice(rng, "topic_20ng", f"20ng:{i}", "topic", NEWSGROUPS, r["label_text"], None, r["text"])
    for rep in range(caps["bbc_repeat"]):
        for i, r in enumerate(_load("SetFit/bbc-news")):
            yield make_choice(rng, "topic_bbc", f"bbc:{rep}:{i}", "news", BBC, r["label_text"], None, r["text"])
    for lang in ("por_Latn", "eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "ita_Latn"):
        for rep in range(caps["sib_repeat"]):
            for i, r in enumerate(_load("Davlan/sib200", lang)):
                yield make_choice(rng, "topic_sib200", f"sib:{lang}:{rep}:{i}", "topic", SIB, r["category"], None, r["text"])
    for rep in range(caps["ptnews_repeat"]):
        for i, r in enumerate(_load("celsowm/cnn_news_ptbr")):
            if r["categoria"] in CNN_BR and r["texto"]:
                yield make_choice(rng, "topic_cnnbr", f"cnnbr:{rep}:{i}", "news", CNN_BR, r["categoria"], r["titulo"], r["texto"])
        for i, r in enumerate(_load("celsowm/bbc_news_ptbr")):
            if r["categoria"] in BBC_BR and r["texto"]:
                yield make_choice(rng, "topic_bbcbr", f"bbcbr:{rep}:{i}", "news", BBC_BR, r["categoria"], r["titulo"], r["texto"])
    ag = _load("fancyzhx/ag_news")
    names = ag.features["label"].names
    for i, r in enumerate(take(ag, caps["agnews"])):
        yield make_choice(rng, "topic_agnews", f"ag:{i}", "news", AGNEWS, names[r["label"]], None, r["text"])
    em = _load("dair-ai/emotion")
    names = em.features["label"].names
    for i, r in enumerate(take(em, caps["emotion"])):
        yield make_choice(rng, "topic_emotion", f"emo:{i}", "emotion", EMOTION, names[r["label"]], None, r["text"])


def build_webnews(path: str, seed: int = 0) -> Iterator[dict]:
    """Rows from FineWeb-2 news pages whose section was read from the URL (see scripts/webnews.py)."""
    rng = random.Random(seed)
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        yield make_choice(rng, "topic_webnews", f"webnews:{i}", "news", WEB_NEWS, r["section"], r.get("title"), r["text"])


def diversify_row(rng: random.Random, row: dict) -> dict:
    """Random label subset and neutral ids for an existing choice row (bare-name label sets included)."""
    if row["type"] != "choice" or len(row["options"]) < 3:
        return row
    gold = max(range(len(row["target"])), key=row["target"].__getitem__)
    labels = []
    for o in row["options"]:
        key, sep, desc = o.partition(": ")
        human = key.replace("_", " ").strip()
        labels.append((key, desc if sep else human, key, desc if sep else human))
    opts, g = render_options(rng, labels, gold)
    target = [0.0] * len(opts)
    target[g] = 1.0
    return {**row, "options": opts, "target": target}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="groot-topics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--cap", action="append", default=[], help="name=N")
    w = sub.add_parser("webnews")
    w.add_argument("--raw", required=True)
    w.add_argument("--out", required=True)
    d = sub.add_parser("diversify")
    d.add_argument("--rows", required=True)
    d.add_argument("--sources", required=True, help="comma-separated source prefixes")
    d.add_argument("--max-rows", type=int, default=None)
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        caps = {"dbpedia": 30000, "yahoo": 40000, "huffpost": 50000, "nyt": 60000, "newsgroups": 11314, "bbc_repeat": 3,
                "sib_repeat": 2, "ptnews_repeat": 3, "agnews": 20000, "emotion": 10000}
        for c in args.cap:
            k, v = c.split("=")
            caps[k] = int(v)
        n = 0
        with Path(args.out).open("w", encoding="utf-8") as f:
            for row in build(caps, args.seed):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(json.dumps({"rows": n, "out": args.out}))
    elif args.cmd == "webnews":
        n = 0
        with Path(args.out).open("w", encoding="utf-8") as f:
            for row in build_webnews(args.raw):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(json.dumps({"rows": n, "out": args.out}))
    else:
        rng = random.Random(args.seed)
        prefixes = tuple(args.sources.split(","))
        n = 0
        with Path(args.out).open("w", encoding="utf-8") as f:
            for line in open(args.rows, encoding="utf-8"):
                row = json.loads(line)
                if not row["source"].startswith(prefixes):
                    continue
                f.write(json.dumps(diversify_row(rng, row), ensure_ascii=False) + "\n")
                n += 1
                if args.max_rows and n >= args.max_rows:
                    break
        print(json.dumps({"rows": n, "out": args.out}))


if __name__ == "__main__":
    main()
