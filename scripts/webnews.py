"""Portuguese news pages from FineWeb-2 with the newsroom section read from the URL (no Folha/UOL)."""
import json, re, sys, collections, urllib.parse
from datasets import load_dataset

OUTLETS = ("g1.globo.com", "oglobo.globo.com", "globoesporte.globo.com", "ge.globo.com", "estadao.com.br", "r7.com",
           "terra.com.br", "correiobraziliense.com.br", "em.com.br", "gazetadopovo.com.br", "veja.abril.com.br",
           "exame.abril.com.br", "exame.com", "istoe.com.br", "cartacapital.com.br", "metropoles.com", "poder360.com.br",
           "cnnbrasil.com.br", "agenciabrasil.ebc.com.br", "zh.clicrbs.com.br", "gauchazh.clicrbs.com.br",
           "jornaldocommercio", "diariodepernambuco.com.br", "opovo.com.br", "otempo.com.br", "correio24horas.com.br",
           "acritica.com", "valor.globo.com", "epoca.globo.com", "extra.globo.com", "odia.ig.com.br", "ig.com.br",
           "noticias.r7.com", "bbc.com", "dw.com", "publico.pt", "dn.pt", "jn.pt", "expresso.pt", "observador.pt", "sapo.pt",
           "rtp.pt", "cmjornal.pt")
BLOCK = ("folha", "uol.com.br")
SECTIONS = {
    "politica": ("politica", "poder", "eleicoes", "eleicoes-2014", "eleicao", "politics"),
    "economia": ("economia", "negocios", "mercado", "mercados", "financas", "seu-dinheiro", "dinheiro", "agronegocio", "economy", "business"),
    "mundo": ("mundo", "internacional", "world", "america-latina"),
    "esporte": ("esporte", "esportes", "futebol", "desporto", "globoesporte", "ge", "sport", "sports"),
    "cidades": ("cidades", "cotidiano", "gerais", "metropole", "sociedade", "pais", "local"),
    "policia": ("policia", "seguranca", "crime", "justica"),
    "cultura": ("cultura", "pop-arte", "diversao", "diversao-e-arte", "entretenimento", "cinema", "musica", "televisao",
                "famosos", "celebridades", "artes", "pop", "gente", "tv"),
    "tecnologia": ("tecnologia", "tec", "tech", "link", "games", "informatica"),
    "saude": ("saude", "bem-estar", "bem-viver", "vida-e-estilo-saude", "medicina"),
    "ciencia": ("ciencia", "science", "espaco"),
    "educacao": ("educacao", "vestibular", "enem", "ensino"),
    "meio_ambiente": ("natureza", "meio-ambiente", "sustentabilidade", "ambiente", "clima"),
    "turismo": ("turismo", "viagem", "viagens"),
    "carros": ("carros", "autos", "automoveis", "autoesporte", "jornal-do-carro"),
    "comida": ("gastronomia", "receitas", "comida", "paladar", "comer-e-beber"),
}
KEY2SEC = {k: s for s, ks in SECTIONS.items() for k in ks}

def section(url):
    u = urllib.parse.urlparse(url)
    host = u.netloc.lower()
    if any(b in host for b in BLOCK) or not any(o in host for o in OUTLETS):
        return None
    sub = host.split(".")[0]
    if host.startswith(("globoesporte.", "ge.")):
        return "esporte"
    if sub in KEY2SEC and sub not in ("www",):
        return KEY2SEC[sub]
    segs = [s for s in u.path.lower().split("/") if s][:3]
    for s in segs:
        if s in KEY2SEC:
            return KEY2SEC[s]
    return None

target, out = int(sys.argv[1]), sys.argv[2]
ds = load_dataset("HuggingFaceFW/fineweb-2", "por_Latn", split="train", streaming=True)
c = collections.Counter(); seen = 0; kept = 0
per_cap = target // 8
with open(out, "w", encoding="utf-8") as f:
    for r in ds:
        seen += 1
        s = section(r["url"])
        if s and len(r["text"]) > 500 and c[s] < per_cap:
            lines = [l for l in r["text"].split("\n") if l.strip()]
            title = lines[0][:200] if lines and len(lines[0]) < 200 else ""
            body = "\n".join(lines[1:] if title else lines)[:4000]
            f.write(json.dumps({"url": r["url"], "section": s, "title": title, "text": body}, ensure_ascii=False) + "\n")
            c[s] += 1; kept += 1
        if seen % 200000 == 0:
            print(json.dumps({"seen": seen, "kept": kept, "by": dict(c)}), flush=True)
        if kept >= target or seen >= 12_000_000:
            break
print(json.dumps({"seen": seen, "kept": kept, "by": dict(c)}), flush=True)
