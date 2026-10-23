"""Second pass: domain- and region-based sections for the weakest classes (health, tech, cities, science, education)."""
import json, sys, collections, urllib.parse
from datasets import load_dataset

BLOCK = ("folha", "uol.com.br")
DOMAINS = {
    "saude": ("minhavida.com.br", "tuasaude.com", "saude.abril.com.br", "saude.ig.com.br", "drauziovarella", "vidaesaude",
              "boasaude.com.br", "saudebemestar", "medicina", "msdmanuals.com/pt", "bemestar"),
    "tecnologia": ("tecmundo.com.br", "olhardigital.com.br", "canaltech.com.br", "tecnoblog.net", "showmetech.com.br",
                   "tudocelular.com", "techtudo.com.br", "meiobit.com", "b9.com.br"),
    "ciencia": ("revistagalileu.globo.com", "superinteressante", "super.abril.com.br", "hypescience.com", "revistapesquisa.fapesp.br",
                "jornal.usp.br", "agencia.fapesp.br"),
    "educacao": ("guiadoestudante.abril.com.br", "educacao.globo.com", "novaescola.org.br", "brasilescola.uol", "porvir.org"),
}
UFS = {"ac","al","ap","am","ba","ce","df","es","go","ma","mt","ms","mg","pa","pb","pr","pe","pi","rj","rn","rs","ro","rr","sc","sp","se","to",
       "sao-paulo","rio-de-janeiro","minas-gerais","bahia","pernambuco","parana","rio-grande-do-sul","distrito-federal","ceara","goias"}

def section(url):
    u = urllib.parse.urlparse(url); host = u.netloc.lower(); full = host + u.path.lower()
    if any(b in host for b in BLOCK):
        return None
    for s, ds in DOMAINS.items():
        if any(d in full for d in ds):
            return s
    segs = [x for x in u.path.lower().split("/") if x]
    if host.endswith("g1.globo.com") and segs and segs[0] in UFS and "politica" not in segs and "economia" not in segs:
        return "cidades"
    if host in ("oglobo.globo.com", "extra.globo.com") and segs and segs[0] in ("rio", "brasil", "sao-paulo"):
        return "cidades" if segs[0] != "brasil" else None
    return None

target, out = int(sys.argv[1]), sys.argv[2]
skip = set()
for l in open(sys.argv[3], encoding="utf-8"):
    skip.add(json.loads(l)["url"])
ds = load_dataset("HuggingFaceFW/fineweb-2", "por_Latn", split="train", streaming=True)
c = collections.Counter(); seen = kept = 0; cap = target // 5
with open(out, "w", encoding="utf-8") as f:
    for r in ds:
        seen += 1
        s = section(r["url"])
        if s and r["url"] not in skip and len(r["text"]) > 500 and c[s] < cap:
            lines = [l for l in r["text"].split("\n") if l.strip()]
            title = lines[0][:200] if lines and len(lines[0]) < 200 else ""
            body = "\n".join(lines[1:] if title else lines)[:4000]
            f.write(json.dumps({"url": r["url"], "section": s, "title": title, "text": body}, ensure_ascii=False) + "\n")
            c[s] += 1; kept += 1
        if seen % 500000 == 0:
            print(json.dumps({"seen": seen, "kept": kept, "by": dict(c)}), flush=True)
        if kept >= target or seen >= 15_000_000:
            break
print(json.dumps({"seen": seen, "kept": kept, "by": dict(c)}), flush=True)
