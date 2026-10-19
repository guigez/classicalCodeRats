"""
ETAPA 1 — INVENTARIO
====================

Responsabilidade unica: transformar `metadata.csv` + a pasta de imagens em um
manifesto canonico validado, e falhar ruidosamente se o dado estiver
inconsistente.

O manifesto e um CSV com uma linha por quadro e as colunas:

    filename      nome do arquivo (hash.jpg)
    filepath      caminho relativo a raiz de imagens
    subject_id    identificador do individuo (unidade de agrupamento)
    class_name    rotulo textual original
    label         rotulo inteiro, dependente da tarefa escolhida

Este modulo NAO decide particao, NAO carrega imagem e NAO conhece framework de
deep learning. Ele so responde "que dados existem e eles sao coerentes?".

Uso:
    python src/manifest.py --metadata metadata.csv --images_dir imagens \\
        --task multiclass --out manifest.csv --check_files
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Definicao de tarefa
# --------------------------------------------------------------------------
# Ordem fisiologica das intensidades. Importa por dois motivos: mantem o
# label inteiro interpretavel como ordinal, e garante que matriz de confusao e
# tabelas saiam sempre na mesma ordem, nao em ordem alfabetica.
CLASS_ORDER = ["warm-up", "moderate", "heavy", "severe"]

# O mapeamento de rotulo e a UNICA definicao de tarefa do projeto. Qualquer
# outro modulo que precise saber "quantas classes" deve importar daqui, para
# nao existirem duas versoes da verdade.
#
# Os quatro cenarios de agrupamento da secao 4.2 da dissertacao do Bernini,
# mais o cenario de 3 classes alternativo (uniao dos dois ultimos):
TASKS = {
    # 4 classes: cada estagio isolado.
    "multiclass": {"warm-up": 0, "moderate": 1, "heavy": 2, "severe": 3},

    # 3 classes (Bernini): une os dois PRIMEIROS. A distincao Baixo/Moderado
    # tem pouca relevancia clinica; agrupa-los simplifica sem perder os niveis
    # criticos. Classe rara continua sendo severe (4 portadores).
    "three_bernini": {"warm-up": 0, "moderate": 0, "heavy": 1, "severe": 2},

    # 3 classes (alternativo): une os dois ULTIMOS, na fronteira do MLSS.
    # ATENCAO: aqui a classe rara deixa de ser severe. Como heavy+severe viram
    # um rotulo so, a classe limitante passa a ser a que tem menos portadores
    # entre as tres — e severe deixa de existir como alvo isolado. "Sujeito
    # completo" para o split muda: com este mapeamento, os completos sao os 10
    # ratos que tem heavy (todos menos o 4), nao os 4 que tem severe. O split
    # complete_subject_paired continua fixo em {1,9,10,11} por decisao de
    # comparabilidade, mas isso e uma restricao, nao uma necessidade.
    "three_mlss": {"warm-up": 0, "moderate": 1, "heavy": 2, "severe": 2},

    # 2 classes: baixa vs alta severidade. warm-up e moderate usaram velocidade
    # fixa; heavy e severe foram individualizados em relacao ao MLSS. O corte e
    # nessa fronteira. Aqui todo rato tem as duas classes, entao "completo" =
    # todos os 11 — o split fixo em {1,9,10,11} e escolha de comparabilidade.
    "binary": {"warm-up": 0, "moderate": 0, "heavy": 1, "severe": 1},
}

REQUIRED_COLUMNS = {"filename", "mice_id", "class"}


def class_names_for(task: str) -> list[str]:
    """Nomes legiveis de cada rotulo inteiro da tarefa, em ordem.

    Deriva do mapeamento em TASKS, entao ha uma unica fonte de verdade. Para
    'three_bernini' devolve ['warm-up+moderate', 'heavy', 'severe']; para
    'three_mlss', ['warm-up', 'moderate', 'heavy+severe']; e assim por diante.
    Classes que compartilham rotulo sao unidas por '+', preservando a ordem
    fisiologica de CLASS_ORDER.
    """
    if task not in TASKS:
        raise ManifestError(f"tarefa '{task}' desconhecida. Opcoes: {list(TASKS)}")
    mapa = TASKS[task]
    por_label: dict[int, list[str]] = {}
    for classe in CLASS_ORDER:
        if classe in mapa:
            por_label.setdefault(mapa[classe], []).append(classe)
    return ["+".join(por_label[lab]) for lab in sorted(por_label)]


class ManifestError(RuntimeError):
    """Erro que invalida o manifesto. Sempre interrompe a execucao."""


# --------------------------------------------------------------------------
# Leitura e validacao
# --------------------------------------------------------------------------
def load_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    """Le o CSV bruto e confere apenas o schema minimo."""
    path = Path(metadata_csv)
    if not path.exists():
        raise ManifestError(f"metadata nao encontrado: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ManifestError(
            f"colunas obrigatorias ausentes em {path}: {sorted(missing)}. "
            f"Encontradas: {df.columns.tolist()}"
        )
    return df


def validate_metadata(df: pd.DataFrame) -> None:
    """Checagens que invalidam o dado. Todas levantam excecao.

    Cada uma corresponde a um modo de falha real:
      - nulo silencioso vira NaN em label e quebra treino la na frente;
      - filename repetido significa que o mesmo quadro pode cair em dois
        conjuntos, o que e vazamento;
      - filename mapeado para dois sujeitos ou duas classes significa que a
        chave nao identifica o quadro, e ai nada mais e confiavel.
    """
    problems: list[str] = []

    nulls = df[list(REQUIRED_COLUMNS)].isna().sum()
    if nulls.sum() > 0:
        problems.append(f"valores nulos: {nulls[nulls > 0].to_dict()}")

    n_dup = len(df) - df["filename"].nunique()
    if n_dup > 0:
        exemplos = df["filename"][df["filename"].duplicated()].unique()[:5]
        problems.append(f"{n_dup} filenames repetidos, ex.: {list(exemplos)}")

    amb = df.groupby("filename").agg(
        n_sub=("mice_id", "nunique"), n_cls=("class", "nunique")
    )
    n_amb_sub = int((amb["n_sub"] > 1).sum())
    n_amb_cls = int((amb["n_cls"] > 1).sum())
    if n_amb_sub:
        problems.append(f"{n_amb_sub} filenames associados a mais de um sujeito")
    if n_amb_cls:
        problems.append(f"{n_amb_cls} filenames associados a mais de uma classe")

    desconhecidas = sorted(set(df["class"].unique()) - set(CLASS_ORDER))
    if desconhecidas:
        problems.append(
            f"classes fora de CLASS_ORDER: {desconhecidas}. "
            f"Esperadas: {CLASS_ORDER}"
        )

    if problems:
        raise ManifestError("metadata invalido:\n  - " + "\n  - ".join(problems))


def apply_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Adiciona a coluna `label` conforme a tarefa escolhida."""
    if task not in TASKS:
        raise ManifestError(f"tarefa '{task}' desconhecida. Opcoes: {list(TASKS)}")
    out = df.copy()
    out["label"] = out["class_name"].map(TASKS[task]).astype(int)
    return out


# --------------------------------------------------------------------------
# Conferencia contra o disco
# --------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def index_images(images_dir: Path) -> tuple[dict, dict]:
    """Indexa TODAS as imagens sob images_dir, recursivamente.

    Devolve dois indices: por nome completo do arquivo e por nome sem
    extensao. A varredura e recursiva de proposito — nada garante que as
    imagens estejam soltas na raiz. Organizar por classe (`imagens/severe/`)
    ou por animal (`imagens/rato_03/`) e comum, e a versao anterior deste
    modulo quebrava nesses dois casos.
    """
    por_nome: dict[str, list[str]] = {}
    por_radical: dict[str, list[str]] = {}
    if not images_dir.exists():
        return por_nome, por_radical
    for f in images_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            rel = str(f.relative_to(images_dir))
            por_nome.setdefault(f.name, []).append(rel)
            por_radical.setdefault(f.stem, []).append(rel)
    return por_nome, por_radical


def resolve_paths(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    """Casa cada filename do metadata com o arquivo real em disco.

    Estrategia em duas tentativas: nome exato, depois nome sem extensao. A
    segunda cobre o caso em que um pre-processamento em lote trocou o formato
    (por exemplo, remocao de fundo que salvou .png sobre um metadata que diz
    .jpg) sem que ninguem atualizasse o CSV.

    Nome ambiguo — o mesmo arquivo em duas subpastas — e erro, nao aviso:
    nao ha como saber qual dos dois e o certo, e escolher em silencio pode
    associar quadros ao animal errado.
    """
    por_nome, por_radical = index_images(images_dir)
    if not por_nome:
        raise ManifestError(
            f"nenhuma imagem encontrada sob {images_dir.resolve()}.\n"
            f"  Verifique se o caminho esta certo e se voce esta rodando da "
            f"raiz do projeto (nao de dentro de src/)."
        )

    caminhos, faltando, ambiguos, por_radical_usados = [], [], [], 0
    for nome in df["filename"].astype(str):
        if nome in por_nome:
            alvos = por_nome[nome]
        elif Path(nome).stem in por_radical:
            alvos = por_radical[Path(nome).stem]
            por_radical_usados += 1
        else:
            faltando.append(nome); caminhos.append(None); continue
        if len(alvos) > 1:
            ambiguos.append((nome, alvos[:3]))
        caminhos.append(alvos[0])

    if ambiguos:
        detalhe = "\n".join(f"    {n} -> {a}" for n, a in ambiguos[:5])
        raise ManifestError(
            f"{len(ambiguos)} nome(s) de arquivo aparecem em mais de uma "
            f"subpasta de {images_dir}. Impossivel decidir qual e o certo:\n"
            f"{detalhe}"
        )

    if faltando:
        subpastas = sorted({str(Path(v[0]).parent) for v in por_nome.values()})[:8]
        exemplos_disco = sorted(por_nome)[:5]
        raise ManifestError(
            f"{len(faltando)} de {len(df)} arquivos do metadata nao foram "
            f"encontrados sob {images_dir.resolve()}.\n"
            f"  procurados (primeiros) : {faltando[:5]}\n"
            f"  em disco   (primeiros) : {exemplos_disco}\n"
            f"  imagens em disco       : {sum(len(v) for v in por_nome.values())}\n"
            f"  subpastas encontradas  : {subpastas}\n"
            f"  Se as listas acima parecem nomes de mundos diferentes, o "
            f"--images_dir esta apontando para o lugar errado."
        )

    if por_radical_usados:
        print(f"[aviso] {por_radical_usados} arquivo(s) casados pelo nome sem "
              f"extensao — o formato em disco difere do que o metadata diz.")

    out = df.copy()
    out["filepath"] = caminhos
    return out


def find_duplicate_content(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    """Procura arquivos byte-identicos sob nomes diferentes.

    Vetor de vazamento que checagem por nome nunca pega: o mesmo quadro
    exportado duas vezes cai em treino e em teste e infla o resultado.
    Custa uma leitura completa do dataset, entao fica atras de um flag.
    """
    por_hash: dict[str, list[str]] = defaultdict(list)
    for rel in df["filepath"]:
        h = hashlib.md5((images_dir / rel).read_bytes()).hexdigest()
        por_hash[h].append(rel)

    linhas = [
        {"md5": h, "n": len(v), "filepaths": " | ".join(sorted(v))}
        for h, v in por_hash.items()
        if len(v) > 1
    ]
    return pd.DataFrame(linhas).sort_values("n", ascending=False) if linhas else pd.DataFrame()


# --------------------------------------------------------------------------
# Construcao
# --------------------------------------------------------------------------
def build_manifest(
    metadata_csv: str | Path,
    images_dir: str | Path,
    task: str = "multiclass",
    check_files: bool = False,
) -> pd.DataFrame:
    """Pipeline completo da etapa 1."""
    images_dir = Path(images_dir)

    raw = load_metadata(metadata_csv)
    validate_metadata(raw)

    df = pd.DataFrame(
        {
            "filename": raw["filename"].astype(str),
            "filepath": raw["filename"].astype(str),  # provisorio
            "subject_id": raw["mice_id"].astype(int),
            "class_name": raw["class"].astype(str),
        }
    )
    df = apply_task(df, task)

    # Resolve contra o disco sempre que a pasta existir: subpasta por classe
    # ou por animal e arranjo comum, e assumir arquivo solto na raiz quebra.
    if images_dir.exists():
        df = resolve_paths(df, images_dir)
    elif check_files:
        raise ManifestError(f"pasta de imagens nao existe: {images_dir.resolve()}")

    ordem = {c: i for i, c in enumerate(CLASS_ORDER)}
    df = df.sort_values(
        ["subject_id", "class_name"], key=lambda s: s.map(ordem) if s.name == "class_name" else s
    ).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Relatorio
# --------------------------------------------------------------------------
def inventory_report(df: pd.DataFrame) -> pd.DataFrame:
    """Imprime o inventario e devolve a tabela sujeito x classe.

    A tabela cruzada nao e enfeite: e ela que mostra quais sujeitos nao
    carregam quais classes, que e a restricao que limita a particao na etapa 2.
    """
    presentes = [c for c in CLASS_ORDER if c in set(df["class_name"])]
    ct = pd.crosstab(df["subject_id"], df["class_name"]).reindex(columns=presentes, fill_value=0)

    print("=" * 68)
    print("INVENTARIO")
    print("=" * 68)
    print(f"quadros ...... {len(df)}")
    print(f"sujeitos ..... {df['subject_id'].nunique()}  -> {sorted(df['subject_id'].unique())}")
    print(f"classes ...... {len(presentes)}  -> {presentes}")
    print()
    print("distribuicao por classe:")
    for c in presentes:
        n = int((df["class_name"] == c).sum())
        print(f"  {c:<10s} {n:>6d}  ({100 * n / len(df):5.2f}%)")
    print()
    print("rotulos inteiros (tarefa aplicada):")
    for lab, grp in df.groupby("label"):
        print(f"  label {lab} <- {sorted(grp['class_name'].unique())}  ({len(grp)} quadros)")
    print()
    print("quadros por sujeito x classe:")
    print(ct.to_string())
    print()

    print("cobertura de classes por sujeito:")
    for c in presentes:
        sem = ct.index[ct[c] == 0].tolist()
        marca = "" if not sem else "   <-- limita a particao"
        print(f"  {c:<10s} ausente em {len(sem):>2d} sujeito(s): {sem}{marca}")
    print("=" * 68)
    return ct


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Etapa 1: constroi o manifesto canonico.")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--task", default="multiclass", choices=list(TASKS))
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--check_files", action="store_true",
                    help="Falha se algum quadro listado nao existir em disco.")
    ap.add_argument("--check_duplicates", action="store_true",
                    help="Procura arquivos byte-identicos (le o dataset inteiro).")
    args = ap.parse_args()

    try:
        df = build_manifest(args.metadata, args.images_dir, args.task, args.check_files)
    except ManifestError as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        return 1

    inventory_report(df)

    if args.check_duplicates:
        dups = find_duplicate_content(df, Path(args.images_dir))
        if len(dups):
            dups.to_csv("duplicates.csv", index=False)
            print(f"[ATENCAO] {len(dups)} grupos de arquivos byte-identicos. "
                  f"Detalhes em duplicates.csv")
        else:
            print("[ok] nenhum arquivo byte-identico duplicado.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[ok] manifesto salvo: {args.out}  ({len(df)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
