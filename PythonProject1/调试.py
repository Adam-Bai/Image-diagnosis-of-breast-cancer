import os, shutil, glob, re

ROOT = os.getcwd()
MAP = {
    "inbreast":    lambda p: "INbreast" in p or re.search(r"INbreast.*Release", p, re.I),
    "wbcd":        lambda p: re.search(r"(wdbc|wisconsin).*diagnostic", p, re.I) or p.endswith(("wdbc.data","wdbc.names")) or re.match(r"7\.csv$", os.path.basename(p)),
    "busi":        lambda p: "BUSI" in p,
    "metabric":    lambda p: re.search(r"metabric", p, re.I) or re.match(r"10brca_metabric", os.path.basename(p)),
    "cbis_ddsm":   lambda p: re.search(r"cbis[-_]?ddsm", p, re.I) or p.endswith(".tcia"),
    "rsna_mammo":  lambda p: re.search(r"rsna.*mammograph", p, re.I) or os.path.basename(p) in ("train.csv","test.csv"),
    "breakhis":    lambda p: re.search(r"breakhis", p, re.I),
    "tcga_brca":   lambda p: re.search(r"tcga.*brca", p, re.I) or "gdc_download" in p,
}

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d, exist_ok=True)

def classify_move():
    for name in MAP:
        ensure_dir(os.path.join(ROOT, name))
    entries = [os.path.join(ROOT, e) for e in os.listdir(ROOT) if e not in MAP and not e.startswith('.')]
    for e in entries:
        # 检索文件/目录内的线索
        target = None
        for k, rule in MAP.items():
            try:
                if os.path.isdir(e):
                    # 看目录名或内部若干文件
                    if rule(e): target = k
                    else:
                        samples = []
                        for pat in ("**/*.dcm","**/*.csv","**/*.txt","**/*.xls*","**/*.json"):
                            samples += glob.glob(os.path.join(e, pat), recursive=True)[:5]
                        if any(rule(s) for s in samples): target = k
                else:
                    if rule(e): target = k
            except Exception:
                pass
            if target: break
        if target:
            dest = os.path.join(ROOT, target, os.path.basename(e))
            print(f"→ {e}  ==>  {dest}")
            try:
                shutil.move(e, dest)
            except Exception as ex:
                print("   !!! 移动失败：", ex)

if __name__ == "__main__":
    classify_move()
    print("\nDone. 请检查各目标目录内部是否合理，再删掉多余空目录。")
