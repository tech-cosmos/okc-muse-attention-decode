"""Make the OKC leaderboard card from live `okc leaderboard` data.

Usage: python3 post/card/make_card.py   (writes post/okc-card-<date>.png, opens it, copies it to the clipboard)
"""
import datetime, json, pathlib, re, subprocess

ME = "tech-cosmos"
PROBLEMS = [  # (okc name, label on the card)
    ("muse_attention_decode", "attention decode"),
    ("muse_moe_w4a16", "moe w4a16 (4-bit)"),
    ("muse_bf16_proj", "bf16 projection"),
    ("muse_decode_layer", "whole decode layer"),
    ("muse_prefill_layer", "prefill layer"),
]
HERE = pathlib.Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ROW = re.compile(r"^\s*(\d+)\s+(\S+).*?([\d.]+)×\s+\S+\s+(\d+)\s*$")


def board(*args):
    return subprocess.run(["okc", "leaderboard", *args], capture_output=True, text=True).stdout.splitlines()


def parse(lines):
    rows = [ROW.match(l) for l in lines]
    rows = [m for m in rows if m]
    top = float(rows[0].group(3))
    mine = next((m for m in rows if m.group(2) == ME), None)
    return top, mine


def main():
    rows, runs = [], 0
    for name, label in PROBLEMS:
        top, mine = parse(board(name))
        if mine:  # skip problems we haven't entered
            rows.append([label, float(mine.group(3)), top, int(mine.group(1))])
            runs += int(mine.group(4))

    overall = next(l for l in board() if ME in l)
    rank = int(overall.split()[0])
    total = int(next(l for l in reversed(board("--all")) if l.strip()).split()[0])

    n = len(rows)
    faster = sum(1 for r in rows if r[1] > 1.0)
    today = datetime.date.today()
    date = today.strftime("%b %-d").lower()

    html = (HERE / "template.html").read_text()
    for k, v in {
        "DATE": date, "N": n, "RANK": rank, "TOTAL": f"{total:,}", "FASTER": faster,
        "SQUARES": '<span class="sq"></span>' * faster, "RUNS": runs, "ROWS": json.dumps(rows),
    }.items():
        html = html.replace("{{%s}}" % k, str(v))
    out_html = HERE / "card.html"
    out_html.write_text(html)

    png = HERE.parent / f"okc-card-{today.isoformat()}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=1600,900", "--virtual-time-budget=4000", f"--screenshot={png}",
                    f"file://{out_html}"], capture_output=True)
    subprocess.run(["osascript", "-e", f'set the clipboard to (read (POSIX file "{png}") as «class PNGf»)'], check=True)
    subprocess.run(["open", str(png)])

    print(f"overall #{rank} of {total:,} · {faster}/{n} faster than SGLang · {runs} judged runs")
    for label, s, top, r in rows:
        print(f"  {label:20s} {s:.3f}x  (#1 {top:.3f}x)  rank #{r}")
    print(png)


if __name__ == "__main__":
    main()
