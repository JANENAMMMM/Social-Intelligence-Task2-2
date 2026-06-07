"""
불만 클러스터 & BERTopic 토픽 네트워크 시각화
- 외부 노드: 클러스터 (크기 = 문장 수)
- 내부 노드: BERTopic 토픽 (크기 = 토픽 문장 수)
- 클러스터 간 엣지: 공존 리뷰 수
- 클러스터-토픽 엣지: 소속 관계
"""

import warnings; warnings.filterwarnings("ignore")
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

HERE    = Path(__file__).resolve().parent
CLUSTER = HERE / "clustered_complaints.csv"
TOPIC   = HERE / "topic_results.csv"
OUT1    = HERE / "network_cooccurrence.png"
OUT2    = HERE / "network_full.png"

def set_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

CLUSTER_INFO = {
    1: {"label": "쿠팡\n배송/포장",  "color": "#EF5350", "short": "C1"},
    2: {"label": "무게/외관\n디자인", "color": "#FF9800", "short": "C2"},
    3: {"label": "완성도\n+배터리",  "color": "#42A5F5", "short": "C3"},
    4: {"label": "카메라\n야간/고스트","color": "#66BB6A","short": "C4"},
    5: {"label": "가격\n/구성품",    "color": "#AB47BC", "short": "C5"},
}

TOPIC_INFO = {
    (1, 0): "CS/교환불만",
    (1, 1): "박스파손",
    (1, 2): "완충재없음",
    (1, 3): "쿠팡전반",
    (2, 0): "일반아쉬움",
    (2, 1): "스크래치/케이스",
    (2, 2): "무게/손목",
    (2, 3): "디자인실망",
    (3, 0): "제품완성도",
    (3, 1): "배터리/발열",
    (4, 0): "카메라야간/고스트",
    (4, 1): "S펜미지원",
    (5, 0): "가격부담",
    (5, 1): "결제/카드",
    (5, 2): "통신사약정",
    (5, 3): "구매후회",
    (5, 4): "충전기미포함",
}

def load():
    cluster = pd.read_csv(CLUSTER, encoding="utf-8-sig")
    topic   = pd.read_csv(TOPIC,   encoding="utf-8-sig")
    return cluster, topic

# ══════════════════════════════════════════════════════════
# 그래프 1: 클러스터 공존 네트워크 (깔끔한 버전)
# ══════════════════════════════════════════════════════════
def draw_cooccurrence(cluster: pd.DataFrame):
    valid = cluster[cluster["cluster"].isin(CLUSTER_INFO)]

    # 공존 쌍 계산
    review_clusters = valid.groupby("review_id")["cluster"].apply(set).reset_index()
    multi = review_clusters[review_clusters["cluster"].apply(len) > 1]

    pair_counts: dict = {}
    for clsets in multi["cluster"]:
        for a, b in combinations(sorted(clsets), 2):
            if a in CLUSTER_INFO and b in CLUSTER_INFO:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    # 클러스터 크기
    sizes = {cid: len(cluster[cluster["cluster"] == cid]) for cid in CLUSTER_INFO}

    G = nx.Graph()
    for cid, info in CLUSTER_INFO.items():
        G.add_node(cid, label=info["label"], size=sizes[cid], color=info["color"])

    max_co = max(pair_counts.values()) if pair_counts else 1
    for (a, b), w in pair_counts.items():
        G.add_edge(a, b, weight=w)

    set_font()
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_title("불만 클러스터 공존 네트워크\n(노드 크기 = 불만 문장 수 / 엣지 굵기 = 공존 리뷰 수)",
                 color="white", fontsize=15, fontweight="bold", pad=20)

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # 엣지
    edges = G.edges(data=True)
    for u, v, d in edges:
        w = d["weight"]
        lw = 1 + 8 * (w / max_co)
        alpha = 0.3 + 0.6 * (w / max_co)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=lw, alpha=alpha,
                               edge_color="white", ax=ax)
        # 엣지 레이블
        mid = ((pos[u][0] + pos[v][0]) / 2,
               (pos[u][1] + pos[v][1]) / 2)
        ax.text(mid[0], mid[1], str(w), fontsize=9,
                ha="center", va="center",
                color="#FFD54F", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#0D1117", ec="none", alpha=0.8))

    # 노드
    for cid in G.nodes():
        info  = CLUSTER_INFO[cid]
        x, y  = pos[cid]
        s     = sizes[cid]
        r     = 0.06 + 0.10 * (s / max(sizes.values()))
        circle = plt.Circle((x, y), r, color=info["color"], alpha=0.9, zorder=3)
        ax.add_patch(circle)

        ax.text(x, y + 0.01, f"{s}개", fontsize=11, ha="center", va="center",
                color="white", fontweight="bold", zorder=5)
        ax.text(x, y - r - 0.06, info["label"], fontsize=10,
                ha="center", va="top", color=info["color"], fontweight="bold", zorder=5)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    # 범례 (엣지 굵기 설명)
    for w, label in [(30, "약한 연결"), (80, "중간 연결"), (119, "강한 연결")]:
        lw = 1 + 8 * (w / max_co)
        ax.plot([], [], color="white", linewidth=lw, alpha=0.7, label=f"{label} (~{w}건)")
    ax.legend(loc="lower right", facecolor="#1A1F2C", edgecolor="gray",
              labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"저장: {OUT1}")

# ══════════════════════════════════════════════════════════
# 그래프 2: 클러스터 + 토픽 계층 네트워크
# ══════════════════════════════════════════════════════════
def draw_full_network(cluster: pd.DataFrame, topic: pd.DataFrame):
    set_font()
    fig, ax = plt.subplots(figsize=(26, 22), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_title("불만 클러스터 & 세부 토픽 계층 네트워크",
                 color="white", fontsize=18, fontweight="bold", pad=24)

    G = nx.Graph()

    sizes = {cid: len(cluster[cluster["cluster"] == cid]) for cid in CLUSTER_INFO}
    for cid, info in CLUSTER_INFO.items():
        G.add_node(f"C{cid}", node_type="cluster",
                   color=info["color"], size=sizes[cid], label=info["label"])

    for (cid, tid), tlabel in TOPIC_INFO.items():
        if cid not in CLUSTER_INFO:
            continue
        tkey = f"T{cid}_{tid}"
        tsub  = topic[(topic["cluster"] == cid) & (topic["topic"] == tid)]
        tsize = len(tsub)
        G.add_node(tkey, node_type="topic",
                   color=CLUSTER_INFO[cid]["color"], size=tsize, label=tlabel)
        G.add_edge(f"C{cid}", tkey, weight=tsize, etype="membership")

    valid = cluster[cluster["cluster"].isin(CLUSTER_INFO)]
    review_clusters = valid.groupby("review_id")["cluster"].apply(set).reset_index()
    multi = review_clusters[review_clusters["cluster"].apply(len) > 1]
    pair_counts: dict = {}
    for clsets in multi["cluster"]:
        for a, b in combinations(sorted(clsets), 2):
            if a in CLUSTER_INFO and b in CLUSTER_INFO:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    for (a, b), w in pair_counts.items():
        G.add_edge(f"C{a}", f"C{b}", weight=w, etype="cooccurrence")

    cluster_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "cluster"]
    topic_nodes   = [n for n, d in G.nodes(data=True) if d["node_type"] == "topic"]

    # 클러스터: 중심 반경 2.5의 원형
    CLUSTER_RADIUS = 2.5
    n_c = len(cluster_nodes)
    cluster_pos = {}
    for i, cn in enumerate(sorted(cluster_nodes)):
        angle = 2 * np.pi * i / n_c - np.pi / 2
        cluster_pos[cn] = (CLUSTER_RADIUS * np.cos(angle),
                           CLUSTER_RADIUS * np.sin(angle))

    # 토픽: 클러스터에서 바깥 방향으로 펼침
    TOPIC_DIST   = 1.5   # 클러스터에서 토픽까지 거리
    TOPIC_SPREAD = 0.55  # 토픽 간 각도 간격 (rad)
    topic_pos = {}
    for tn in topic_nodes:
        parts   = tn.split("_")
        ckey    = "C" + parts[0][1:]
        cx, cy  = cluster_pos[ckey]
        base_angle = np.arctan2(cy, cx)  # 클러스터 방향
        tprefix  = "T" + ckey[1:] + "_"
        c_topics = sorted([t for t in topic_nodes if t.startswith(tprefix)])
        n_t = len(c_topics)
        idx = c_topics.index(tn) if tn in c_topics else 0
        offset = (idx - (n_t - 1) / 2) * (TOPIC_SPREAD / max(n_t - 1, 1))
        angle  = base_angle + offset
        topic_pos[tn] = (cx + TOPIC_DIST * np.cos(angle),
                         cy + TOPIC_DIST * np.sin(angle))

    pos = {**cluster_pos, **topic_pos}

    # ── 공존 엣지 (클러스터 간) ────────────────────────────
    co_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("etype") == "cooccurrence"]
    max_co   = max((d["weight"] for _, _, d in co_edges), default=1)
    for u, v, d in co_edges:
        w  = d["weight"]
        lw = 1.0 + 7 * (w / max_co)
        alpha = 0.2 + 0.55 * (w / max_co)
        xu, yu = pos[u]; xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color="white", lw=lw, alpha=alpha, zorder=1,
                solid_capstyle="round")
        mid = ((xu + xv) / 2, (yu + yv) / 2)
        ax.text(mid[0], mid[1], f"{w}건", fontsize=11, ha="center", va="center",
                color="#FFD54F", fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", fc="#0D1117", ec="none", alpha=0.85))

    # ── 소속 엣지 (클러스터→토픽) ─────────────────────────
    mem_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("etype") == "membership"]
    for u, v, d in mem_edges:
        ckey  = u if u.startswith("C") else v
        tkey  = v if u.startswith("C") else u
        color = G.nodes[ckey]["color"]
        xu, yu = pos[ckey]; xv, yv = pos[tkey]
        ax.plot([xu, xv], [yu, yv], color=color, lw=1.5, alpha=0.35, zorder=1)

    # ── 토픽 노드 ──────────────────────────────────────────
    max_tsize = max((G.nodes[n]["size"] for n in topic_nodes), default=1)
    for tn in topic_nodes:
        nd   = G.nodes[tn]
        x, y = pos[tn]
        r    = 0.08 + 0.14 * (nd["size"] / max_tsize)
        circ = plt.Circle((x, y), r, color=nd["color"], alpha=0.75, zorder=4)
        ax.add_patch(circ)
        ax.text(x, y, str(nd["size"]), fontsize=10,
                ha="center", va="center", color="white", fontweight="bold", zorder=6)
        # 레이블: 노드 바깥쪽
        cx_ref, cy_ref = cluster_pos["C" + tn.split("_")[0][1:]]
        dx = x - cx_ref; dy = y - cy_ref
        norm = np.sqrt(dx**2 + dy**2) + 1e-9
        lx = x + (r + 0.12) * dx / norm
        ly = y + (r + 0.12) * dy / norm
        ax.text(lx, ly, nd["label"], fontsize=10.5,
                ha="center", va="center", color="white", fontweight="bold",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="#1A1F2C", ec=nd["color"],
                          linewidth=1.0, alpha=0.88))

    # ── 클러스터 노드 ──────────────────────────────────────
    max_csize = max(sizes.values())
    for cn in cluster_nodes:
        nd   = G.nodes[cn]
        x, y = pos[cn]
        r    = 0.22 + 0.22 * (nd["size"] / max_csize)
        circ = plt.Circle((x, y), r, color=nd["color"], alpha=0.95, zorder=5)
        ax.add_patch(circ)
        lbl  = nd["label"].replace("\n", "\n")
        ax.text(x, y + 0.06, f"{nd['size']}개", fontsize=14,
                ha="center", va="center", color="white", fontweight="bold", zorder=7)
        ax.text(x, y - 0.10, lbl, fontsize=12,
                ha="center", va="center", color="white", fontweight="bold", zorder=7)

    LIMS = 4.5
    ax.set_xlim(-LIMS, LIMS)
    ax.set_ylim(-LIMS, LIMS)
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=info["color"],
                       label=f"C{cid}: {info['label'].replace(chr(10),' ')}")
        for cid, info in CLUSTER_INFO.items()
    ]
    legend_patches += [
        plt.Line2D([0], [0], color="white", lw=4, alpha=0.7, label="공존 연결 (굵기=공존 건수)"),
        plt.Line2D([0], [0], color="gray",  lw=1.5, alpha=0.5, label="토픽 소속 연결"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1A1F2C", edgecolor="#444", labelcolor="white",
              fontsize=11, framealpha=0.95, borderpad=0.8)

    plt.tight_layout()
    plt.savefig(OUT2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"저장: {OUT2}")

def main():
    cluster, topic = load()
    draw_cooccurrence(cluster)
    draw_full_network(cluster, topic)
    print("완료")

if __name__ == "__main__":
    main()
